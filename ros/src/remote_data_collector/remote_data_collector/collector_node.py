"""
remote_data_collector: one configurable node that records rosbags on the machine it runs on.

Endpoints (all relative to the node, i.e. ``/<namespace>/<node_name>/...``; launch the node
with a unique name per host, e.g. ``mastcam_front`` -> ``/mastcam_front/record``):

* ``~/record``      action  remote_data_collector_interfaces/action/RecordBag
* ``~/status``      service remote_data_collector_interfaces/srv/GetStatus
* ``~/list_bags``   service remote_data_collector_interfaces/srv/ListBags
* ``~/delete_bag``  service remote_data_collector_interfaces/srv/DeleteBag

Recorded bags are served for download by a sidecar ``python3 -m http.server`` rooted at
``data_dir``. See the package README for the parameter reference.
"""

from dataclasses import dataclass, field
import os
import signal
import socket
import threading
import time
from typing import Dict, List, Optional

from builtin_interfaces.msg import Time as TimeMsg
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from remote_data_collector_interfaces.action import RecordBag
from remote_data_collector_interfaces.msg import BagInfo, RecordingStatus
from remote_data_collector_interfaces.srv import DeleteBag, GetStatus, ListBags

from .bag_store import BagRecord, BagStore
from .http_sidecar import HttpSidecar
from .naming import goal_id_to_hex, validate_bag_name, ValidationError
from .profiles import build_record_command, profiles_from_flat_params, RecordSpec, resolve
from .recorder import BagRecorder, RecorderError, reindex_bag, stop_orphan

# Why a recording ended.
END_CANCELED = 'canceled'
END_DURATION = 'duration reached'
END_SHUTDOWN = 'collector shutting down'
END_DIED = 'recorder exited unexpectedly'
END_DISK = 'disk space below stop_free_space_mb'
END_MAX = 'max_duration_sec reached'


@dataclass
class Recording:
    """Book-keeping for one in-flight goal."""

    goal_id: str
    bag_name: str
    spec: RecordSpec
    duration: float
    goal_handle: object
    state: str = RecordingStatus.STATE_STARTING
    recorder: Optional[BagRecorder] = None
    wake: threading.Event = field(default_factory=threading.Event)
    created: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return self.recorder.elapsed() if self.recorder else 0.0

    def to_msg(self) -> RecordingStatus:
        msg = RecordingStatus()
        msg.goal_id = self.goal_id
        msg.bag_name = self.bag_name
        msg.state = self.state
        msg.elapsed_sec = float(self.elapsed())
        msg.duration_sec = float(self.duration)
        if self.recorder:
            msg.size_bytes = int(self.recorder.size_bytes())
            msg.subscribed_topics = self.recorder.subscribed_topics()
        return msg


def _sanitize_node_name(name: str) -> str:
    out = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
    if not out or not (out[0].isalpha() or out[0] == '_'):
        out = 'n_' + out
    return out


class RemoteDataCollector(Node):
    """Action server that records rosbags locally and serves them over HTTP."""

    def __init__(self, node_name: str = 'remote_data_collector', **kwargs):
        super().__init__(
            node_name, automatically_declare_parameters_from_overrides=True, **kwargs)
        self._lock = threading.Lock()
        self._recordings: Dict[str, Recording] = {}
        self._shutting_down = threading.Event()

        # ---------------------------------------------------------------- parameters
        self.data_dir = self._param('data_dir', '~/remote_data_collector')
        self.default_bag_name = self._param('default_bag_name', self.get_name())
        self.default_profiles = list(self._param('default_profiles', Parameter.Type.STRING_ARRAY)
                                     or [])
        self.max_concurrent = int(self._param('max_concurrent_recordings', 1))
        self.max_duration = float(self._param('max_duration_sec', 0.0))
        self.min_free_mb = float(self._param('min_free_space_mb', 1024))
        self.stop_free_mb = float(self._param('stop_free_space_mb', 256))
        self.feedback_period = max(0.05, float(self._param('feedback_period_sec', 1.0)))
        self.startup_timeout = float(self._param('startup_timeout_sec', 20.0))
        self.stop_timeout = float(self._param('stop_timeout_sec', 60.0))
        self.ros2_executable = self._param('ros2_executable', 'ros2')
        self.record_args = list(self._param('record_args', Parameter.Type.STRING_ARRAY) or [])
        self.reindex_unfinished = bool(self._param('reindex_unfinished_bags', True))
        self.reindex_timeout = float(self._param('reindex_timeout_sec', 300.0))
        self.result_timeout = int(self._param('result_timeout_sec', 3600))
        http_enabled = bool(self._param('http.enabled', True))
        http_port = int(self._param('http.port', 8000))
        http_bind = self._param('http.bind', '0.0.0.0')
        http_host = self._param('http.advertise_host', '')

        flat = {k: p.value for k, p in self.get_parameters_by_prefix('profiles').items()}
        self.profiles = profiles_from_flat_params(flat)  # raises on a bad config -> fail fast
        for name in self.default_profiles:
            if name not in self.profiles:
                raise ValidationError(
                    f'default_profiles references unknown profile {name!r}; '
                    f'defined: {sorted(self.profiles) or "(none)"}')

        # ---------------------------------------------------------------- storage & http
        self.store = BagStore(self.data_dir)
        self.store.ensure()
        self._recover_after_crash()

        self.http: Optional[HttpSidecar] = None
        if http_enabled:
            self.http = HttpSidecar(self.store.data_dir, http_port, http_bind, http_host,
                                    log_path=self.store.logs_dir / 'http_sidecar.log')
            if self.http.start():
                self.get_logger().info(f'HTTP sidecar serving {self.store.data_dir} at '
                                       f'{self.http.base_url}')
            else:
                self.get_logger().error(f'HTTP sidecar failed to start: {self.http.last_error}')

        # ---------------------------------------------------------------- ROS endpoints
        self._action_group = ReentrantCallbackGroup()
        self._service_group = ReentrantCallbackGroup()
        self._timer_group = MutuallyExclusiveCallbackGroup()
        base = self.get_name()  # relative -> /<namespace>/<node_name>/<endpoint>
        self._action_server = ActionServer(
            self, RecordBag, f'{base}/record',
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._action_group,
            result_timeout=self.result_timeout)
        self.create_service(GetStatus, f'{base}/status', self._on_status,
                            callback_group=self._service_group)
        self.create_service(ListBags, f'{base}/list_bags', self._on_list,
                            callback_group=self._service_group)
        self.create_service(DeleteBag, f'{base}/delete_bag', self._on_delete,
                            callback_group=self._service_group)
        if self.http:
            self.create_timer(5.0, self._supervise_http, callback_group=self._timer_group)
        # With rclpy's own signal handling disabled (see main), the executor can block in
        # rcl_wait indefinitely; Python signal handlers only run once it wakes. Wake it often.
        self.create_timer(0.25, lambda: None, callback_group=self._timer_group)

        ns = self.get_fully_qualified_name()
        self.get_logger().info(
            f'remote_data_collector ready: action {ns}/record, services {ns}/status, '
            f'{ns}/list_bags, {ns}/delete_bag | data_dir={self.store.data_dir} | '
            f'profiles={sorted(self.profiles)} default={self.default_profiles}')

    # ==================================================================== helpers
    def _param(self, name: str, default):
        """Return a parameter value, declaring it with `default` if it was not overridden."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        try:
            value = self.get_parameter(name).value
        except ParameterUninitializedException:
            return None  # declared by type only (e.g. an optional list) and never set
        if value is None and not isinstance(default, Parameter.Type):
            return default
        return value

    def _log_ok(self, ok: bool, msg: str) -> None:
        # rclpy forbids one call site from logging at different severities, so keep two.
        if ok:
            self.get_logger().info(msg)
        else:
            self.get_logger().error(msg)

    def _active_names(self) -> List[str]:
        with self._lock:
            return [r.bag_name for r in self._recordings.values()]

    def _bag_info_msg(self, record: BagRecord) -> BagInfo:
        msg = BagInfo()
        msg.name = record.name
        msg.path = str(record.path)
        msg.download_url = self.http.url_for(record.name + '/') if self.http else ''
        msg.files = list(record.files)
        msg.file_sizes = [int(s) for s in record.file_sizes]
        msg.size_bytes = int(record.size_bytes)
        msg.complete = bool(record.complete)
        msg.active = bool(record.active)
        secs = int(record.modified)
        msg.modified = TimeMsg(sec=secs, nanosec=int((record.modified - secs) * 1e9))
        return msg

    def _free_mb(self) -> float:
        free, _total = self.store.disk_usage()
        return free / (1024.0 * 1024.0)

    def _recover_after_crash(self) -> None:
        """Stop recorders orphaned by a previous run and repair their bags."""
        for name, info in self.store.active_markers().items():
            pid = int(info.get('pid', 0) or 0)
            bag_path = info.get('bag_path') or str(self.store.data_dir / name)
            if pid and stop_orphan(pid, bag_path, timeout=self.stop_timeout):
                self.get_logger().warn(f'Stopped orphaned recorder (pid {pid}) for bag {name}')
            if os.path.isdir(bag_path) and not os.path.exists(
                    os.path.join(bag_path, 'metadata.yaml')) and self.reindex_unfinished:
                ok = reindex_bag(bag_path, self.ros2_executable, self.reindex_timeout,
                                 log_path=self.store.logs_dir / f'{name}.log')
                self.get_logger().warn(
                    f'Bag {name} was not finalized by a previous run; reindex '
                    f'{"succeeded" if ok else "FAILED"}')
            self.store.clear_active(name)

    def _supervise_http(self) -> None:
        if self.http and not self.http.is_alive() and not self._shutting_down.is_set():
            if self.http.ensure_running():
                self.get_logger().warn(f'HTTP sidecar restarted (restart #{self.http.restarts})')
            elif self.http.last_error:
                self.get_logger().error(f'HTTP sidecar down: {self.http.last_error}',
                                        throttle_duration_sec=60.0)

    # ==================================================================== action
    def _on_goal(self, goal_request) -> GoalResponse:
        # Validation happens in _execute so the client receives an explanatory message.
        if self._shutting_down.is_set():
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle) -> CancelResponse:
        gid = goal_id_to_hex(goal_handle.goal_id.uuid)
        with self._lock:
            rec = self._recordings.get(gid)
        if rec:
            self.get_logger().info(f'Stop requested (cancel) for {rec.bag_name}')
            rec.wake.set()
        return CancelResponse.ACCEPT

    def _finish(self, goal_handle, terminal: str, result):
        """Move the goal to a terminal state; tolerate races with cancel requests."""
        try:
            if terminal == 'succeed':
                goal_handle.succeed()
            elif terminal == 'canceled' and goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.abort()
        except Exception as exc:  # noqa: B902 - invalid transition must not lose the result
            self.get_logger().warn(f'Could not set terminal goal state {terminal}: {exc}')
        return result

    def _execute(self, goal_handle):
        goal = goal_handle.request
        gid = goal_id_to_hex(goal_handle.goal_id.uuid)
        result = RecordBag.Result()
        if goal_handle.is_cancel_requested:
            result.message = 'Canceled before recording started'
            return self._finish(goal_handle, 'canceled', result)

        # ---------------------------------------------------- validate the request
        try:
            prefix = goal.bag_name.strip() or self.default_bag_name
            validate_bag_name(prefix)
            spec = resolve(self.profiles, list(goal.profiles) or self.default_profiles,
                           list(goal.topics), goal.regex.strip())
        except ValidationError as exc:
            result.message = f'Rejected: {exc}'
            self.get_logger().error(result.message)
            return self._finish(goal_handle, 'abort', result)
        duration = float(goal.duration)
        if self.max_duration > 0 and (duration <= 0 or duration > self.max_duration):
            duration = self.max_duration

        # ---------------------------------------------------- reserve a slot
        with self._lock:
            reason = ''
            if self._shutting_down.is_set():
                reason = 'collector is shutting down'
            elif len(self._recordings) >= self.max_concurrent:
                busy = ', '.join(r.bag_name for r in self._recordings.values())
                reason = (f'busy: {len(self._recordings)} recording(s) already active ({busy}); '
                          f'max_concurrent_recordings={self.max_concurrent}. Stop it first.')
            elif self._free_mb() < self.min_free_mb:
                reason = (f'only {self._free_mb():.0f} MiB free in {self.store.data_dir} '
                          f'(min_free_space_mb={self.min_free_mb:.0f})')
            if not reason:
                try:
                    bag_name = self.store.unique_bag_name(
                        prefix, reserved=[r.bag_name for r in self._recordings.values()])
                except ValidationError as exc:
                    reason = str(exc)
            if reason:
                result.message = f'Rejected: {reason}'
                self.get_logger().error(result.message)
                return self._finish(goal_handle, 'abort', result)
            rec = Recording(goal_id=gid, bag_name=bag_name, spec=spec, duration=duration,
                            goal_handle=goal_handle)
            self._recordings[gid] = rec
        try:
            try:
                return self._run(rec, result)
            except Exception as exc:  # noqa: B902 - stop the recorder whatever went wrong
                self.get_logger().error(f'Recording {rec.bag_name} failed: {exc!r}')
                if rec.recorder is not None:
                    rec.recorder.stop(self.stop_timeout)
                result.success = False
                result.message = f'Internal error while recording {rec.bag_name}: {exc!r}'
                try:
                    self._fill_result(rec, result)
                    result.success = (self.store.bag_path(rec.bag_name) /
                                      'metadata.yaml').is_file()
                except Exception:  # noqa: B902 - best effort
                    pass
                return self._finish(goal_handle, 'abort', result)
        finally:
            with self._lock:
                self._recordings.pop(gid, None)
            self.store.clear_active(rec.bag_name)

    def _publish_feedback(self, rec: Recording) -> None:
        feedback = RecordBag.Feedback()
        feedback.status = rec.to_msg()
        try:
            rec.goal_handle.publish_feedback(feedback)
        except Exception as exc:  # noqa: B902 - feedback is best effort
            self.get_logger().debug(f'feedback publish failed: {exc}')

    def _run(self, rec: Recording, result):
        goal_handle = rec.goal_handle
        bag_path = self.store.bag_path(rec.bag_name)
        log_path = self.store.log_path(rec.bag_name)
        recorder_node = _sanitize_node_name(f'rdc_{self.get_name()}_{rec.goal_id[:8]}')
        cmd = build_record_command(rec.spec, str(bag_path), recorder_node,
                                   self.ros2_executable, self.record_args)
        rec.recorder = BagRecorder(cmd, bag_path, log_path)
        self.get_logger().info(
            f'Starting bag {rec.bag_name} ({rec.spec.describe()}; '
            f'duration={"until stopped" if rec.duration <= 0 else f"{rec.duration:.1f}s"})')
        try:
            rec.recorder.start()
        except RecorderError as exc:
            result.message = f'Failed to start recorder: {exc}'
            self.get_logger().error(result.message)
            return self._finish(goal_handle, 'abort', result)
        self.store.mark_active(rec.bag_name, {
            'pid': rec.recorder.pid, 'goal_id': rec.goal_id, 'bag_path': str(bag_path),
            'started': time.time()})
        self._publish_feedback(rec)

        def abort_check():
            return goal_handle.is_cancel_requested or self._shutting_down.is_set()

        if not rec.recorder.wait_until_recording(self.startup_timeout, abort_check):
            running = rec.recorder.is_running()
            rec.state = RecordingStatus.STATE_STOPPING
            rec.recorder.stop(self.stop_timeout)
            if abort_check():
                end = END_CANCELED if goal_handle.is_cancel_requested else END_SHUTDOWN
                return self._complete(rec, result, end)
            why = (f'did not start recording within {self.startup_timeout:.0f}s' if running
                   else f'exited with code {rec.recorder.returncode}')
            result.message = (f'Recorder {why}. Log tail:\n{rec.recorder.log_tail()}')
            self.get_logger().error(result.message)
            self._fill_result(rec, result)
            result.success = False
            return self._finish(goal_handle, 'abort', result)

        rec.state = RecordingStatus.STATE_RECORDING
        self.get_logger().info(f'Recording {rec.bag_name} -> {bag_path}')
        end = ''
        while not end:
            self._publish_feedback(rec)
            rec.wake.wait(self.feedback_period)
            rec.wake.clear()
            elapsed = rec.recorder.elapsed()
            if goal_handle.is_cancel_requested:
                end = END_CANCELED
            elif self._shutting_down.is_set():
                end = END_SHUTDOWN
            elif not rec.recorder.is_running():
                end = END_DIED
            elif rec.duration > 0 and elapsed >= rec.duration:
                clamped = rec.duration != float(goal_handle.request.duration)
                end = END_MAX if clamped else END_DURATION
            elif self._free_mb() < self.stop_free_mb:
                end = END_DISK
        return self._complete(rec, result, end)

    def _fill_result(self, rec: Recording, result) -> None:
        subscribed = rec.recorder.subscribed_topics() if rec.recorder else []
        result.recorded_sec = float(rec.recorder.elapsed()) if rec.recorder else 0.0
        result.subscribed_topics = subscribed
        result.missing_topics = [t for t in rec.spec.topics if t not in subscribed]
        if self.http:
            result.log_url = self.http.url_for(
                self.store.relative_url_path(self.store.log_path(rec.bag_name)))
        try:
            result.bag = self._bag_info_msg(self.store.bag_record(rec.bag_name))
        except (OSError, ValidationError):
            result.bag = BagInfo(name=rec.bag_name)

    def _complete(self, rec: Recording, result, end: str):
        """Stop the recorder, finalize the bag and set the goal's terminal state."""
        rec.state = RecordingStatus.STATE_STOPPING
        self._publish_feedback(rec)
        self.get_logger().info(f'Stopping {rec.bag_name}: {end}')
        rc = rec.recorder.stop(self.stop_timeout)
        bag_path = self.store.bag_path(rec.bag_name)
        finalized = (bag_path / 'metadata.yaml').is_file()
        notes = []
        if not finalized and bag_path.is_dir() and self.reindex_unfinished:
            finalized = reindex_bag(bag_path, self.ros2_executable, self.reindex_timeout,
                                    log_path=self.store.log_path(rec.bag_name))
            notes.append('metadata regenerated with `ros2 bag reindex`' if finalized
                         else '`ros2 bag reindex` failed')
        self._fill_result(rec, result)
        result.success = bool(finalized)
        if rc not in (0, None):
            notes.append(f'recorder exit code {rc}')
        if result.missing_topics:
            notes.append(f'{len(result.missing_topics)} requested topic(s) never subscribed: '
                         + ', '.join(result.missing_topics))
        if not finalized:
            notes.append('bag NOT finalized (no metadata.yaml); log tail:\n'
                         + rec.recorder.log_tail())
        result.message = f'{rec.bag_name}: stopped ({end}) after {result.recorded_sec:.1f}s, ' \
            f'{result.bag.size_bytes} bytes' + (''.join(f'; {n}' for n in notes))
        self._log_ok(finalized, result.message)

        if end == END_CANCELED:
            terminal = 'canceled'
        elif end in (END_DURATION, END_MAX) and finalized:
            terminal = 'succeed'
        else:
            terminal = 'abort'
        return self._finish(rec.goal_handle, terminal, result)

    # ==================================================================== services
    def _on_status(self, request, response):
        response.collector_name = self.get_fully_qualified_name()
        response.hostname = socket.gethostname()
        response.data_dir = str(self.store.data_dir)
        response.http_url = self.http.base_url if self.http else ''
        response.http_alive = bool(self.http and self.http.is_alive())
        free, total = self.store.disk_usage()
        response.disk_free_bytes = int(free)
        response.disk_total_bytes = int(total)
        response.profiles = sorted(self.profiles)
        response.default_profiles = list(self.default_profiles)
        response.max_concurrent_recordings = int(self.max_concurrent)
        with self._lock:
            recs = list(self._recordings.values())
        response.active = [r.to_msg() for r in recs]
        return response

    def _on_list(self, request, response):
        try:
            records = self.store.list_bags(request.pattern.strip(), self._active_names())
            response.bags = [self._bag_info_msg(r) for r in records]
            response.success = True
            response.message = f'{len(records)} bag(s)'
        except OSError as exc:
            response.success = False
            response.message = f'Failed to list bags: {exc}'
        return response

    def _on_delete(self, request, response):
        name = request.name.strip()
        try:
            response.freed_bytes = int(self.store.delete(name, self._active_names()))
            response.success = True
            response.message = f'Deleted {name}'
            self.get_logger().info(f'Deleted bag {name} ({response.freed_bytes} bytes)')
        except (ValidationError, FileNotFoundError, OSError) as exc:
            response.success = False
            response.message = f'Not deleted: {exc}'
            self.get_logger().warn(response.message)
        return response

    # ==================================================================== shutdown
    def shutdown_recordings(self, timeout: Optional[float] = None) -> bool:
        """Stop all recordings gracefully and wait for them. Return True if all finished."""
        self._shutting_down.set()
        with self._lock:
            recs = list(self._recordings.values())
        for rec in recs:
            rec.wake.set()
        deadline = time.monotonic() + (timeout if timeout is not None
                                       else self.shutdown_budget())
        while time.monotonic() < deadline:
            with self._lock:
                if not self._recordings:
                    return True
            time.sleep(0.1)
        return False

    def shutdown_budget(self) -> float:
        """Return worst-case seconds to finalize recordings (SIGINT/TERM/KILL + reindex)."""
        reindex = self.reindex_timeout if self.reindex_unfinished else 0.0
        return self.stop_timeout + 20.0 + reindex + 15.0

    def signal_recorders(self, sig: int) -> None:
        """Send `sig` to every running recorder without waiting (used for a forced exit)."""
        with self._lock:
            recorders = [r.recorder for r in self._recordings.values() if r.recorder]
        for recorder in recorders:
            if recorder.is_running():
                recorder.signal(sig)

    def destroy_node(self):
        self.shutdown_recordings()
        if self.http:
            self.http.stop()
        return super().destroy_node()


def main(args=None):
    """Entry point: spin the collector; on SIGINT/SIGTERM finalize recordings, then exit."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = RemoteDataCollector()
    except Exception as exc:  # noqa: B902 - report configuration errors clearly
        print(f'[remote_data_collector] startup failed: {exc}', flush=True)
        rclpy.try_shutdown()
        raise SystemExit(1)

    executor = MultiThreadedExecutor(num_threads=max(4, node.max_concurrent + 4))
    executor.add_node(node)
    shutdown_started = threading.Event()

    def graceful_shutdown():
        node.get_logger().info('Shutdown requested: finalizing active recordings...')
        if not node.shutdown_recordings():
            node.get_logger().error('Timed out waiting for recordings to finalize')
        time.sleep(0.5)  # let final results reach any waiting clients
        executor.shutdown(timeout_sec=1.0)

    def on_signal(signum, _frame):
        if shutdown_started.is_set() and signum == signal.SIGTERM:
            node.get_logger().warn('SIGTERM during shutdown: still finalizing recordings '
                                   '(press Ctrl-C again to force exit)')
            return
        if shutdown_started.is_set():
            # Second signal: exit now. Recorders live in their own sessions, so ask each to
            # finalize on its own first; the next start's crash recovery tidies up the rest.
            node.signal_recorders(signal.SIGINT)
            if node.http:
                node.http.stop(timeout=1.0)
            os._exit(1)
        shutdown_started.set()
        threading.Thread(target=graceful_shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
