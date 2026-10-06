"""
Perception ground control: the "master client" for any number of remote_data_collectors.

Ground control does not record anything itself. It starts/stops recordings on the configured
remotes (RecordBag action goals), downloads finished bags over each remote's HTTP sidecar into
a local session directory, and optionally deletes them on the remote afterwards.

Services (call them with scripts/collect.sh, `rdc gc ...` or `ros2 service call`):

* ``~/start``        remote_data_collector_interfaces/srv/StartCollection
* ``~/stop``         remote_data_collector_interfaces/srv/StopCollection
* ``~/status``       std_srvs/srv/Trigger   (multi-line summary in `message`)
* ``~/new_session``  std_srvs/srv/Trigger   (new local session directory)

Timed recordings (duration > 0) are downloaded automatically when they finish; open-ended
ones are downloaded when ~/stop is called. Stopping is stateless (uses the action cancel
protocol), so ground control can be restarted mid-trial and still stop/collect recordings.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from remote_data_collector.client import (CollectorError, RecordingOutcome,
                                          RemoteCollectorClient, StartedRecording)
from remote_data_collector.download import DownloadError, format_bytes
from remote_data_collector_interfaces.srv import StartCollection, StopCollection
from std_srvs.srv import Trigger

from .fleet import (FleetConfigError, groups_from_params, RemoteConfig,
                    remotes_from_flat_params, resolve_targets)

JOB_QUEUED, JOB_DOWNLOADING, JOB_DONE, JOB_FAILED, JOB_SKIPPED = (
    'queued', 'downloading', 'done', 'FAILED', 'not downloaded')


@dataclass
class DownloadJob:
    """One finished bag and what happened to it locally."""

    remote: str
    goal_id: str
    bag_name: str
    session_dir: Path
    state: str = JOB_QUEUED
    done_bytes: int = 0
    total_bytes: int = 0
    local_path: str = ''
    error: str = ''
    future: Optional[Future] = None
    finished: threading.Event = field(default_factory=threading.Event)
    reported: bool = False  # included in a ~/stop reply already

    def describe(self) -> str:
        if self.state == JOB_DOWNLOADING and self.total_bytes:
            pct = 100.0 * self.done_bytes / self.total_bytes
            return (f'{self.state} {pct:.0f}% ({format_bytes(self.done_bytes)} / '
                    f'{format_bytes(self.total_bytes)})')
        if self.state == JOB_DONE:
            return f'{self.state} -> {self.local_path}'
        if self.state in (JOB_FAILED, JOB_SKIPPED):
            return f'{self.state}: {self.error}'
        return self.state


class GroundControl(Node):
    """Orchestrate remote_data_collectors and collect their bags."""

    def __init__(self, **kwargs):
        super().__init__('ground_control', automatically_declare_parameters_from_overrides=True,
                         **kwargs)
        # ---------------------------------------------------------------- parameters
        self.data_root = Path(self._param('data_root', '/mnt/e/perception-data')).expanduser()
        self.session_format = self._param('session_format', '%m%d%Y/%H-%M-%S')
        self.default_group = self._param('default_group', 'trial')
        self.start_timeout = float(self._param('start_timeout_sec', 30.0))
        self.stop_timeout = float(self._param('stop_timeout_sec', 180.0))
        self.http_timeout = float(self._param('download_socket_timeout_sec', 60.0))
        self.download_retries = int(self._param('download_retries', 3))
        workers = int(self._param('max_parallel_downloads', 3))

        flat_remotes = {k: p.value for k, p in self.get_parameters_by_prefix('remotes').items()}
        flat_groups = {k: p.value for k, p in self.get_parameters_by_prefix('groups').items()}
        self.remotes: Dict[str, RemoteConfig] = remotes_from_flat_params(flat_remotes)
        self.groups: Dict[str, List[str]] = groups_from_params(flat_groups, self.remotes)
        if not self.remotes:
            self.get_logger().warn('No remotes configured (parameters remotes.<name>.*)')
        if self.default_group and self.default_group not in self.groups \
                and self.default_group not in self.remotes:
            raise FleetConfigError(f'default_group {self.default_group!r} is not a group or '
                                   'remote')

        # ---------------------------------------------------------------- state
        self._lock = threading.RLock()
        self._client_group = ReentrantCallbackGroup()
        self._service_group = ReentrantCallbackGroup()
        self.collectors = {name: RemoteCollectorClient(self, cfg.node, self._client_group)
                           for name, cfg in self.remotes.items()}
        self._active: Dict[str, Dict[str, StartedRecording]] = {n: {} for n in self.remotes}
        self._goal_sessions: Dict[Tuple[str, str], Path] = {}
        self.jobs: Dict[Tuple[str, str], DownloadJob] = {}
        self._downloads = ThreadPoolExecutor(max_workers=max(1, workers),
                                             thread_name_prefix='gc_download')
        self._fanout = ThreadPoolExecutor(max_workers=max(4, len(self.remotes)),
                                          thread_name_prefix='gc_fanout')
        # Status gets its own pool so it stays responsive during a long stop.
        self._status_pool = ThreadPoolExecutor(max_workers=max(2, len(self.remotes)),
                                               thread_name_prefix='gc_status')
        self.session_dir = self._new_session_dir()

        # ---------------------------------------------------------------- services
        self.create_service(StartCollection, '~/start', self._on_start,
                            callback_group=self._service_group)
        self.create_service(StopCollection, '~/stop', self._on_stop,
                            callback_group=self._service_group)
        self.create_service(Trigger, '~/status', self._on_status,
                            callback_group=self._service_group)
        self.create_service(Trigger, '~/new_session', self._on_new_session,
                            callback_group=self._service_group)
        self._discovery_timer = self.create_timer(5.0, self._report_discovery)

        self.get_logger().info(
            f'Ground control ready. Session dir: {self.session_dir} | remotes: '
            + ', '.join(f'{n}->{c.node}' for n, c in self.remotes.items())
            + ' | groups: ' + ', '.join(f'{g}={m}' for g, m in self.groups.items()))

    # ==================================================================== helpers
    def _param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        value = self.get_parameter(name).value
        return default if value is None else value

    def _log_ok(self, ok: bool, msg: str) -> None:
        # rclpy forbids one call site from logging at different severities, so keep two.
        if ok:
            self.get_logger().info(msg)
        else:
            self.get_logger().error(msg)

    def _new_session_dir(self) -> Path:
        return self.data_root / datetime.now().strftime(self.session_format)

    def _report_discovery(self) -> None:
        self._discovery_timer.cancel()
        online = [n for n, c in self.collectors.items() if c.is_available()]
        offline = [n for n in self.collectors if n not in online]
        self.get_logger().info(f'Remotes online: {online or "none"}; '
                               f'offline: {offline or "none"}')

    def _resolve(self, targets, empty_means) -> List[str]:
        return resolve_targets(list(targets), self.remotes, self.groups, empty_means)

    def _default_targets(self) -> List[str]:
        if self.default_group in self.groups:
            return list(self.groups[self.default_group])
        return [self.default_group] if self.default_group in self.remotes else []

    def _log_event(self, session_dir: Path, record: dict) -> None:
        """Append a JSON line to <session>/collection_log.jsonl (best effort)."""
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            record = dict(record, time=datetime.now().isoformat(timespec='seconds'))
            with open(session_dir / 'collection_log.jsonl', 'a') as f:
                f.write(json.dumps(record) + '\n')
        except OSError as exc:
            self.get_logger().warn(f'Could not write collection log: {exc}')

    # ==================================================================== outcomes & downloads
    def _on_outcome(self, remote: str, outcome: RecordingOutcome) -> Optional[DownloadJob]:
        """Handle a finished recording (from a done callback or a stop). Idempotent."""
        key = (remote, outcome.goal_id)
        if not outcome.bag.name:
            # Rejected before recording started (busy, bad profile...): nothing to collect.
            with self._lock:
                self._active[remote].pop(outcome.goal_id, None)
                self._goal_sessions.pop(key, None)
            self.get_logger().warn(f'[{remote}] {outcome.status_name}: {outcome.result.message}')
            return None
        with self._lock:
            self._active[remote].pop(outcome.goal_id, None)
            if key in self.jobs:
                return self.jobs[key]
            session = self._goal_sessions.pop(key, self.session_dir)
            job = DownloadJob(remote, outcome.goal_id, outcome.bag.name or '?', session)
            self.jobs[key] = job
        r = outcome.result
        self._log_ok(outcome.ok, f'[{remote}] {outcome.status_name}: {r.message}')
        if r.missing_topics:
            self.get_logger().warn(f'[{remote}] topics never recorded: {r.missing_topics}')
        self._log_event(session, {
            'event': 'recording_finished', 'remote': remote, 'goal_id': outcome.goal_id,
            'status': outcome.status_name, 'success': bool(r.success), 'message': r.message,
            'bag': r.bag.name, 'size_bytes': int(r.bag.size_bytes),
            'recorded_sec': r.recorded_sec, 'subscribed_topics': list(r.subscribed_topics),
            'missing_topics': list(r.missing_topics), 'download_url': r.bag.download_url,
            'log_url': r.log_url})
        if not outcome.ok:
            job.state, job.error = JOB_FAILED, f'recording failed: {r.message}'
            job.finished.set()
        elif not self.remotes[remote].download:
            job.state, job.error = JOB_SKIPPED, f'download disabled; bag at {r.bag.download_url}'
            job.finished.set()
        else:
            job.total_bytes = int(r.bag.size_bytes)
            job.future = self._downloads.submit(self._download, job, outcome)
        return job

    def _download(self, job: DownloadJob, outcome: RecordingOutcome) -> None:
        cfg = self.remotes[job.remote]
        client = self.collectors[job.remote]
        job.state = JOB_DOWNLOADING

        def progress(done, total):
            job.done_bytes, job.total_bytes = done, total

        t0 = time.monotonic()
        try:
            path = client.download(outcome.bag, job.session_dir, progress=progress,
                                   timeout=self.http_timeout, retries=self.download_retries)
            job.local_path = str(path)
            job.state = JOB_DONE
            secs = max(time.monotonic() - t0, 1e-3)
            self.get_logger().info(
                f'[{job.remote}] downloaded {job.bag_name} ({format_bytes(job.total_bytes)} in '
                f'{secs:.1f}s, {format_bytes(job.total_bytes / secs)}/s) -> {path}')
            if cfg.delete_after_download:
                try:
                    client.delete_bag(job.bag_name)
                    self.get_logger().info(f'[{job.remote}] deleted {job.bag_name} on remote')
                except CollectorError as exc:
                    self.get_logger().warn(f'[{job.remote}] remote delete failed: {exc}')
        except Exception as exc:  # noqa: B902 - any failure must surface as FAILED
            job.state, job.error = JOB_FAILED, f'{type(exc).__name__}: {exc}' \
                if not isinstance(exc, (DownloadError, OSError)) else str(exc)
            self.get_logger().error(
                f'[{job.remote}] download of {job.bag_name} failed: {exc}. The bag is still on '
                f'the remote: rdc download {cfg.node} {job.bag_name} --dest {job.session_dir}')
        finally:
            self._log_event(job.session_dir, {
                'event': 'download', 'remote': job.remote, 'bag': job.bag_name,
                'state': job.state, 'local_path': job.local_path, 'error': job.error})
            job.finished.set()

    # ==================================================================== services
    def _start_one(self, name: str, label: str, duration: float) -> StartedRecording:
        cfg = self.remotes[name]
        session = self.session_dir
        started = self.collectors[name].start(
            bag_name=cfg.bag_prefix(label), profiles=cfg.profiles, topics=cfg.topics,
            regex=cfg.regex, duration=cfg.effective_duration(duration),
            timeout=self.start_timeout,
            done_callback=lambda outcome, n=name: self._on_outcome(n, outcome))
        with self._lock:
            key = (name, started.goal_id)
            if key not in self.jobs:  # a very short recording may already be finished
                self._active[name][started.goal_id] = started
                self._goal_sessions[key] = session
        return started

    def _on_start(self, request, response):
        try:
            names = self._resolve(request.targets, self._default_targets())
        except FleetConfigError as exc:
            response.success, response.message = False, str(exc)
            return response
        if not names:
            response.success, response.message = False, 'no targets (and no default_group)'
            return response
        futures = {n: self._fanout.submit(self._start_one, n, request.label, request.duration)
                   for n in names}
        lines = []
        for name, fut in futures.items():
            try:
                started = fut.result()
                dur = self.remotes[name].effective_duration(request.duration)
                how = f'for {dur:.0f}s' if dur > 0 else 'until stopped'
                lines.append(f'  OK    {name}: recording {started.bag_name} ({how})')
                response.started.append(name)
            except Exception as exc:  # noqa: B902 - report every remote, whatever failed
                lines.append(f'  FAIL  {name}: {exc}')
                response.failed.append(name)
        response.success = not response.failed
        head = (f'Started {len(response.started)}/{len(names)} remote(s); '
                f'session {self.session_dir}')
        response.message = '\n'.join([head] + lines)
        self._log_ok(response.success, response.message)
        return response

    def _stop_one(self, name: str, explicit: bool) -> Optional[List[RecordingOutcome]]:
        """Stop one remote. Return None if it was skipped (offline, not explicitly targeted)."""
        client = self.collectors[name]
        with self._lock:
            started_here = bool(self._active[name])
        if not explicit and not started_here and not client.wait_until_available(2.0):
            return None
        outcomes = client.stop(timeout=self.stop_timeout)
        for outcome in outcomes:
            self._on_outcome(name, outcome)
        return outcomes

    def _on_stop(self, request, response):
        # Without explicit targets, stop every remote but don't fail on remotes that are
        # offline and have nothing of ours recording (e.g. a configured but unused camera).
        explicit = any(t.strip() for t in request.targets)
        try:
            names = self._resolve(request.targets, list(self.remotes))
        except FleetConfigError as exc:
            response.success, response.message = False, str(exc)
            return response
        futures = {n: self._fanout.submit(self._stop_one, n, explicit) for n in names}
        lines, ok = [], True
        for name, fut in futures.items():
            try:
                outcomes = fut.result()
            except Exception as exc:  # noqa: B902 - report every remote, whatever failed
                lines.append(f'  FAIL  {name}: {exc}')
                ok = False
                continue
            if outcomes is None:
                lines.append(f'  --    {name}: offline (skipped)')
                continue
            if not outcomes:
                lines.append(f'  --    {name}: nothing was recording')
            for o in outcomes:
                response.bags.append(o.bag.name)
                tag = 'OK  ' if o.ok else 'FAIL'
                ok = ok and o.ok
                lines.append(f'  {tag}  {name}: {o.bag.name or "?"} {o.status_name} '
                             f'({format_bytes(o.bag.size_bytes)}, {o.result.recorded_sec:.0f}s)')
                if o.result.missing_topics:
                    lines.append(f'        missing topics: {", ".join(o.result.missing_topics)}')

        # Every bag of these remotes not yet reported by a previous stop: this includes timed
        # recordings that finished on their own (e.g. `collect.sh scan`), even if they failed.
        with self._lock:
            unreported = [j for (remote, _), j in self.jobs.items()
                          if remote in names and not j.reported]
        pending = [j for j in unreported if not j.finished.is_set()]
        if request.wait_for_download:
            for job in unreported:
                job.finished.wait()
                job.reported = True
                lines.append(f'  {job.remote}/{job.bag_name}: {job.describe()}')
                if job.state == JOB_DONE:
                    response.local_paths.append(job.local_path)
                elif job.state == JOB_FAILED:
                    ok = False
        elif pending:
            lines.append(f'  {len(pending)} download(s) continuing in background '
                         '(see ~/status)')
        response.success = ok
        response.message = '\n'.join([f'Stopped {len(response.bags)} recording(s)'] + lines)
        self._log_ok(ok, response.message)
        return response

    def _remote_line(self, name: str) -> List[str]:
        cfg, client = self.remotes[name], self.collectors[name]
        if not client.is_available():
            return [f'  {name:16s} {cfg.node:20s} OFFLINE']
        try:
            st = client.status(timeout=3.0)
        except CollectorError as exc:
            return [f'  {name:16s} {cfg.node:20s} ERROR {exc}']
        http = 'http ok' if st.http_alive else 'HTTP DOWN'
        lines = [f'  {name:16s} {cfg.node:20s} online ({st.hostname}, '
                 f'{format_bytes(st.disk_free_bytes)} free, {http})']
        for a in st.active:
            dur = 'until stopped' if a.duration_sec <= 0 else f'/{a.duration_sec:.0f}s'
            lines.append(f'      recording {a.bag_name} [{a.state}] {a.elapsed_sec:.0f}s {dur} '
                         f'{format_bytes(a.size_bytes)} topics={len(a.subscribed_topics)}')
        return lines

    def _on_status(self, request, response):
        futures = [self._status_pool.submit(self._remote_line, n) for n in self.remotes]
        lines = [f'Session: {self.session_dir}', 'Remotes:']
        for fut in futures:
            lines += fut.result()
        with self._lock:
            jobs = list(self.jobs.values())
        if jobs:
            lines.append('Bags this run:')
            lines += [f'  {j.remote}/{j.bag_name}: {j.describe()}' for j in jobs[-20:]]
        response.success = True
        response.message = '\n'.join(lines)
        return response

    def _on_new_session(self, request, response):
        with self._lock:
            self.session_dir = self._new_session_dir()
        response.success = True
        response.message = str(self.session_dir)
        self.get_logger().info(f'New session directory: {self.session_dir}')
        return response

    def destroy_node(self):
        self._fanout.shutdown(wait=False, cancel_futures=True)
        self._status_pool.shutdown(wait=False, cancel_futures=True)
        self._downloads.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = GroundControl()
    except FleetConfigError as exc:
        print(f'[ground_control] configuration error: {exc}', flush=True)
        rclpy.try_shutdown()
        raise SystemExit(1)
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
