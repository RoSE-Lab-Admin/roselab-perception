"""
Client library for remote_data_collector instances.

Used by the ``rdc`` CLI and by perception_ground_control. All methods are blocking and
thread-safe, and they *assume the node is being spun by an executor in another thread*
(a MultiThreadedExecutor for a long-running node, or :func:`spin_in_background` for scripts).
Never call them from a MutuallyExclusiveCallbackGroup callback of the same node.

Stopping is stateless: :meth:`RemoteCollectorClient.stop` uses the standard action
cancel/get_result protocol, so a recording started by one process (e.g. ``rdc start``) can be
stopped and collected by another (e.g. ground control or ``rdc stop``).
"""

from dataclasses import dataclass, field
import threading
import time
from typing import Callable, List, Optional, Sequence

from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from remote_data_collector_interfaces.action import RecordBag
from remote_data_collector_interfaces.msg import BagInfo, RecordingStatus
from remote_data_collector_interfaces.srv import DeleteBag, GetStatus, ListBags

from .download import download_bag, ProgressCallback
from .naming import goal_id_to_hex, hex_to_goal_id

ACTION_TYPE = 'remote_data_collector_interfaces/action/RecordBag'

STATUS_NAMES = {
    GoalStatus.STATUS_UNKNOWN: 'UNKNOWN',
    GoalStatus.STATUS_ACCEPTED: 'ACCEPTED',
    GoalStatus.STATUS_EXECUTING: 'EXECUTING',
    GoalStatus.STATUS_CANCELING: 'CANCELING',
    GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
    GoalStatus.STATUS_CANCELED: 'CANCELED',
    GoalStatus.STATUS_ABORTED: 'ABORTED',
}


class CollectorError(RuntimeError):
    """A request to a collector failed (unreachable, rejected, timed out...)."""


@dataclass
class StartedRecording:
    """Handle returned by :meth:`RemoteCollectorClient.start`."""

    remote: str
    goal_id: str
    bag_name: str
    goal_handle: object = None
    result_future: object = None


@dataclass
class RecordingOutcome:
    """Final outcome of one recording (one action goal)."""

    remote: str
    goal_id: str
    status: int
    result: RecordBag.Result = field(default_factory=RecordBag.Result)

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, str(self.status))

    @property
    def ok(self) -> bool:
        """Return True if a finalized bag is available for download."""
        return bool(self.result.success)

    @property
    def bag(self) -> BagInfo:
        return self.result.bag


def normalize_remote(remote: str) -> str:
    """Return the collector's fully qualified node name: leading '/', no trailing '/'."""
    remote = remote.strip().rstrip('/')
    if not remote:
        raise ValueError('empty collector name')
    return remote if remote.startswith('/') else '/' + remote


def wait_future(future, timeout: Optional[float], what: str = 'request'):
    """Block until an rclpy future completes (it must be spun elsewhere) and return its result."""
    if not future.done():
        event = threading.Event()
        future.add_done_callback(lambda _f: event.set())
        if not future.done() and not event.wait(timeout):
            raise CollectorError(f'{what} timed out after {timeout:.1f}s')
    exc = future.exception()
    if exc is not None:
        raise CollectorError(f'{what} failed: {exc}')
    return future.result()


class BackgroundSpinner:
    """A MultiThreadedExecutor spinning one node in a background thread."""

    def __init__(self, node: Node, num_threads: int = 4):
        self.executor = MultiThreadedExecutor(num_threads=num_threads)
        self.executor.add_node(node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        try:
            self.executor.spin()
        except ExternalShutdownException:
            pass

    def shutdown(self, timeout_sec: float = 5.0) -> None:
        """
        Stop spinning and JOIN the thread.

        The thread must be joined before the node or context is destroyed: tearing them down
        while it is still inside rclpy aborts the process ("terminate called without an
        active exception").
        """
        self.executor.shutdown(timeout_sec=timeout_sec)
        self._thread.join(timeout_sec)


def spin_in_background(node: Node, num_threads: int = 4) -> BackgroundSpinner:
    """Spin `node` in a background thread; call .shutdown() on the result before cleanup."""
    return BackgroundSpinner(node, num_threads)


def discover_collectors(node: Node, wait: float = 2.0) -> List[str]:
    """Return the fully qualified names of collectors visible on the ROS graph."""
    from rclpy.action import get_action_names_and_types
    time.sleep(wait)  # give DDS discovery a moment
    found = []
    for name, types in get_action_names_and_types(node):
        if ACTION_TYPE in types and name.endswith('/record'):
            found.append(name[:-len('/record')])
    return sorted(found)


def _cancel_if_accepted(send_future) -> None:
    try:
        handle = send_future.result()
    except Exception:  # noqa: B902 - nothing to cancel
        return
    if handle is not None and handle.accepted:
        handle.cancel_goal_async()


class RemoteCollectorClient:
    """Talk to one remote_data_collector (identified by its fully qualified node name)."""

    def __init__(self, node: Node, remote: str, callback_group=None):
        self.node = node
        self.remote = normalize_remote(remote)
        cg = callback_group
        action = f'{self.remote}/record'
        self._action = ActionClient(node, RecordBag, action, callback_group=cg)
        self._status = node.create_client(GetStatus, f'{self.remote}/status', callback_group=cg)
        self._list = node.create_client(ListBags, f'{self.remote}/list_bags', callback_group=cg)
        self._delete = node.create_client(DeleteBag, f'{self.remote}/delete_bag',
                                          callback_group=cg)
        # Raw action protocol services: let us cancel / fetch results of goals we did not send.
        self._cancel = node.create_client(CancelGoal, f'{action}/_action/cancel_goal',
                                          callback_group=cg)
        self._result = node.create_client(RecordBag.Impl.GetResultService,
                                          f'{action}/_action/get_result', callback_group=cg)

    def destroy(self) -> None:
        """Release the ROS entities (the node itself is not destroyed)."""
        self._action.destroy()
        for cli in (self._status, self._list, self._delete, self._cancel, self._result):
            self.node.destroy_client(cli)

    # ------------------------------------------------------------------ availability
    def wait_until_available(self, timeout: float = 10.0) -> bool:
        """Return True once the action server and services are discovered."""
        deadline = time.monotonic() + timeout
        if not self._action.wait_for_server(timeout_sec=max(0.0, timeout)):
            return False
        for cli in (self._status, self._cancel, self._result):
            if not cli.wait_for_service(timeout_sec=max(0.1, deadline - time.monotonic())):
                return False
        return True

    def is_available(self) -> bool:
        """Return True if the collector is currently discovered (non-blocking)."""
        return self._action.server_is_ready() and self._status.service_is_ready()

    def require_available(self, timeout: float) -> None:
        """Raise CollectorError unless the collector is discovered within `timeout`."""
        if not self.wait_until_available(timeout):
            raise CollectorError(
                f'collector {self.remote} is not reachable (is the node running, on the same '
                'ROS_DOMAIN_ID/network, and built with the same interfaces?)')

    def _call(self, client, request, timeout: float, what: str):
        if not client.wait_for_service(timeout_sec=timeout):
            raise CollectorError(f'{what}: service {client.srv_name} not available')
        return wait_future(client.call_async(request), timeout, what)

    # ------------------------------------------------------------------ recording
    def start(
        self,
        bag_name: str = '',
        profiles: Sequence[str] = (),
        topics: Sequence[str] = (),
        regex: str = '',
        duration: float = 0.0,
        timeout: float = 30.0,
        feedback_callback: Optional[Callable[[RecordingStatus], None]] = None,
        done_callback: Optional[Callable[[RecordingOutcome], None]] = None,
    ) -> StartedRecording:
        """
        Start a recording and return once the collector confirms it is RECORDING.

        Raise CollectorError if the goal is rejected, aborted (with the collector's reason) or
        not confirmed within `timeout` (in which case the goal is canceled again).
        `done_callback` is invoked (from an executor thread) with the final outcome.
        """
        self.require_available(min(timeout, 10.0))
        goal = RecordBag.Goal()
        goal.bag_name = bag_name
        goal.profiles = list(profiles)
        goal.topics = list(topics)
        goal.regex = regex
        goal.duration = float(duration)

        confirmed = threading.Event()
        state = {'bag_name': '', 'outcome': None}

        def on_feedback(msg):
            status = msg.feedback.status
            state['bag_name'] = status.bag_name
            if status.state == RecordingStatus.STATE_RECORDING:
                confirmed.set()
            if feedback_callback:
                self._safe_call(feedback_callback, status)

        send_future = self._action.send_goal_async(goal, feedback_callback=on_feedback)
        try:
            goal_handle = wait_future(send_future, timeout, f'{self.remote}: send goal')
        except BaseException:
            # Timed out / interrupted: if the goal is accepted later, cancel it right away.
            send_future.add_done_callback(_cancel_if_accepted)
            raise
        if not goal_handle.accepted:
            raise CollectorError(f'{self.remote}: goal rejected (collector shutting down?)')
        goal_id = goal_id_to_hex(goal_handle.goal_id.uuid)

        result_future = goal_handle.get_result_async()

        def on_result(fut):
            try:
                resp = fut.result()
                outcome = RecordingOutcome(self.remote, goal_id, resp.status, resp.result)
            except Exception as exc:  # noqa: B902 - surface as an ABORTED outcome
                res = RecordBag.Result()
                res.message = f'failed to get result: {exc}'
                outcome = RecordingOutcome(self.remote, goal_id, GoalStatus.STATUS_ABORTED, res)
            state['outcome'] = outcome
            confirmed.set()
            if done_callback:
                self._safe_call(done_callback, outcome)

        result_future.add_done_callback(on_result)

        try:
            ok = confirmed.wait(timeout)
        except BaseException:  # e.g. Ctrl-C before the recording was confirmed
            goal_handle.cancel_goal_async()
            raise
        if not ok:
            goal_handle.cancel_goal_async()
            raise CollectorError(
                f'{self.remote}: recording not confirmed within {timeout:.0f}s; goal canceled')
        outcome = state['outcome']
        if outcome is not None and not state['bag_name']:
            raise CollectorError(f'{self.remote}: {outcome.status_name}: '
                                 f'{outcome.result.message}')
        if outcome is not None and outcome.status == GoalStatus.STATUS_ABORTED:
            raise CollectorError(f'{self.remote}: {outcome.result.message}')
        return StartedRecording(self.remote, goal_id, state['bag_name'] or
                                (outcome.bag.name if outcome else ''), goal_handle,
                                result_future)

    def _safe_call(self, callback, arg) -> None:
        """Run a user callback; an exception must never kill the executor thread."""
        try:
            callback(arg)
        except Exception as exc:  # noqa: B902
            self.node.get_logger().error(
                f'{self.remote}: callback {getattr(callback, "__name__", callback)} raised: '
                f'{exc!r}')

    def get_result(self, goal_id: str, timeout: float = 120.0) -> RecordingOutcome:
        """Wait for and return the outcome of any goal on this collector."""
        req = RecordBag.Impl.GetResultService.Request()
        req.goal_id.uuid = hex_to_goal_id(goal_id)
        resp = self._call(self._result, req, timeout, f'{self.remote}: get result {goal_id[:8]}')
        return RecordingOutcome(self.remote, goal_id, resp.status, resp.result)

    def cancel(self, goal_ids: Optional[Sequence[str]] = None, timeout: float = 10.0) -> List[str]:
        """Request a stop of the given goals (all active goals if None). Return canceled ids."""
        requests = []
        if goal_ids is None:
            requests.append(CancelGoal.Request())  # zero UUID + zero stamp = cancel all goals
        else:
            for gid in goal_ids:
                req = CancelGoal.Request()
                req.goal_info.goal_id.uuid = hex_to_goal_id(gid)
                requests.append(req)
        canceled = []
        for req in requests:
            resp = self._call(self._cancel, req, timeout, f'{self.remote}: cancel')
            canceled += [goal_id_to_hex(g.goal_id.uuid) for g in resp.goals_canceling]
        return canceled

    def stop(
        self,
        goal_ids: Optional[Sequence[str]] = None,
        bag_names: Optional[Sequence[str]] = None,
        timeout: float = 120.0,
    ) -> List[RecordingOutcome]:
        """
        Stop recordings and return their outcomes (with download info).

        With no arguments every active recording on the collector is stopped. Recordings can
        also be selected by goal id or by bag name. Returns an empty list if nothing matched.
        """
        self.require_available(min(timeout, 10.0))
        if bag_names:
            active = self.status(timeout=min(timeout, 10.0)).active
            wanted = set(bag_names)
            goal_ids = list(goal_ids or []) + [a.goal_id for a in active
                                               if a.bag_name in wanted]
            if not goal_ids:
                return []
        ids = self.cancel(goal_ids, timeout=min(timeout, 10.0))
        return [self.get_result(gid, timeout) for gid in ids]

    # ------------------------------------------------------------------ services
    def status(self, timeout: float = 5.0) -> GetStatus.Response:
        """Return the collector's GetStatus response."""
        return self._call(self._status, GetStatus.Request(), timeout, f'{self.remote}: status')

    def list_bags(self, pattern: str = '', timeout: float = 10.0) -> List[BagInfo]:
        """Return the bags stored on the collector (newest first)."""
        resp = self._call(self._list, ListBags.Request(pattern=pattern), timeout,
                          f'{self.remote}: list_bags')
        if not resp.success:
            raise CollectorError(f'{self.remote}: {resp.message}')
        return list(resp.bags)

    def find_bag(self, name: str, timeout: float = 10.0) -> BagInfo:
        """Return the BagInfo of one bag by exact name."""
        for bag in self.list_bags(name, timeout):
            if bag.name == name:
                return bag
        raise CollectorError(f'{self.remote}: no bag named {name!r}')

    def delete_bag(self, name: str, timeout: float = 30.0) -> DeleteBag.Response:
        """Delete one bag on the collector. Raise CollectorError on failure."""
        resp = self._call(self._delete, DeleteBag.Request(name=name), timeout,
                          f'{self.remote}: delete_bag')
        if not resp.success:
            raise CollectorError(f'{self.remote}: {resp.message}')
        return resp

    # ------------------------------------------------------------------ data
    @staticmethod
    def download(
        bag: BagInfo,
        dest_dir,
        progress: Optional[ProgressCallback] = None,
        timeout: float = 30.0,
        retries: int = 3,
    ):
        """Download a bag (from a BagInfo) into ``dest_dir/<bag.name>``; return the path."""
        return download_bag(bag.download_url, list(bag.files), dest_dir, bag.name,
                            file_sizes=list(bag.file_sizes), progress=progress,
                            timeout=timeout, retries=retries)
