"""
Integration tests: ground_control <-> two remote_data_collectors, all in one process.

Collectors use the fake `ros2` recorder from remote_data_collector.testing, so no DDS traffic
between processes is needed.
"""

import json
from pathlib import Path
import threading
import time

from perception_ground_control.ground_control import GroundControl
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from remote_data_collector.client import wait_future
from remote_data_collector.collector_node import RemoteDataCollector
from remote_data_collector.testing import free_port, install_fake_ros2
from remote_data_collector_interfaces.srv import StartCollection, StopCollection
from std_srvs.srv import Trigger

_counter = [0]


def _params(d):
    return [Parameter(k, value=v) for k, v in d.items()]


@pytest.fixture(scope='module')
def ros_context():
    rclpy.init()
    yield
    rclpy.try_shutdown()


class Fleet:
    """Two collectors (cam, lidar) + ground control + a caller node."""

    def __init__(self, tmp_path, **gc_overrides):
        _counter[0] += 1
        n = _counter[0]
        fake = install_fake_ros2(tmp_path / 'bin')
        self.cam_name, self.lidar_name = f'gc_cam_{n}', f'gc_lidar_{n}'
        self.cam_dir, self.lidar_dir = tmp_path / 'cam_data', tmp_path / 'lidar_data'
        self.data_root = tmp_path / 'ground'
        common = {
            'ros2_executable': fake, 'feedback_period_sec': 0.2, 'startup_timeout_sec': 5.0,
            'stop_timeout_sec': 5.0, 'min_free_space_mb': 0, 'stop_free_space_mb': 0,
            'http.bind': '127.0.0.1', 'http.advertise_host': '127.0.0.1',
        }
        self.cam = RemoteDataCollector(self.cam_name, parameter_overrides=_params(dict(
            common, **{'data_dir': str(self.cam_dir), 'http.port': free_port(),
                       'default_profiles': ['cam'], 'profiles.cam.topics': ['/cam/image']})))
        self.lidar = RemoteDataCollector(self.lidar_name, parameter_overrides=_params(dict(
            common, **{'data_dir': str(self.lidar_dir), 'http.port': free_port(),
                       'profiles.scan.topics': ['/lidar/points', '/lidar/missing_x'],
                       'profiles.state.topics': ['/gantry/state']})))
        gc = {
            'data_root': str(self.data_root),
            'session_format': 'session_%H%M%S_%f',
            'default_group': 'trial',
            'start_timeout_sec': 10.0,
            'stop_timeout_sec': 20.0,
            'remotes.cam.node': '/' + self.cam_name,
            'remotes.cam.bag_name': 'mastcam_bags',
            'remotes.cam.delete_after_download': True,
            'remotes.lidar.node': '/' + self.lidar_name,
            'remotes.lidar.bag_name': 'lidar_bags',
            'remotes.lidar.profiles': ['scan', 'state'],
            'remotes.lidar.duration': 1.0,
            'remotes.offline.node': '/does_not_exist',
            'groups.trial': ['cam'],
            'groups.scan': ['lidar'],
        }
        gc.update(gc_overrides)
        self.gc = GroundControl(parameter_overrides=_params(gc))
        self.caller = Node(f'gc_caller_{n}')
        self.executor = MultiThreadedExecutor(num_threads=12)
        for node in (self.cam, self.lidar, self.gc, self.caller):
            self.executor.add_node(node)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        self.start_cli = self.caller.create_client(StartCollection, '/ground_control/start')
        self.stop_cli = self.caller.create_client(StopCollection, '/ground_control/stop')
        self.status_cli = self.caller.create_client(Trigger, '/ground_control/status')
        self.session_cli = self.caller.create_client(Trigger, '/ground_control/new_session')
        for cli in (self.start_cli, self.stop_cli, self.status_cli, self.session_cli):
            assert cli.wait_for_service(timeout_sec=10.0)
        for client in (self.gc.collectors['cam'], self.gc.collectors['lidar']):
            assert client.wait_until_available(10.0)

    def call(self, cli, request, timeout=60.0):
        return wait_future(cli.call_async(request), timeout, cli.srv_name)

    def start(self, targets=(), label='', duration=0.0):
        return self.call(self.start_cli, StartCollection.Request(
            targets=list(targets), label=label, duration=float(duration)))

    def stop(self, targets=(), wait=True):
        return self.call(self.stop_cli, StopCollection.Request(
            targets=list(targets), wait_for_download=wait))

    def status(self):
        return self.call(self.status_cli, Trigger.Request()).message

    def close(self):
        for c in (self.cam, self.lidar):
            c.shutdown_recordings(timeout=10)
        self.executor.shutdown(timeout_sec=2)
        for node in (self.gc, self.cam, self.lidar, self.caller):
            node.destroy_node()


@pytest.fixture
def fleet(ros_context, tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_ROS2_MODE', 'normal')
    f = Fleet(tmp_path)
    yield f
    f.close()


def _bags(path: Path):
    return sorted(p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith('.'))


def test_trial_start_stop_downloads_and_deletes(fleet):
    resp = fleet.start(label='run01')  # default group "trial" -> cam
    assert resp.success, resp.message
    assert list(resp.started) == ['cam'] and list(resp.failed) == []
    assert 'run01_mastcam_bags_' in resp.message
    time.sleep(0.5)
    assert 'recording run01_mastcam_bags_' in fleet.status()

    resp = fleet.stop(['trial'])
    assert resp.success, resp.message
    assert len(resp.bags) == 1 and resp.bags[0].startswith('run01_mastcam_bags_')
    assert len(resp.local_paths) == 1
    local = Path(resp.local_paths[0])
    assert local.parent == fleet.gc.session_dir
    assert (local / 'metadata.yaml').is_file()
    # delete_after_download: the remote copy is gone.
    assert _bags(fleet.cam_dir) == []
    log = [json.loads(line) for line in
           (fleet.gc.session_dir / 'collection_log.jsonl').read_text().splitlines()]
    assert [e['event'] for e in log] == ['recording_finished', 'download']
    assert log[0]['status'] == 'CANCELED' and log[0]['success']
    assert log[1]['state'] == 'done'


def test_timed_scan_downloads_automatically_and_keeps_remote(fleet):
    resp = fleet.start(['scan'])
    assert resp.success, resp.message
    assert 'for 1s' in resp.message

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not any(
            j.state == 'done' for j in fleet.gc.jobs.values()):
        time.sleep(0.1)
    job = next(iter(fleet.gc.jobs.values()))
    assert job.state == 'done', job.describe()
    assert Path(job.local_path).name.startswith('lidar_bags_')
    assert len(_bags(fleet.lidar_dir)) == 1  # delete_after_download is false for lidar
    status = fleet.status()
    assert 'lidar/lidar_bags_' in status and 'done ->' in status

    # Stopping afterwards is harmless and reports the finished download.
    resp = fleet.stop(['scan'])
    assert resp.success, resp.message
    assert 'nothing was recording' in resp.message


def test_duration_override_and_label(fleet):
    resp = fleet.start(['cam'], label='short', duration=1.0)
    assert resp.success, resp.message
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not fleet.gc.jobs:
        time.sleep(0.1)
    resp = fleet.stop(['cam'])  # waits for the auto-download that is in flight
    assert resp.success, resp.message
    assert any(Path(p).name.startswith('short_mastcam_bags_') for p in
               [j.local_path for j in fleet.gc.jobs.values()])


def test_errors_are_reported_per_remote(fleet):
    resp = fleet.start(['nope'])
    assert not resp.success and 'unknown target' in resp.message

    resp = fleet.start(['cam', 'offline'])
    assert not resp.success
    assert list(resp.started) == ['cam'] and list(resp.failed) == ['offline']
    assert 'not reachable' in resp.message
    assert 'OFFLINE' in fleet.status()

    # The cam is busy now: a second start fails cleanly with the collector's reason.
    resp = fleet.start(['cam'])
    assert not resp.success and 'busy' in resp.message

    # Bare stop: everything, but the offline (unused) remote is skipped, not a failure.
    resp = fleet.stop()
    assert resp.success, resp.message
    assert 'offline: offline (skipped)' in resp.message
    assert len(resp.bags) == 1 and len(resp.local_paths) == 1
    # Explicitly targeting an offline remote is a failure.
    resp = fleet.stop(['offline'])
    assert not resp.success and 'not reachable' in resp.message


def test_stop_collects_recordings_started_elsewhere(fleet):
    """Stateless stop: a recording started directly on the collector is still collected."""
    from remote_data_collector.client import RemoteCollectorClient
    direct = RemoteCollectorClient(fleet.caller, '/' + fleet.cam_name)
    started = direct.start(bag_name='manual')
    resp = fleet.stop(['cam'])
    assert resp.success, resp.message
    assert list(resp.bags) == [started.bag_name]
    assert Path(resp.local_paths[0]).name == started.bag_name


def test_new_session_changes_download_dir(fleet):
    first = fleet.gc.session_dir
    time.sleep(0.01)
    msg = fleet.call(fleet.session_cli, Trigger.Request()).message
    assert msg == str(fleet.gc.session_dir) and fleet.gc.session_dir != first
    fleet.start(['cam'])
    resp = fleet.stop(['cam'])
    assert Path(resp.local_paths[0]).parent == fleet.gc.session_dir


def test_rdc_gc_commands(fleet, capsys):
    """The `rdc gc ...` commands used by scripts/collect.sh, against the real ground control."""
    from remote_data_collector import cli

    def rdc_gc(*argv):
        code = cli.run(['--timeout', '5', 'gc'] + list(argv), fleet.caller)
        out = capsys.readouterr()
        return code, out.out + out.err

    code, out = rdc_gc('start', 'trial', '--label', 'clirun')
    assert code == 0 and 'recording clirun_mastcam_bags_' in out
    code, out = rdc_gc('status')
    assert code == 0 and 'recording clirun_mastcam_bags_' in out and 'OFFLINE' in out
    code, out = rdc_gc('stop')
    assert code == 0 and 'done ->' in out
    code, out = rdc_gc('start', 'bogus_target')
    assert code == 1 and 'unknown target' in out
    code, out = rdc_gc('new-session')
    assert code == 0 and out.strip() == str(fleet.gc.session_dir)
    code = cli.run(['--timeout', '1', 'gc', '--gc-node', '/nope', 'status'], fleet.caller)
    assert code == 3 and 'not reachable' in capsys.readouterr().err


def test_stop_reports_timed_recording_that_already_failed(fleet, monkeypatch):
    """Regression: a timed scan whose recorder died before `stop` must make stop fail."""
    monkeypatch.setenv('FAKE_ROS2_MODE', 'crash')
    monkeypatch.setenv('FAKE_ROS2_REINDEX', 'fail')
    resp = fleet.start(['scan'], duration=30.0)
    assert resp.success, resp.message
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not any(
            j.finished.is_set() for j in fleet.gc.jobs.values()):
        time.sleep(0.1)
    resp = fleet.stop(['scan'])
    assert not resp.success, resp.message
    assert 'nothing was recording' in resp.message and 'FAILED' in resp.message
    # Reported once: the next stop has nothing new to say and succeeds.
    assert fleet.stop(['scan']).success
