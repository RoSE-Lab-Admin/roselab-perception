"""LidarCapture (scan_automation's replacement for gantry_capture_service) vs a real collector."""

from pathlib import Path
import sys
import threading

import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('remote_data_collector')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from remote_data_collector.client import CollectorError  # noqa: E402
from remote_data_collector.collector_node import RemoteDataCollector  # noqa: E402
from remote_data_collector.testing import free_port, install_fake_ros2  # noqa: E402
from scan_automation.lidar_capture import LidarCapture  # noqa: E402


@pytest.fixture
def gantry(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_ROS2_MODE', 'normal')
    rclpy.init()
    params = {
        'data_dir': str(tmp_path / 'gantry_data'),
        'ros2_executable': install_fake_ros2(tmp_path / 'bin'),
        'feedback_period_sec': 0.2, 'min_free_space_mb': 0, 'stop_free_space_mb': 0,
        'http.port': free_port(), 'http.bind': '127.0.0.1', 'http.advertise_host': '127.0.0.1',
        'profiles.gantry_state.topics': ['/gantry/gantry_status/gantry_state'],
        'profiles.p_l515_center.topics': ['/p_l515_center/color/image_raw'],
    }
    node = RemoteDataCollector('gantry_test', parameter_overrides=[
        Parameter(k, value=v) for k, v in params.items()])
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    yield node
    node.shutdown_recordings(timeout=10)
    executor.shutdown(timeout_sec=2)
    node.destroy_node()
    rclpy.try_shutdown()


def test_timed_scan_download_and_delete(gantry, tmp_path):
    lidar = LidarCapture(remote='/gantry_test', bag_name='lidar_bags',
                         profiles=['gantry_state', 'p_l515_center'])
    try:
        lidar.wait_until_available(timeout=10)
        started = lidar.start(1.0)
        assert started.bag_name.startswith('lidar_bags_')
        path = lidar.finish(tmp_path / 'slade', delete=True, timeout=30)
        port = gantry.http.port
        # Same layout as the legacy `wget -r`: <dest>/<host>:<port>/<bag>/
        assert path == tmp_path / 'slade' / f'127.0.0.1:{port}' / started.bag_name
        assert (path / 'metadata.yaml').is_file()
        assert lidar.client.list_bags() == []  # deleted on the gantry
        with pytest.raises(CollectorError):
            lidar.finish(tmp_path)  # nothing started
    finally:
        lidar.close()


def test_unknown_profile_is_reported(gantry):
    lidar = LidarCapture(remote='/gantry_test', profiles=['p_l515_nope'])
    try:
        with pytest.raises(CollectorError, match='unknown profile'):
            lidar.start(1.0)
    finally:
        lidar.close()
