"""
Shared pytest fixtures for remote_data_collector tests.

The ROS harness imports rclpy lazily so the pure-Python tests also run without ROS.
"""

import os
from pathlib import Path
import sys
import threading
import time

import pytest

# Allow running the tests straight from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remote_data_collector.testing import free_port, install_fake_ros2  # noqa: E402


@pytest.fixture
def fake_ros2(tmp_path) -> str:
    """Path to an executable fake `ros2` (see remote_data_collector/testing/fake_ros2.py)."""
    return install_fake_ros2(tmp_path / 'bin')


@pytest.fixture
def fake_mode(monkeypatch):
    """Set FAKE_ROS2_* behaviour for recorders spawned during the test."""
    def set_mode(mode='normal', **extra):
        monkeypatch.setenv('FAKE_ROS2_MODE', mode)
        for key, value in extra.items():
            monkeypatch.setenv(f'FAKE_ROS2_{key.upper()}', str(value))
    for var in [v for v in os.environ if v.startswith('FAKE_ROS2_')]:
        monkeypatch.delenv(var)
    return set_mode


@pytest.fixture(scope='module')
def ros_context():
    # Per module, so DDS threads are gone before tests that fork (e.g. linters) run.
    rclpy = pytest.importorskip('rclpy')
    pytest.importorskip('remote_data_collector_interfaces')
    rclpy.init()
    yield
    rclpy.try_shutdown()


_counter = [0]


class Harness:
    """A collector node + client node spun by one background executor."""

    def __init__(self, tmp_path, ros2_executable, **overrides):
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from remote_data_collector.client import RemoteCollectorClient
        from remote_data_collector.collector_node import RemoteDataCollector
        _counter[0] += 1
        self.name = f'rdc_test_{_counter[0]}'
        self.data_dir = tmp_path / 'data'
        params = {
            'data_dir': str(self.data_dir),
            'ros2_executable': ros2_executable,
            'default_profiles': ['demo'],
            'default_bag_name': 'test_bags',
            'feedback_period_sec': 0.2,
            'startup_timeout_sec': 5.0,
            'stop_timeout_sec': 5.0,
            'min_free_space_mb': 0,
            'stop_free_space_mb': 0,
            'http.port': free_port(),
            'http.bind': '127.0.0.1',
            'http.advertise_host': '127.0.0.1',
            'profiles.demo.topics': ['/demo/a', '/demo/missing_b'],
            'profiles.tf.topics': ['/tf'],
        }
        params.update(overrides)
        self.collector = RemoteDataCollector(
            self.name, parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
        self.client_node = Node(f'{self.name}_client')
        self.executor = MultiThreadedExecutor(num_threads=8)
        self.executor.add_node(self.collector)
        self.executor.add_node(self.client_node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        self.client = RemoteCollectorClient(self.client_node, '/' + self.name)
        assert self.client.wait_until_available(10.0), 'collector not discovered in-process'

    def new_client(self):
        from remote_data_collector.client import RemoteCollectorClient
        return RemoteCollectorClient(self.client_node, '/' + self.name)

    def close(self):
        self.collector.shutdown_recordings(timeout=10)
        self.executor.shutdown(timeout_sec=2)
        self.collector.destroy_node()
        self.client_node.destroy_node()


@pytest.fixture
def harness_factory(ros_context, tmp_path, fake_ros2, fake_mode):
    fake_mode('normal')
    made = []

    def make(**overrides):
        h = Harness(tmp_path, overrides.pop('ros2_executable', fake_ros2), **overrides)
        made.append(h)
        return h
    yield make
    for h in made:
        h.close()


def wait_for(predicate, timeout=10.0, period=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(period)
    return False
