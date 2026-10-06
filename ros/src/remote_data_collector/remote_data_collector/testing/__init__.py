"""
Test helpers shared by remote_data_collector and the packages that use it.

``install_fake_ros2(directory)`` drops an executable stand-in for ``ros2`` (see fake_ros2.py)
that implements ``ros2 bag record`` / ``ros2 bag reindex`` without DDS, so collector behaviour
can be tested quickly and deterministically. Point a collector's ``ros2_executable`` parameter
at the returned path.
"""

from pathlib import Path
import shutil
import socket
import stat

FAKE_ROS2 = Path(__file__).resolve().parent / 'fake_ros2.py'


def install_fake_ros2(directory) -> str:
    """Copy the fake `ros2` into `directory`, make it executable and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dst = directory / 'ros2'
    shutil.copy(FAKE_ROS2, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(dst)


def free_port() -> int:
    """Return a TCP port that is currently free on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]
