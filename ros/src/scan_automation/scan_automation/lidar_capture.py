"""
Blocking lidar capture on the gantry's remote_data_collector, for the scan_automation nodes.

Replaces the legacy gantry_capture_service calls (capture -> download/name -> wget -> delete).
Uses its own helper node spun in a background thread, so it works from the single-threaded,
spin_once-style code in GantryCommand / LidarScan.
"""

import os
from pathlib import Path
import time
from typing import Optional, Sequence
from urllib.parse import urlparse

import rclpy
from remote_data_collector.client import (CollectorError, RemoteCollectorClient,
                                          spin_in_background, StartedRecording)

DEFAULT_PROFILES = ('gantry_state', 'p_l515_center', 'p_l515_west', 'p_l515_east')


class LidarCapture:
    """Start a timed lidar recording, then wait for it and download it."""

    def __init__(self, remote: str = '/gantry', bag_name: str = 'lidar_bags',
                 profiles: Sequence[str] = DEFAULT_PROFILES, logger=None):
        self.bag_name = bag_name
        self.profiles = list(profiles)
        self._node = rclpy.create_node(f'scan_automation_lidar_capture_{os.getpid()}')
        self._executor = spin_in_background(self._node, num_threads=2)
        self.client = RemoteCollectorClient(self._node, remote)
        self.logger = logger or self._node.get_logger()
        self.started: Optional[StartedRecording] = None

    def wait_until_available(self, timeout: Optional[float] = None) -> None:
        """Block until the collector is reachable (forever if timeout is None)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.client.wait_until_available(2.0):
            self.logger.info(f'Waiting for data collector {self.client.remote} ...')
            if deadline is not None and time.monotonic() > deadline:
                raise CollectorError(f'{self.client.remote} not reachable')

    def start(self, duration: float) -> StartedRecording:
        """Start recording for `duration` seconds; return once recording is confirmed."""
        self.started = self.client.start(bag_name=self.bag_name, profiles=self.profiles,
                                         duration=float(duration))
        self.logger.info(f'Lidar recording {self.started.bag_name} for {duration:.0f}s')
        return self.started

    def finish(self, dest_dir, delete: bool = False, timeout: float = 3600.0,
               host_subdir: bool = True) -> Path:
        """Wait for the recording to end, download it, optionally delete it on the gantry.

        With `host_subdir` the bag lands in ``dest_dir/<host>:<port>/<bag>/``, the layout the
        legacy ``wget -r`` download produced; otherwise in ``dest_dir/<bag>/``.
        """
        if self.started is None:
            raise CollectorError('finish() called before start()')
        self.logger.info('lidar scanning.....')
        outcome = self.client.get_result(self.started.goal_id, timeout=timeout)
        if not outcome.ok:
            raise CollectorError(f'lidar recording failed: {outcome.result.message}')
        if outcome.result.missing_topics:
            self.logger.warn(f'Topics not recorded: {list(outcome.result.missing_topics)}')
        dest = Path(dest_dir)
        if host_subdir and outcome.bag.download_url:
            dest = dest / urlparse(outcome.bag.download_url).netloc
        path = self.client.download(outcome.bag, dest)
        self.logger.info(f'bag saved to {path}')
        if delete:
            self.client.delete_bag(outcome.bag.name)
            self.logger.info(f'Deleted {outcome.bag.name} on the gantry')
        self.started = None
        return path

    def close(self) -> None:
        self._executor.shutdown(timeout_sec=1.0)
        self._node.destroy_node()
