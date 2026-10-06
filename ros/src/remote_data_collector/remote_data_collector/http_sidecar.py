"""
Sidecar ``python3 -m http.server`` that serves the collector's data directory.

The server runs as a separate process (no GIL contention with the ROS node, and a crash in
one does not take down the other). The collector supervises it and restarts it if it dies.
"""

import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Optional
from urllib.parse import quote

from .procutil import supervised_argv


def detect_advertise_host() -> str:
    """
    Best-effort guess of this host's LAN IP address.

    Uses the "connect a UDP socket" trick (no packet is sent), which picks the interface of the
    default route. ``gethostbyname(gethostname())`` is avoided because it commonly returns
    127.0.1.1 on Debian/Ubuntu.
    """
    for probe in ('192.168.2.1', '10.255.255.255', '8.8.8.8'):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 9))
                ip = s.getsockname()[0]
                if ip and not ip.startswith('127.'):
                    return ip
        except OSError:
            continue
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return '127.0.0.1'


def port_is_free(port: int, bind: str = '0.0.0.0') -> bool:
    """Return True if a TCP server could bind `bind:port` right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((bind, port))
        except OSError:
            return False
    return True


class HttpSidecar:
    """Run and supervise ``python3 -m http.server`` for one directory."""

    def __init__(self, directory: Path, port: int = 8000, bind: str = '0.0.0.0',
                 advertise_host: str = '', log_path: Optional[Path] = None):
        self.directory = Path(directory)
        self.port = int(port)
        self.bind = bind
        self.advertise_host = advertise_host or detect_advertise_host()
        self.log_path = Path(log_path) if log_path else None
        self.process: Optional[subprocess.Popen] = None
        self.restarts = 0
        self.last_error = ''
        self._next_restart = 0.0

    @property
    def base_url(self) -> str:
        """Return the URL of the served directory root, ending in '/'."""
        return f'http://{self.advertise_host}:{self.port}/'

    def url_for(self, relative_path: str) -> str:
        """Return the URL of a path relative to the served directory."""
        return self.base_url + quote(relative_path.lstrip('/'))

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> bool:
        """Start the server. Return True if it is running after a short settle time."""
        if self.is_alive():
            return True
        self.directory.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-m', 'http.server', str(self.port),
               '--directory', str(self.directory), '--bind', self.bind]
        out = subprocess.DEVNULL
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            out = open(self.log_path, 'ab')
        try:
            # SIGTERM'd by the kernel if the collector dies: never orphan the port.
            self.process = subprocess.Popen(
                supervised_argv(cmd, os.getpid(), signal.SIGTERM), stdin=subprocess.DEVNULL,
                stdout=out, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
        except OSError as exc:
            self.last_error = f'failed to launch http.server: {exc}'
            return False
        finally:
            if out is not subprocess.DEVNULL:
                out.close()
        # http.server exits immediately if the port is taken; give it a moment to fail.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.last_error = (
                    f'http.server exited with code {self.process.returncode} '
                    f'(is port {self.port} already in use on {self.bind}?)')
                return False
            time.sleep(0.05)
        self.last_error = ''
        return True

    def ensure_running(self, backoff_sec: float = 5.0) -> bool:
        """Restart the server if it died (rate limited). Return True if it is running."""
        if self.is_alive():
            return True
        now = time.monotonic()
        if now < self._next_restart:
            return False
        self._next_restart = now + backoff_sec
        if self.process is not None:
            self.restarts += 1
        return self.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the server."""
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
