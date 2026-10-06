"""
Lifecycle management of one ``ros2 bag record`` subprocess.

Notes on robustness (learned the hard way with the legacy capture services):

* ``ros2 bag record`` only writes ``metadata.yaml`` when it shuts down cleanly on SIGINT. If
  the parent process was started with SIGINT *ignored* (e.g. a bash background job, some
  service managers), the child inherits the ignored disposition across ``exec`` and silently
  ignores our stop request, leaving an un-indexed bag behind. We therefore launch the recorder
  through a tiny trampoline (procutil.supervised_argv) that restores default signal handling
  before ``exec`` and makes the kernel SIGINT the recorder if the collector dies.
* The recorder runs in its own session/process group so a Ctrl-C in the collector's terminal
  does not race with the collector's own, orderly shutdown of the recorder.
* The recorder log (stdout+stderr) is kept on disk next to the data so failures can be
  diagnosed after the fact (and downloaded over HTTP).
"""

import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

from .procutil import supervised_argv

_SUBSCRIBED_RE = re.compile(r"Subscribed to topic '([^']+)'")
_RECORDING_MARKERS = ('Recording...', 'Listening for topics', 'Starting recording to')


def dir_size(path: Path) -> int:
    """Return the total size in bytes of all regular files below `path` (0 if missing)."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
    except OSError:
        return 0
    return total


class RecorderError(RuntimeError):
    """Raised when the recorder cannot be started."""


class BagRecorder:
    """Start, monitor and gracefully stop one ``ros2 bag record`` process."""

    def __init__(
        self,
        cmd: Sequence[str],
        bag_path: Path,
        log_path: Path,
        env: Optional[Dict[str, str]] = None,
    ):
        self.cmd = list(cmd)
        self.bag_path = Path(bag_path)
        self.log_path = Path(log_path)
        self.env = env
        self.process: Optional[subprocess.Popen] = None
        self.started_monotonic: Optional[float] = None
        self.stopped_monotonic: Optional[float] = None
        self._log_offset = 0
        self._subscribed: List[str] = []
        self._saw_recording_marker = False
        self._lock = threading.Lock()  # serializes stop()
        self._log_lock = threading.Lock()  # serializes log parsing (status + execute threads)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Spawn the recorder. Raise RecorderError if it cannot be executed."""
        if self.process is not None:
            raise RecorderError('recorder already started')
        if self.bag_path.exists():
            raise RecorderError(f'output path already exists: {self.bag_path}')
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ if self.env is None else self.env)
        # Make sure recorder log lines reach the file promptly.
        env.setdefault('RCUTILS_LOGGING_BUFFERED_STREAM', '0')
        env.setdefault('PYTHONUNBUFFERED', '1')
        # Default signal handling + SIGINT (= finalize the bag) if the collector dies.
        argv = supervised_argv(self.cmd, os.getpid(), signal.SIGINT)
        with open(self.log_path, 'ab') as log:
            log.write(('$ ' + ' '.join(self.cmd) + '\n').encode())
            log.flush()
            try:
                self.process = subprocess.Popen(
                    argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True, close_fds=True)
            except OSError as exc:
                raise RecorderError(f'failed to launch {self.cmd[0]!r}: {exc}') from exc
        self.started_monotonic = time.monotonic()

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process else None

    @property
    def returncode(self) -> Optional[int]:
        return self.process.poll() if self.process else None

    def is_running(self) -> bool:
        """Return True while the recorder process is alive."""
        return self.process is not None and self.process.poll() is None

    def elapsed(self) -> float:
        """Return seconds since start (frozen once the recorder has been stopped)."""
        if self.started_monotonic is None:
            return 0.0
        end = self.stopped_monotonic or time.monotonic()
        return max(0.0, end - self.started_monotonic)

    def wait_until_recording(
        self,
        timeout: float,
        abort_check: Optional[Callable[[], bool]] = None,
        poll: float = 0.1,
    ) -> bool:
        """
        Block until the recorder is writing, it exits, `abort_check()` is true, or timeout.

        Return True only if the recorder is confirmed to be recording.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_running():
                return False
            self._scan_log()
            if self._saw_recording_marker and self.bag_path.exists():
                return True
            if abort_check is not None and abort_check():
                return False
            time.sleep(poll)
        return False

    def stop(self, sigint_timeout: float = 60.0, sigterm_timeout: float = 10.0) -> Optional[int]:
        """
        Stop the recorder: SIGINT (graceful), then SIGTERM, then SIGKILL.

        Return the exit code (None if it never started). Safe to call more than once.
        """
        with self._lock:
            if self.process is None:
                return None
            for sig, timeout in ((signal.SIGINT, sigint_timeout),
                                 (signal.SIGTERM, sigterm_timeout),
                                 (signal.SIGKILL, 10.0)):
                if self.process.poll() is not None:
                    break
                self._signal_group(sig)
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    continue
            if self.stopped_monotonic is None:
                self.stopped_monotonic = time.monotonic()
            self._scan_log()
            return self.process.poll()

    def signal(self, sig: int) -> None:
        """Send `sig` to the recorder's process group without waiting."""
        if self.process is not None and self.process.poll() is None:
            self._signal_group(sig)

    def _signal_group(self, sig: int) -> None:
        try:
            os.killpg(self.process.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            self.process.send_signal(sig)

    # ------------------------------------------------------------------ introspection
    def _scan_log(self) -> None:
        """Incrementally parse new recorder log output."""
        with self._log_lock:
            try:
                with open(self.log_path, 'rb') as f:
                    f.seek(self._log_offset)
                    chunk = f.read()
            except OSError:
                return
            # Only consume complete lines so a topic name is never split across reads.
            end = chunk.rfind(b'\n')
            if end < 0:
                return
            self._log_offset += end + 1
            text = chunk[:end + 1].decode('utf-8', errors='replace')
            for match in _SUBSCRIBED_RE.finditer(text):
                if match.group(1) not in self._subscribed:
                    self._subscribed.append(match.group(1))
            if any(marker in text for marker in _RECORDING_MARKERS):
                self._saw_recording_marker = True

    def subscribed_topics(self) -> List[str]:
        """Return topics the recorder reported subscribing to so far (best effort)."""
        self._scan_log()
        return list(self._subscribed)

    def size_bytes(self) -> int:
        """Return the current on-disk size of the bag folder."""
        return dir_size(self.bag_path)

    def log_tail(self, lines: int = 15) -> str:
        """Return the last `lines` lines of the recorder log."""
        try:
            with open(self.log_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 16384))
                data = f.read().decode('utf-8', errors='replace')
        except OSError:
            return ''
        return '\n'.join(data.splitlines()[-lines:])


def reindex_bag(
    bag_path: Path,
    ros2_executable: str = 'ros2',
    timeout: float = 300.0,
    log_path: Optional[Path] = None,
) -> bool:
    """Try to regenerate metadata.yaml for an unfinalized bag. Return True on success."""
    if (Path(bag_path) / 'metadata.yaml').exists():
        return True
    out = open(log_path, 'ab') if log_path else subprocess.DEVNULL
    try:
        subprocess.run([ros2_executable, 'bag', 'reindex', str(bag_path)],
                       stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                       timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    finally:
        if log_path:
            out.close()
    return (Path(bag_path) / 'metadata.yaml').exists()


def process_cmdline(pid: int) -> str:
    """Return the command line of a running process ('' if unavailable)."""
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            return f.read().replace(b'\0', b' ').decode('utf-8', errors='replace')
    except OSError:
        return ''


def stop_orphan(pid: int, expected_fragment: str, timeout: float = 60.0) -> bool:
    """
    SIGINT a leftover recorder from a previous collector run if it still matches.

    Return True if a matching process was found (and asked to stop).
    """
    if expected_fragment not in process_cmdline(pid):
        return False
    try:
        os.killpg(pid, signal.SIGINT)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and process_cmdline(pid):
        time.sleep(0.2)
    if process_cmdline(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return True
