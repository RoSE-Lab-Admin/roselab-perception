"""
Child-process launching helpers.

:func:`supervised_argv` wraps a command in a tiny Python trampoline that, in the child and
before ``exec``:

* restores default SIGINT/SIGTERM handling (a parent started with SIGINT *ignored*, e.g. as a
  bash background job, would otherwise pass the ignored disposition on and the child could
  never be stopped gracefully), and
* on Linux, asks the kernel to send ``death_signal`` to the child when its parent dies
  (``PR_SET_PDEATHSIG``), so a crashed or SIGKILLed collector never leaves an orphaned
  ``http.server`` holding the port or a recorder writing forever. Recorders get SIGINT, which
  makes ``ros2 bag record`` finalize the bag on its own.

The kernel delivers the death signal when the *thread* that spawned the child exits. The
collector spawns from long-lived threads (main thread / executor pool), so that is fine.
"""

import signal
import sys
from typing import List, Sequence

_TRAMPOLINE = """
import os, signal, sys
for s in (signal.SIGINT, signal.SIGTERM):
    signal.signal(s, signal.SIG_DFL)
signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
expected_parent, death_signal = int(sys.argv[1]), int(sys.argv[2])
if death_signal:
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, death_signal, 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:
        pass
    if os.getppid() != expected_parent:
        sys.exit(1)  # the parent died before prctl took effect
os.execvp(sys.argv[3], sys.argv[3:])
"""


def supervised_argv(cmd: Sequence[str], parent_pid: int,
                    death_signal: int = signal.SIGTERM) -> List[str]:
    """Return argv that runs `cmd` with default signals and a parent-death signal."""
    return [sys.executable, '-c', _TRAMPOLINE, str(parent_pid), str(int(death_signal))] + \
        list(cmd)
