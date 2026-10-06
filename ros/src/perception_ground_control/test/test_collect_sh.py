"""Tests for scripts/collect.sh using a stub `rdc` (checks the script's wiring, not ROS)."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / 'scripts' / 'collect.sh'

pytestmark = pytest.mark.skipif(not SCRIPT.is_file() or shutil.which('bash') is None,
                                reason='scripts/collect.sh not found (installed copy?)')

# Stub rdc: logs its arguments and answers like ground control would.
STUB = r"""#!/usr/bin/env bash
echo "$*" >> "${RDC_LOG}"
if [ "$1" = gc ] && [ "$2" = start ]; then
  if [[ " $* " == *" lidar "* ]]; then
    echo "Started 1/1 remote(s); session /tmp/s"
    echo "  OK    gantry: recording lidar_bags_X (for ${STUB_SECS:-1}s)"
  elif [ -n "${STUB_PARTIAL:-}" ]; then
    echo "Started 1/2 remote(s); session /tmp/s"
    echo "  OK    cam: recording mastcam_bags_X (until stopped)"
    echo "  FAIL  ground_station: not reachable"
    exit 1
  else
    echo "Started 1/1 remote(s); session /tmp/s"
    echo "  OK    cam: recording mastcam_bags_X (until stopped)"
  fi
  exit "${STUB_RC:-0}"
fi
echo "stub reply"
exit "${STUB_RC:-0}"
"""


@pytest.fixture
def env(tmp_path):
    prefix = tmp_path / 'prefix'
    rdc = prefix / 'lib' / 'remote_data_collector' / 'rdc'
    rdc.parent.mkdir(parents=True)
    rdc.write_text(STUB)
    rdc.chmod(0o755)
    e = dict(os.environ)
    e.update(AMENT_PREFIX_PATH=str(prefix), RDC_LOG=str(tmp_path / 'rdc.log'))
    return e, tmp_path / 'rdc.log'


def run(env, *args, stdin=''):
    return subprocess.run(['bash', str(SCRIPT)] + list(args), env=env, input=stdin, text=True,
                          capture_output=True, timeout=60)


def calls(log: Path):
    return log.read_text().splitlines() if log.exists() else []


def test_help_and_usage_errors(env):
    e, log = env
    r = run(e, '--help')
    assert r.returncode == 0 and 'collect.sh start' in r.stdout
    assert run(e).returncode == 2
    r = run(e, 'bogus')
    assert r.returncode == 2 and 'unknown command' in r.stderr
    r = run(e, 'start', '-d', 'abc')
    assert r.returncode == 2 and 'duration must be a number' in r.stderr
    r = run(e, 'start', '--frob')
    assert r.returncode == 2
    assert calls(log) == []  # nothing reached rdc


def test_start_stop_status_session_passthrough(env):
    e, log = env
    assert run(e, 'start').returncode == 0
    assert run(e, 'start', 'trial', 'gantry', '-l', 'run07', '-d', '30').returncode == 0
    assert run(e, 'stop').returncode == 0
    assert run(e, 'stop', 'all', '--no-wait').returncode == 0
    assert run(e, 'status').returncode == 0
    assert run(e, 'session').returncode == 0
    assert run(e, 'rdc', 'status', '/mastcam_front').returncode == 0
    assert calls(log) == [
        'gc start',
        'gc start trial gantry --label run07 --duration 30',
        'gc stop',
        'gc stop all --no-wait',
        'gc status',
        'gc new-session',
        'status /mastcam_front',
    ]


def test_failures_propagate_exit_code(env):
    e, _ = env
    e['STUB_RC'] = '3'
    assert run(e, 'start').returncode == 3
    assert run(e, 'stop').returncode == 3


def test_scan_waits_configured_duration_then_collects(env):
    e, log = env
    r = run(e, 'scan', '-l', 'pre')
    assert r.returncode == 0, r.stderr
    assert 'recording finished; collecting' in r.stdout
    assert calls(log) == ['gc start lidar --label pre', 'gc stop lidar']


def test_scan_with_seconds_and_record(env):
    e, log = env
    assert run(e, 'scan', '1').returncode == 0
    assert run(e, 'record', 'lidar', '-d', '1').returncode == 0
    assert run(e, 'record', 'lidar').returncode == 2  # needs -d
    assert run(e, 'scan', 'ten').returncode == 2
    assert calls(log) == ['gc start lidar --duration 1', 'gc stop lidar',
                          'gc start lidar --duration 1', 'gc stop lidar']


def test_open_ended_record_waits_for_enter(env):
    e, log = env
    # The stub reports "until stopped" for non-lidar targets, so the script waits for Enter
    # (bash only prints the read prompt on a TTY, so check the calls instead).
    r = run(e, 'record', 'trial', '-d', '5', stdin='\n')
    assert r.returncode == 0, r.stderr
    assert calls(log) == ['gc start trial --duration 5', 'gc stop trial']


def test_missing_workspace_is_reported(tmp_path):
    e = dict(os.environ, AMENT_PREFIX_PATH=str(tmp_path / 'nothing'),
             ROS_SETUP=str(tmp_path / 'none.bash'), WS_SETUP=str(tmp_path / 'none.bash'))
    r = run(e, 'status')
    assert r.returncode == 3 and 'not built/sourced' in r.stderr


def test_partial_start_failure_stops_what_started(env):
    """A failed start must not leave half the remotes recording (a retry would be 'busy')."""
    e, log = env
    e['STUB_PARTIAL'] = '1'
    r = run(e, 'record', 'trial', '-d', '5')
    assert r.returncode == 1
    assert 'Stopping the remotes that did start (cam)' in r.stdout
    assert calls(log) == ['gc start trial --duration 5', 'gc stop cam']
