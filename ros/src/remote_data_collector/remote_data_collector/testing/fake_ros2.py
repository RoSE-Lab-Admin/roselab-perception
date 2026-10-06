#!/usr/bin/env python3
"""
Minimal stand-in for the `ros2` executable, used by the tests.

Implements just enough of ``ros2 bag record`` and ``ros2 bag reindex`` to exercise the
collector without DDS traffic. Behaviour is selected with environment variables (read when the
recorder is spawned):

FAKE_ROS2_MODE   normal | fail_start | slow_start | crash | ignore_sigint
FAKE_ROS2_DELAY  seconds to sleep before "recording" in slow_start mode (default 30)
FAKE_ROS2_REINDEX ok | fail   behaviour of `ros2 bag reindex`

Topics containing "missing" are never reported as subscribed.
"""

import argparse
import os
from pathlib import Path
import signal
import sys
import time


def log(msg: str) -> None:
    print(f'[INFO] [{time.time():.6f}] [fake_recorder]: {msg}', flush=True)


def record(argv) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--topics', nargs='+', default=[])
    p.add_argument('--regex', default='')
    p.add_argument('--node-name', default='fake')
    args, _unknown = p.parse_known_args(argv)
    mode = os.environ.get('FAKE_ROS2_MODE', 'normal')
    out = Path(args.output)

    if mode == 'fail_start':
        print('[ERROR] [fake_recorder]: could not open storage (simulated)', flush=True)
        return 3
    if out.exists():
        print(f"[ERROR] [fake_recorder]: Output folder '{out}' already exists.", flush=True)
        return 1
    if mode == 'slow_start':
        time.sleep(float(os.environ.get('FAKE_ROS2_DELAY', '30')))

    stop = {'flag': False}

    def on_sigint(_signum, _frame):
        stop['flag'] = True

    if mode == 'ignore_sigint':
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGINT, on_sigint)

    out.mkdir(parents=True)
    log(f"Starting recording to '{out}'")
    for topic in args.topics:
        if 'missing' not in topic:
            log(f"Subscribed to topic '{topic}'")
    log('Recording...')
    data = out / f'{out.name}_0.mcap'
    n = 0
    with open(data, 'ab') as f:
        while not stop['flag']:
            f.write(b'\0' * 4096)
            f.flush()
            time.sleep(0.05)
            n += 1
            if mode == 'crash' and n >= 10:
                os._exit(9)  # die without writing metadata.yaml
    (out / 'metadata.yaml').write_text(
        'rosbag2_bagfile_information:\n  version: 9\n  storage_identifier: mcap\n')
    log('Recording stopped')
    return 0


def reindex(argv) -> int:
    path = Path(argv[0])
    if os.environ.get('FAKE_ROS2_REINDEX', 'ok') != 'ok':
        print('reindex failed (simulated)', flush=True)
        return 1
    (path / 'metadata.yaml').write_text('rosbag2_bagfile_information:\n  reindexed: true\n')
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv[:2] == ['bag', 'record']:
        return record(argv[2:])
    if argv[:2] == ['bag', 'reindex']:
        return reindex(argv[2:])
    print(f'fake_ros2: unsupported command {argv}', file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
