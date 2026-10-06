"""
rdc - command line client for remote_data_collector (and perception ground control).

Talk to one collector directly (no ground control needed)::

    rdc discover                                  # list collectors on the network
    rdc status  /mastcam_front
    rdc start   /mastcam_front -n trial01         # returns once recording is confirmed
    rdc stop    /mastcam_front --download ~/data  # stop, download, print the local path
    rdc start   /gantry -d 60 --wait --download ~/data --delete
    rdc list    /mastcam_front
    rdc download /mastcam_front mastcam_bags_2026-10-06T12-00-00 --dest ~/data
    rdc delete  /mastcam_front mastcam_bags_2026-10-06T12-00-00

Or drive the ground control node (what scripts/collect.sh uses)::

    rdc gc start trial --label run03
    rdc gc stop trial
    rdc gc status
    rdc gc new-session

Exit codes: 0 success, 1 failure, 2 usage error, 3 collector/ground control unreachable.
"""

import argparse
import os
import sys
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from .client import (CollectorError, discover_collectors, RecordingOutcome,
                     RemoteCollectorClient, spin_in_background)
from .download import DownloadError, format_bytes

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_UNREACHABLE = 0, 1, 2, 3


def _err(msg: str) -> None:
    print(f'rdc: {msg}', file=sys.stderr, flush=True)


class _Progress:
    """Single-line progress printer (only animates on a TTY)."""

    def __init__(self, label: str):
        self.label = label
        self.tty = sys.stdout.isatty()
        self._last = 0.0

    def __call__(self, done: int, total: int) -> None:
        now = time.monotonic()
        if not self.tty or (now - self._last < 0.2 and done < total):
            return
        self._last = now
        pct = f'{100.0 * done / total:5.1f}%' if total else ''
        print(f'\r  {self.label}: {format_bytes(done)} / {format_bytes(total)} {pct}   ',
              end='', flush=True)

    def finish(self) -> None:
        if self.tty:
            print()


def _print_outcome(outcome: RecordingOutcome) -> None:
    r = outcome.result
    print(f'{outcome.remote}: {outcome.status_name}: {r.message}')
    if r.bag.name:
        print(f'  bag:  {r.bag.name} ({format_bytes(r.bag.size_bytes)}, '
              f'{len(r.bag.files)} files, complete={r.bag.complete})')
        print(f'  url:  {r.bag.download_url}')
    if r.subscribed_topics:
        print(f'  subscribed ({len(r.subscribed_topics)}): ' + ', '.join(r.subscribed_topics))
    if r.missing_topics:
        print(f'  MISSING ({len(r.missing_topics)}): ' + ', '.join(r.missing_topics))
    if r.log_url:
        print(f'  log:  {r.log_url}')


def _download(client: RemoteCollectorClient, bag, dest: str, delete: bool,
              timeout: float) -> Optional[str]:
    progress = _Progress(bag.name)
    try:
        path = client.download(bag, dest, progress=progress, timeout=timeout)
    except Exception as exc:  # noqa: B902 - report, don't traceback
        progress.finish()
        _err(f'download failed: {exc}')
        return None
    progress.finish()
    print(f'  downloaded -> {path}')
    if delete:
        resp = client.delete_bag(bag.name)
        print(f'  deleted on collector ({format_bytes(resp.freed_bytes)} freed)')
    return str(path)


# ======================================================================== collector commands
def cmd_discover(node, args) -> int:
    found = discover_collectors(node, wait=args.wait)
    if not found:
        print('No collectors found (check ROS_DOMAIN_ID / network, or increase --wait).')
        return EXIT_FAIL
    for name in found:
        print(name)
    return EXIT_OK


def cmd_status(node, args) -> int:
    client = RemoteCollectorClient(node, args.remote)
    st = client.status(timeout=args.timeout) if client.wait_until_available(args.timeout) \
        else None
    if st is None:
        _err(f'{client.remote} not reachable')
        return EXIT_UNREACHABLE
    print(f'{st.collector_name} on {st.hostname}')
    print(f'  data_dir: {st.data_dir}  ({format_bytes(st.disk_free_bytes)} free of '
          f'{format_bytes(st.disk_total_bytes)})')
    print(f'  http:     {st.http_url or "(disabled)"}  alive={st.http_alive}')
    print(f'  profiles: {", ".join(st.profiles) or "(none)"}  '
          f'default: {", ".join(st.default_profiles) or "(none)"}')
    print(f'  active:   {len(st.active)} / {st.max_concurrent_recordings}')
    for a in st.active:
        dur = 'until stopped' if a.duration_sec <= 0 else f'{a.duration_sec:.0f}s'
        print(f'    {a.bag_name} [{a.state}] {a.elapsed_sec:.0f}s / {dur}, '
              f'{format_bytes(a.size_bytes)}, {len(a.subscribed_topics)} topics, '
              f'goal {a.goal_id}')
    return EXIT_OK


def cmd_start(node, args) -> int:
    client = RemoteCollectorClient(node, args.remote)
    tty = sys.stdout.isatty()

    def feedback(status):
        if args.wait:
            dur = '' if status.duration_sec <= 0 else f'/{status.duration_sec:.0f}'
            line = (f'{status.bag_name} [{status.state}] {status.elapsed_sec:.0f}{dur}s '
                    f'{format_bytes(status.size_bytes)} '
                    f'topics={len(status.subscribed_topics)}')
            print(('\r' + line + '   ') if tty else line, end='' if tty else '\n', flush=True)

    outcome_holder = {}
    started = client.start(bag_name=args.name, profiles=args.profile, topics=args.topic,
                           regex=args.regex, duration=args.duration, timeout=args.timeout,
                           feedback_callback=feedback,
                           done_callback=lambda o: outcome_holder.setdefault('o', o))
    print(f'{client.remote}: recording {started.bag_name} (goal {started.goal_id})')
    if not args.wait:
        print(f'Stop it with: rdc stop {client.remote}')
        return EXIT_OK
    try:
        while 'o' not in outcome_holder:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print('\nStopping...', flush=True)
        client.cancel([started.goal_id])
        outcome_holder['o'] = client.get_result(started.goal_id, timeout=args.stop_timeout)
    if tty:
        print()
    outcome = outcome_holder['o']
    _print_outcome(outcome)
    if not outcome.ok:
        return EXIT_FAIL
    if args.download:
        if not _download(client, outcome.bag, args.download, args.delete, args.timeout):
            return EXIT_FAIL
    return EXIT_OK


def cmd_stop(node, args) -> int:
    client = RemoteCollectorClient(node, args.remote)
    outcomes = client.stop(bag_names=args.bag or None, timeout=args.stop_timeout)
    if not outcomes:
        print(f'{client.remote}: nothing was recording')
        return EXIT_OK
    rc = EXIT_OK
    for outcome in outcomes:
        _print_outcome(outcome)
        if not outcome.ok:
            rc = EXIT_FAIL
        elif args.download and not _download(client, outcome.bag, args.download, args.delete,
                                             args.timeout):
            rc = EXIT_FAIL
    return rc


def cmd_list(node, args) -> int:
    client = RemoteCollectorClient(node, args.remote)
    client.require_available(args.timeout)
    bags = client.list_bags(args.pattern, timeout=args.timeout)
    for b in bags:
        flag = 'RECORDING' if b.active else ('ok' if b.complete else 'INCOMPLETE')
        stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(b.modified.sec))
        print(f'{b.name:55s} {format_bytes(b.size_bytes):>10s}  {stamp}  {flag}')
    if not bags:
        print('(no bags)')
    return EXIT_OK


def cmd_download(node, args) -> int:
    client = RemoteCollectorClient(node, args.remote)
    client.require_available(args.timeout)
    rc = EXIT_OK
    for name in args.bags:
        bag = client.find_bag(name, timeout=args.timeout)
        if bag.active or not bag.complete:
            _err(f'{name} is {"still recording" if bag.active else "incomplete"}; skipping')
            rc = EXIT_FAIL
            continue
        if not _download(client, bag, args.dest, args.delete, args.timeout):
            rc = EXIT_FAIL
    return rc


def cmd_delete(node, args) -> int:
    client = RemoteCollectorClient(node, args.remote)
    client.require_available(args.timeout)
    for name in args.bags:
        resp = client.delete_bag(name, timeout=args.timeout)
        print(f'{resp.message} ({format_bytes(resp.freed_bytes)} freed)')
    return EXIT_OK


# ======================================================================== ground control
def _gc_call(node, args, srv_type, endpoint: str, request, timeout: float):
    from .client import wait_future
    name = f'{args.gc_node.rstrip("/")}/{endpoint}'
    cli = node.create_client(srv_type, name)
    if not cli.wait_for_service(timeout_sec=args.timeout):
        raise CollectorError(f'ground control service {name} not reachable (is ground_control '
                             'running?)')
    return wait_future(cli.call_async(request), timeout, name)


def cmd_gc(node, args) -> int:
    from remote_data_collector_interfaces.srv import StartCollection, StopCollection
    from std_srvs.srv import Trigger
    if args.gc_cmd == 'start':
        req = StartCollection.Request(targets=args.targets, label=args.label,
                                      duration=float(args.duration))
        resp = _gc_call(node, args, StartCollection, 'start', req, args.stop_timeout)
    elif args.gc_cmd == 'stop':
        req = StopCollection.Request(targets=args.targets, wait_for_download=not args.no_wait)
        resp = _gc_call(node, args, StopCollection, 'stop', req, args.download_timeout)
    elif args.gc_cmd == 'status':
        resp = _gc_call(node, args, Trigger, 'status', Trigger.Request(), args.timeout)
    else:
        resp = _gc_call(node, args, Trigger, 'new_session', Trigger.Request(), args.timeout)
    print(resp.message)
    return EXIT_OK if resp.success else EXIT_FAIL


# ======================================================================== argument parsing
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='rdc', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--timeout', type=float, default=15.0,
                   help='seconds to wait for discovery / ordinary requests (default 15)')
    p.add_argument('--stop-timeout', type=float, default=180.0,
                   help='seconds to wait for a recording to finalize (default 180)')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('discover', help='list collectors visible on the ROS graph')
    s.add_argument('--wait', type=float, default=3.0, help='discovery time (default 3s)')
    s.set_defaults(func=cmd_discover)

    s = sub.add_parser('status', help='show collector health and active recordings')
    s.add_argument('remote', help='collector node name, e.g. /mastcam_front')
    s.set_defaults(func=cmd_status)

    s = sub.add_parser('start', help='start a recording')
    s.add_argument('remote')
    s.add_argument('-n', '--name', default='', help="bag name prefix (default: the collector's)")
    s.add_argument('-p', '--profile', action='append', default=[],
                   help="topic profile (repeatable; default: the collector's default_profiles)")
    s.add_argument('-t', '--topic', action='append', default=[], help='extra topic (repeatable)')
    s.add_argument('-e', '--regex', default='', help='extra topic regex')
    s.add_argument('-d', '--duration', type=float, default=0.0,
                   help='seconds to record (default 0 = until stopped)')
    s.add_argument('-w', '--wait', action='store_true',
                   help='stay attached: show progress, Ctrl-C stops the recording')
    s.add_argument('--download', metavar='DIR', help='with --wait: download the bag to DIR')
    s.add_argument('--delete', action='store_true',
                   help='with --download: delete the bag on the collector afterwards')
    s.set_defaults(func=cmd_start)

    s = sub.add_parser('stop', help='stop recordings (all, or --bag NAME)')
    s.add_argument('remote')
    s.add_argument('--bag', action='append', default=[], help='only stop this bag (repeatable)')
    s.add_argument('--download', metavar='DIR', help='download the bag(s) to DIR')
    s.add_argument('--delete', action='store_true',
                   help='with --download: delete the bag on the collector afterwards')
    s.set_defaults(func=cmd_stop)

    s = sub.add_parser('list', help='list bags stored on a collector')
    s.add_argument('remote')
    s.add_argument('pattern', nargs='?', default='', help='shell-style filter, e.g. "trial*"')
    s.set_defaults(func=cmd_list)

    s = sub.add_parser('download', help='download stored bag(s)')
    s.add_argument('remote')
    s.add_argument('bags', nargs='+')
    s.add_argument('--dest', default='.', help='destination directory (default .)')
    s.add_argument('--delete', action='store_true', help='delete on collector afterwards')
    s.set_defaults(func=cmd_download)

    s = sub.add_parser('delete', help='delete stored bag(s) on a collector')
    s.add_argument('remote')
    s.add_argument('bags', nargs='+')
    s.set_defaults(func=cmd_delete)

    s = sub.add_parser('gc', help='call the perception ground control node')
    s.add_argument('--gc-node', default=os.environ.get('GROUND_CONTROL_NODE', '/ground_control'),
                   help='ground control node name (default /ground_control or '
                        '$GROUND_CONTROL_NODE)')
    s.add_argument('--download-timeout', type=float, default=3600.0,
                   help='seconds to wait for "gc stop" incl. downloads (default 3600)')
    gsub = s.add_subparsers(dest='gc_cmd', required=True)
    g = gsub.add_parser('start', help='start recording on remotes/groups')
    g.add_argument('targets', nargs='*', help='remote and/or group names (default group if none)')
    g.add_argument('-l', '--label', default='', help='label prepended to bag names')
    g.add_argument('-d', '--duration', type=float, default=0.0,
                   help='0 = remote default, >0 seconds, <0 = until stopped')
    g = gsub.add_parser('stop', help='stop, download (and optionally delete) bags')
    g.add_argument('targets', nargs='*', help='remote and/or group names (all if none)')
    g.add_argument('--no-wait', action='store_true',
                   help='return before downloads finish (they continue in ground control)')
    gsub.add_parser('status', help='ground control summary')
    gsub.add_parser('new-session', help='start a new local session directory')
    s.set_defaults(func=cmd_gc)
    return p


def run(argv: List[str], node: Node) -> int:
    """Run one rdc command using an existing, already spinning node. Return the exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(node, args)
    except CollectorError as exc:
        _err(str(exc))
        return EXIT_UNREACHABLE if 'not reachable' in str(exc) else EXIT_FAIL
    except DownloadError as exc:
        _err(f'download failed: {exc}')
        return EXIT_FAIL


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    build_parser().parse_args(argv)  # usage errors exit(2) before touching ROS
    # Keep SIGINT as KeyboardInterrupt so `start --wait` can cancel the goal on Ctrl-C.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node(f'rdc_cli_{os.getpid()}')
    spinner = spin_in_background(node)
    try:
        return run(argv, node)
    except KeyboardInterrupt:
        _err('interrupted')
        return EXIT_FAIL
    finally:
        spinner.shutdown(timeout_sec=5.0)
        node.destroy_node()
        rclpy.try_shutdown()


def entry_point() -> None:
    sys.exit(main())
