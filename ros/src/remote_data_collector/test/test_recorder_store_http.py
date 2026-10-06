"""Unit tests for the recorder process manager, bag store, HTTP sidecar and downloader (no ROS)."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from remote_data_collector.bag_store import BagStore
from remote_data_collector.download import download_bag, DownloadError, format_bytes
from remote_data_collector.http_sidecar import HttpSidecar, port_is_free
from remote_data_collector.naming import ValidationError
from remote_data_collector.recorder import BagRecorder, dir_size, reindex_bag, stop_orphan
from remote_data_collector.testing import free_port


def _cmd(fake_ros2, bag_path, topics=('/a', '/missing_b')):
    return [fake_ros2, 'bag', 'record', '-o', str(bag_path), '--topics', *topics]


# ------------------------------------------------------------------------------ recorder
def test_recorder_start_stop_finalizes(tmp_path, fake_ros2, fake_mode):
    fake_mode('normal')
    bag = tmp_path / 'bag1'
    rec = BagRecorder(_cmd(fake_ros2, bag), bag, tmp_path / 'logs' / 'bag1.log')
    rec.start()
    assert rec.wait_until_recording(10.0)
    time.sleep(0.3)
    assert rec.is_running()
    assert rec.size_bytes() > 0
    assert rec.subscribed_topics() == ['/a']
    assert rec.stop(sigint_timeout=10) == 0
    assert (bag / 'metadata.yaml').is_file()
    assert not rec.is_running()
    elapsed = rec.elapsed()
    time.sleep(0.2)
    assert rec.elapsed() == elapsed  # frozen after stop
    assert rec.stop() == 0  # idempotent
    assert '$ ' in (tmp_path / 'logs' / 'bag1.log').read_text()  # command echoed in log


def test_recorder_survives_parent_ignoring_sigint(tmp_path, fake_ros2, fake_mode):
    """Regression: a parent with SIGINT ignored must still be able to stop the recorder."""
    fake_mode('normal')
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        bag = tmp_path / 'bag_ign'
        rec = BagRecorder(_cmd(fake_ros2, bag), bag, tmp_path / 'bag_ign.log')
        rec.start()
        assert rec.wait_until_recording(10.0)
        t0 = time.monotonic()
        assert rec.stop(sigint_timeout=10) == 0
        assert time.monotonic() - t0 < 5  # stopped by SIGINT, not by escalation
        assert (bag / 'metadata.yaml').is_file()
    finally:
        signal.signal(signal.SIGINT, previous)


def test_recorder_escalates_when_sigint_ignored_by_child(tmp_path, fake_ros2, fake_mode):
    fake_mode('ignore_sigint')
    bag = tmp_path / 'bag2'
    rec = BagRecorder(_cmd(fake_ros2, bag), bag, tmp_path / 'bag2.log')
    rec.start()
    assert rec.wait_until_recording(10.0)
    rc = rec.stop(sigint_timeout=0.5, sigterm_timeout=5)
    assert rc == -signal.SIGTERM
    assert not (bag / 'metadata.yaml').exists()


def test_recorder_start_failure(tmp_path, fake_ros2, fake_mode):
    fake_mode('fail_start')
    bag = tmp_path / 'bag3'
    rec = BagRecorder(_cmd(fake_ros2, bag), bag, tmp_path / 'bag3.log')
    rec.start()
    assert not rec.wait_until_recording(10.0)
    assert rec.returncode == 3
    assert 'could not open storage' in rec.log_tail()


def test_recorder_wait_aborts_on_check(tmp_path, fake_ros2, fake_mode):
    fake_mode('slow_start', delay=30)
    bag = tmp_path / 'bag4'
    rec = BagRecorder(_cmd(fake_ros2, bag), bag, tmp_path / 'bag4.log')
    rec.start()
    t0 = time.monotonic()
    assert not rec.wait_until_recording(20.0, abort_check=lambda: time.monotonic() - t0 > 0.5)
    assert time.monotonic() - t0 < 5
    rec.stop(sigint_timeout=5)


def test_recorder_refuses_existing_output(tmp_path, fake_ros2):
    bag = tmp_path / 'exists'
    bag.mkdir()
    from remote_data_collector.recorder import RecorderError
    with pytest.raises(RecorderError):
        BagRecorder(_cmd(fake_ros2, bag), bag, tmp_path / 'x.log').start()


def test_reindex(tmp_path, fake_ros2, fake_mode):
    bag = tmp_path / 'unfinished'
    bag.mkdir()
    (bag / 'unfinished_0.mcap').write_bytes(b'x')
    fake_mode('normal', reindex='fail')
    assert not reindex_bag(bag, fake_ros2)
    fake_mode('normal', reindex='ok')
    assert reindex_bag(bag, fake_ros2)
    assert (bag / 'metadata.yaml').exists()


def test_stop_orphan(tmp_path):
    marker = f'orphan-marker-{os.getpid()}'
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', marker],
                            start_new_session=True)
    try:
        assert not stop_orphan(proc.pid, 'some-other-command', timeout=1)
        assert proc.poll() is None
        assert stop_orphan(proc.pid, marker, timeout=5)
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()


# ------------------------------------------------------------------------------ bag store
def _make_bag(store, name, complete=True, payload=b'abc'):
    path = store.data_dir / name
    (path / 'sub').mkdir(parents=True)
    (path / f'{name}_0.mcap').write_bytes(payload)
    (path / 'sub' / 'extra.bin').write_bytes(b'1234')
    if complete:
        (path / 'metadata.yaml').write_text('x: 1\n')
    return path


def test_store_listing_and_records(tmp_path):
    store = BagStore(tmp_path / 'data')
    store.ensure()
    _make_bag(store, 'a_bag', complete=True)
    time.sleep(0.01)
    _make_bag(store, 'b_bag', complete=False)
    (store.data_dir / 'loose_file.txt').write_text('ignored')
    names = [r.name for r in store.list_bags()]
    assert names == ['b_bag', 'a_bag']  # newest first, .rdc and files ignored
    rec = store.bag_record('a_bag')
    assert rec.complete and not rec.active
    assert sorted(rec.files) == ['a_bag_0.mcap', 'metadata.yaml', 'sub/extra.bin']
    assert rec.size_bytes == 3 + 4 + 5
    assert not store.bag_record('b_bag').complete
    assert store.bag_record('a_bag', active_names=['a_bag']).active
    assert [r.name for r in store.list_bags('a_*')] == ['a_bag']


def test_store_delete_is_safe(tmp_path):
    store = BagStore(tmp_path / 'data')
    store.ensure()
    _make_bag(store, 'keep')
    _make_bag(store, 'victim')
    outside = tmp_path / 'outside'
    outside.mkdir()
    for bad in ['*', '../outside', '/etc', '', '.rdc', 'kee?']:
        with pytest.raises((ValidationError, FileNotFoundError)):
            store.delete(bad)
    with pytest.raises(ValidationError, match='being recorded'):
        store.delete('victim', active_names=['victim'])
    with pytest.raises(FileNotFoundError):
        store.delete('does_not_exist')
    store.log_path('victim').write_text('log')
    freed = store.delete('victim')
    assert freed > 0
    assert not (store.data_dir / 'victim').exists()
    assert not store.log_path('victim').exists()
    assert (store.data_dir / 'keep').exists() and outside.exists()


def test_store_unique_names_and_markers(tmp_path):
    from datetime import datetime
    store = BagStore(tmp_path / 'data')
    store.ensure()
    now = datetime(2026, 1, 2, 3, 4, 5)
    first = store.unique_bag_name('x', now)
    (store.data_dir / first).mkdir()
    second = store.unique_bag_name('x', now)
    third = store.unique_bag_name('x', now, reserved=[second])
    assert (first, second, third) == ('x_2026-01-02T03-04-05', 'x_2026-01-02T03-04-05-1',
                                      'x_2026-01-02T03-04-05-2')
    store.mark_active('bag', {'pid': 123})
    assert store.active_markers() == {'bag': {'pid': 123}}
    store.clear_active('bag')
    store.clear_active('bag')
    assert store.active_markers() == {}


def test_dir_size(tmp_path):
    assert dir_size(tmp_path / 'missing') == 0
    (tmp_path / 'f').write_bytes(b'x' * 10)
    assert dir_size(tmp_path) == 10


# ------------------------------------------------------------------------------ http + download
@pytest.fixture
def served_store(tmp_path):
    store = BagStore(tmp_path / 'served')
    store.ensure()
    _make_bag(store, 'dl_bag', payload=os.urandom(3 * 1024 * 1024 + 17))
    port = free_port()
    http = HttpSidecar(store.data_dir, port, '127.0.0.1', '127.0.0.1',
                       log_path=store.logs_dir / 'http.log')
    assert http.start(), http.last_error
    yield store, http
    http.stop()


def test_http_sidecar_and_download(served_store, tmp_path):
    store, http = served_store
    rec = store.bag_record('dl_bag')
    url = http.url_for('dl_bag/')
    assert url == f'http://127.0.0.1:{http.port}/dl_bag/'
    seen = []
    dest = tmp_path / 'dest'
    path = download_bag(url, rec.files, dest, 'dl_bag', rec.file_sizes,
                        progress=lambda done, total: seen.append((done, total)))
    assert path == dest / 'dl_bag'
    for rel, size in zip(rec.files, rec.file_sizes):
        assert (path / rel).stat().st_size == size
        assert (path / rel).read_bytes() == (rec.path / rel).read_bytes()
    assert seen[-1] == (rec.size_bytes, rec.size_bytes)
    assert not (dest / 'dl_bag.partial').exists()
    # Idempotent: same sizes -> reused without error.
    assert download_bag(url, rec.files, dest, 'dl_bag', rec.file_sizes) == path


def test_download_failures(served_store, tmp_path):
    store, http = served_store
    rec = store.bag_record('dl_bag')
    url = http.url_for('dl_bag/')
    with pytest.raises(DownloadError, match='unsafe'):
        download_bag(url, ['../etc/passwd'], tmp_path / 'd1', 'dl_bag')
    with pytest.raises(DownloadError, match='after 2 attempts'):
        download_bag(url, ['nope.bin'], tmp_path / 'd2', 'dl_bag', retries=2)
    with pytest.raises(DownloadError, match='expected'):
        download_bag(url, rec.files, tmp_path / 'd3', 'dl_bag',
                     [s + 1 for s in rec.file_sizes], retries=1)
    # Final folder never appears for a failed download.
    assert not (tmp_path / 'd3' / 'dl_bag').exists()
    with pytest.raises(DownloadError, match='no download URL'):
        download_bag('', rec.files, tmp_path / 'd4', 'dl_bag')
    # Existing, different destination is not silently overwritten.
    (tmp_path / 'd5' / 'dl_bag').mkdir(parents=True)
    with pytest.raises(DownloadError, match='already exists'):
        download_bag(url, rec.files, tmp_path / 'd5', 'dl_bag', rec.file_sizes)


def test_http_sidecar_port_conflict_and_restart(tmp_path):
    port = free_port()
    first = HttpSidecar(tmp_path, port, '127.0.0.1', '127.0.0.1')
    second = HttpSidecar(tmp_path, port, '127.0.0.1', '127.0.0.1')
    try:
        assert first.start()
        assert not port_is_free(port, '127.0.0.1')
        assert not second.start()
        assert 'already in use' in second.last_error
        first.process.kill()
        first.process.wait()
        assert not first.is_alive()
        time.sleep(0.2)
        assert first.ensure_running(backoff_sec=0)
        assert first.restarts == 1
    finally:
        first.stop()
        second.stop()


def test_format_bytes():
    assert format_bytes(0) == '0 B'
    assert format_bytes(1536) == '1.5 KiB'
    assert format_bytes(5 * 1024 ** 3) == '5.0 GiB'


def test_paths_are_posix_relative(tmp_path):
    store = BagStore(tmp_path)
    store.ensure()
    assert store.relative_url_path(store.logs_dir / 'x.log') == '.rdc/logs/x.log'
    assert Path(store.bag_path('ok')).parent == store.data_dir


def test_supervised_child_dies_with_parent(tmp_path):
    """Regression: a hard-killed collector must not leave orphans (e.g. http.server on 8000)."""
    pidfile = tmp_path / 'child.pid'
    parent_code = (
        'import os, subprocess, sys, time\n'
        'from remote_data_collector.procutil import supervised_argv\n'
        'child = subprocess.Popen(supervised_argv([sys.executable, "-c", '
        '"import time; time.sleep(120)"], os.getpid()), start_new_session=True)\n'
        f'open({str(pidfile)!r}, "w").write(str(child.pid))\n'
        'time.sleep(120)\n')
    parent = subprocess.Popen([sys.executable, '-c', parent_code],
                              env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not pidfile.exists():
        time.sleep(0.05)
    child_pid = int(pidfile.read_text())
    time.sleep(0.5)  # let the trampoline set PR_SET_PDEATHSIG and exec
    parent.kill()
    parent.wait()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and Path(f'/proc/{child_pid}').exists():
        try:
            os.waitpid(child_pid, os.WNOHANG)
        except ChildProcessError:
            pass
        status = Path(f'/proc/{child_pid}/status')
        if status.exists() and 'State:\tZ' in status.read_text():
            break
        time.sleep(0.1)
    alive = Path(f'/proc/{child_pid}/status').exists() and \
        'State:\tZ' not in Path(f'/proc/{child_pid}/status').read_text()
    if alive:
        os.kill(child_pid, signal.SIGKILL)
    assert not alive, 'supervised child survived its parent'
