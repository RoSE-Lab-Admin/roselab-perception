"""
Integration tests: RemoteDataCollector action server <-> RemoteCollectorClient, in one process.

The collector and the client share one rclpy context (no network discovery needed) and the
recorder is the fake `ros2` executable, so these tests are fast and deterministic. One test
also drives the real `ros2 bag record` (recording the collector's own /rosout) when available.
Fixtures (`harness_factory`, `ros_context`) live in conftest.py.
"""

import shutil
import time

from action_msgs.msg import GoalStatus
from conftest import Harness, wait_for
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('remote_data_collector_interfaces')

from remote_data_collector.client import CollectorError  # noqa: E402


# ------------------------------------------------------------------------------ tests
def test_status_reports_configuration(harness_factory):
    h = harness_factory()
    st = h.client.status()
    assert st.collector_name == '/' + h.name
    assert st.profiles == ['demo', 'tf']
    assert st.default_profiles == ['demo']
    assert st.http_alive
    assert st.http_url.startswith('http://127.0.0.1:')
    assert st.active == []
    assert st.disk_total_bytes > 0


def test_open_ended_record_stop_download_delete(harness_factory, tmp_path):
    h = harness_factory()
    feedback = []
    started = h.client.start(bag_name='trial01', feedback_callback=feedback.append)
    assert started.bag_name.startswith('trial01_')
    assert len(started.goal_id) == 32

    st = h.client.status()
    assert [a.bag_name for a in st.active] == [started.bag_name]
    assert wait_for(lambda: any(f.size_bytes > 0 for f in feedback))
    assert any(f.state == 'RECORDING' for f in feedback)
    listed = h.client.list_bags()
    assert listed[0].name == started.bag_name and listed[0].active and not listed[0].complete

    outcomes = h.client.stop()
    assert len(outcomes) == 1
    out = outcomes[0]
    assert out.status == GoalStatus.STATUS_CANCELED
    assert out.ok, out.result.message
    assert out.bag.name == started.bag_name and out.bag.complete
    assert 'metadata.yaml' in out.bag.files
    assert out.result.subscribed_topics == ['/demo/a']
    assert out.result.missing_topics == ['/demo/missing_b']
    assert out.result.recorded_sec > 0
    assert out.result.log_url.endswith(f'.rdc/logs/{started.bag_name}.log')
    assert h.client.status().active == []

    path = h.client.download(out.bag, tmp_path / 'ground')
    assert (path / 'metadata.yaml').is_file()
    assert sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) == out.bag.size_bytes

    resp = h.client.delete_bag(out.bag.name)
    assert resp.freed_bytes == out.bag.size_bytes
    assert h.client.list_bags() == []
    assert h.client.stop() == []  # nothing left to stop


def test_timed_recording_succeeds_and_calls_done(harness_factory):
    h = harness_factory()
    done = []
    started = h.client.start(duration=1.0, profiles=['tf'], topics=['/extra'],
                             done_callback=done.append)
    assert wait_for(lambda: done, timeout=15)
    out = done[0]
    assert out.status == GoalStatus.STATUS_SUCCEEDED, out.result.message
    assert out.ok and out.bag.name == started.bag_name
    assert 1.0 <= out.result.recorded_sec < 5.0
    assert out.result.subscribed_topics == ['/tf', '/extra']


def test_rejections_carry_reasons(harness_factory):
    h = harness_factory()
    with pytest.raises(CollectorError, match='unknown profile'):
        h.client.start(profiles=['nope'])
    with pytest.raises(CollectorError, match='invalid bag name'):
        h.client.start(bag_name='../escape')
    with pytest.raises(CollectorError, match='invalid topic'):
        h.client.start(topics=['bad topic'])
    h.client.start()
    with pytest.raises(CollectorError, match='busy'):
        h.client.start()
    assert len(h.client.status().active) == 1


def test_concurrent_recordings_and_stop_by_name(harness_factory):
    h = harness_factory(max_concurrent_recordings=2)
    a = h.client.start(bag_name='a')
    # A different client instance (as if another process) starts and stops by bag name.
    other = h.new_client()
    b = other.start(bag_name='b', profiles=['tf'])
    assert {x.bag_name for x in h.client.status().active} == {a.bag_name, b.bag_name}
    outs = other.stop(bag_names=[a.bag_name])
    assert [o.bag.name for o in outs] == [a.bag_name]
    assert [x.bag_name for x in h.client.status().active] == [b.bag_name]
    outs = h.client.stop(goal_ids=[b.goal_id])
    assert [o.bag.name for o in outs] == [b.bag_name] and outs[0].ok


def test_recorder_start_failure_aborts_with_log(harness_factory, fake_mode):
    h = harness_factory()
    fake_mode('fail_start')
    with pytest.raises(CollectorError, match='could not open storage'):
        h.client.start()
    assert h.client.status().active == []


def test_recorder_crash_is_reported_and_bag_salvaged(harness_factory, fake_mode):
    h = harness_factory()
    fake_mode('crash')
    done = []
    h.client.start(done_callback=done.append)
    assert wait_for(lambda: done, timeout=15)
    out = done[0]
    assert out.status == GoalStatus.STATUS_ABORTED
    assert 'exited unexpectedly' in out.result.message
    assert 'reindex' in out.result.message
    assert out.ok and out.bag.complete  # fake reindex regenerated metadata.yaml


def test_unresponsive_recorder_is_escalated(harness_factory, fake_mode):
    h = harness_factory(stop_timeout_sec=0.5, reindex_unfinished_bags=False)
    fake_mode('ignore_sigint')
    h.client.start()
    out = h.client.stop()[0]
    assert out.status == GoalStatus.STATUS_CANCELED
    assert not out.ok
    assert 'NOT finalized' in out.result.message
    assert 'exit code -15' in out.result.message


def test_delete_guards(harness_factory):
    h = harness_factory()
    started = h.client.start()
    with pytest.raises(CollectorError, match='being recorded'):
        h.client.delete_bag(started.bag_name)
    for bad in ['*', '../data', '']:
        with pytest.raises(CollectorError):
            h.client.delete_bag(bad)
    h.client.stop()
    h.client.delete_bag(started.bag_name)


def test_shutdown_finalizes_active_recordings(harness_factory):
    h = harness_factory()
    done = []
    h.client.start(done_callback=done.append)
    assert h.collector.shutdown_recordings(timeout=15)
    assert wait_for(lambda: done, timeout=10)
    out = done[0]
    assert out.status == GoalStatus.STATUS_ABORTED
    assert 'shutting down' in out.result.message
    assert out.ok and out.bag.complete
    with pytest.raises(CollectorError):
        h.client.start()  # new goals are refused while shutting down


def test_max_duration_safety_net(harness_factory):
    h = harness_factory(max_duration_sec=1.0)
    done = []
    h.client.start(done_callback=done.append)  # open-ended request
    assert wait_for(lambda: done, timeout=15)
    assert done[0].status == GoalStatus.STATUS_SUCCEEDED
    assert 'max_duration_sec' in done[0].result.message


def test_crash_recovery_reindexes_leftover_bag(ros_context, tmp_path, fake_ros2, fake_mode):
    fake_mode('normal')
    data = tmp_path / 'data'
    leftover = data / 'left_2026-01-01T00-00-00'
    leftover.mkdir(parents=True)
    (leftover / 'left_0.mcap').write_bytes(b'x')
    (data / '.rdc' / 'active').mkdir(parents=True)
    (data / '.rdc' / 'active' / f'{leftover.name}.json').write_text(
        '{"pid": 999999999, "bag_path": "%s"}' % leftover)
    h = Harness(tmp_path, fake_ros2)
    try:
        assert (leftover / 'metadata.yaml').exists()
        assert not (data / '.rdc' / 'active' / f'{leftover.name}.json').exists()
        assert h.client.list_bags()[0].complete
    finally:
        h.close()


def test_bad_configuration_fails_fast(ros_context, tmp_path, fake_ros2):
    from remote_data_collector.naming import ValidationError
    with pytest.raises(ValidationError, match='default_profiles'):
        Harness(tmp_path, fake_ros2, default_profiles=['missing'])
    with pytest.raises(ValidationError, match='unknown key'):
        Harness(tmp_path, fake_ros2, **{'profiles.demo.topicz': ['/x']})


def test_forced_exit_signal_finalizes_recorders(harness_factory):
    """The second-Ctrl-C path: recorders are told to finalize on their own (no orphans)."""
    import signal
    h = harness_factory()
    done = []
    h.client.start(done_callback=done.append)
    assert len(h.collector.store.active_markers()) == 1  # crash-recovery marker exists
    h.collector.signal_recorders(signal.SIGINT)
    assert wait_for(lambda: done, timeout=15)
    out = done[0]
    assert out.status == GoalStatus.STATUS_ABORTED and 'exited unexpectedly' in out.result.message
    assert out.ok and out.bag.complete  # the recorder finalized the bag itself


@pytest.mark.skipif(shutil.which('ros2') is None, reason='real ros2 CLI not on PATH')
def test_real_ros2_bag_record(harness_factory, tmp_path):
    """End-to-end with the real recorder: record the collector's own /rosout."""
    h = harness_factory(ros2_executable='ros2', startup_timeout_sec=30.0,
                        stop_timeout_sec=30.0, **{'profiles.demo.topics': ['/rosout']})
    started = h.client.start(timeout=45.0)
    time.sleep(2.0)
    out = h.client.stop(timeout=60.0)[0]
    assert out.ok, out.result.message
    assert out.bag.name == started.bag_name
    assert 'metadata.yaml' in out.bag.files
    assert any(f.endswith('.mcap') for f in out.bag.files)
    path = h.client.download(out.bag, tmp_path / 'ground')
    assert (path / 'metadata.yaml').read_text().startswith('rosbag2_bagfile_information')
