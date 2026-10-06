"""Unit tests for ground control fleet configuration and target resolution (no ROS)."""

from perception_ground_control.fleet import (
    FleetConfigError, groups_from_params, remotes_from_flat_params, resolve_targets)
import pytest

FLAT = {
    'mastcam_front.node': 'mastcam_front',
    'mastcam_front.bag_name': 'mastcam_bags',
    'mastcam_front.delete_after_download': True,
    'gantry.bag_name': 'lidar_bags',
    'gantry.profiles': ['lidar_scan'],
    'gantry.duration': 60,
    'ground_station.download': False,
}


def test_remote_parsing_defaults():
    remotes = remotes_from_flat_params(FLAT)
    assert remotes['mastcam_front'].node == '/mastcam_front'
    assert remotes['mastcam_front'].delete_after_download
    assert remotes['gantry'].node == '/gantry'  # default "/<name>"
    assert remotes['gantry'].duration == 60.0
    assert remotes['gantry'].profiles == ['lidar_scan']
    assert remotes['ground_station'].download is False
    assert remotes['ground_station'].bag_name == ''


def test_remote_parsing_errors():
    with pytest.raises(FleetConfigError, match='unknown key'):
        remotes_from_flat_params({'x.nodes': '/x'})
    with pytest.raises(FleetConfigError):
        remotes_from_flat_params({'x': '/x'})


def test_duration_and_label_rules():
    r = remotes_from_flat_params(FLAT)
    gantry, mast = r['gantry'], r['mastcam_front']
    assert gantry.effective_duration(0) == 60.0     # 0 -> configured default
    assert gantry.effective_duration(15) == 15.0    # >0 -> override
    assert gantry.effective_duration(-1) == 0.0     # <0 -> until stopped
    assert mast.effective_duration(0) == 0.0
    assert mast.bag_prefix('') == 'mastcam_bags'
    assert mast.bag_prefix(' run03 ') == 'run03_mastcam_bags'
    assert r['ground_station'].bag_prefix('run03') == 'run03'


def test_groups_and_target_resolution():
    remotes = remotes_from_flat_params(FLAT)
    groups = groups_from_params({'trial': ['mastcam_front', 'ground_station'],
                                 'lidar': ['gantry']}, remotes)
    default = groups['trial']
    assert resolve_targets([], remotes, groups, default) == ['mastcam_front', 'ground_station']
    assert resolve_targets(['lidar', 'trial', 'gantry'], remotes, groups, default) == [
        'gantry', 'mastcam_front', 'ground_station']
    assert resolve_targets(['all'], remotes, groups, default) == list(remotes)
    assert resolve_targets(['  ', 'gantry'], remotes, groups, default) == ['gantry']
    with pytest.raises(FleetConfigError, match='unknown target'):
        resolve_targets(['nope'], remotes, groups, default)


def test_group_errors():
    remotes = remotes_from_flat_params(FLAT)
    with pytest.raises(FleetConfigError, match='unknown remote'):
        groups_from_params({'g': ['missing']}, remotes)
    with pytest.raises(FleetConfigError, match='same name'):
        groups_from_params({'gantry': ['gantry']}, remotes)
