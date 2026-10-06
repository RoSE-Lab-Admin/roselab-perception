"""Unit tests for name validation, profile resolution and command construction (no ROS)."""

from datetime import datetime

import pytest
from remote_data_collector.naming import (
    goal_id_to_hex, hex_to_goal_id, join_regexes, make_bag_name, parse_bag_timestamp,
    validate_bag_name, validate_regex, validate_topic, ValidationError)
from remote_data_collector.profiles import (
    build_record_command, profiles_from_flat_params, resolve)


# ------------------------------------------------------------------------------ naming
@pytest.mark.parametrize('name', ['mastcam_bags', 'a', 'trial-03.v2', 'X_2026-10-06T12-00-00'])
def test_valid_bag_names(name):
    assert validate_bag_name(name) == name


@pytest.mark.parametrize('name', [
    '', '.', '..', '../etc', 'a/b', '/abs', '*', 'trial*', '-rf', '.hidden', 'sp ace', 'x' * 200,
    'semi;colon'])
def test_invalid_bag_names(name):
    with pytest.raises(ValidationError):
        validate_bag_name(name)


def test_topic_normalization_and_rejection():
    assert validate_topic('tf_static') == '/tf_static'  # legacy config had this typo
    assert validate_topic(' /MastCam/Front/color/image_raw ') == '/MastCam/Front/color/image_raw'
    for bad in ['', '/', '/a//b', '/a/', '-x', '/a b', '~/private', '/{ns}/x', '/1abc/../x']:
        with pytest.raises(ValidationError):
            validate_topic(bad)


def test_regex_validation():
    assert validate_regex('^/bno055/.*') == '^/bno055/.*'
    with pytest.raises(ValidationError):
        validate_regex('(unclosed')
    with pytest.raises(ValidationError):
        validate_regex('--all')


def test_join_regexes():
    assert join_regexes([]) == ''
    assert join_regexes(['', 'a']) == 'a'
    assert join_regexes(['a', 'b|c']) == '(?:a)|(?:b|c)'


def test_bag_name_timestamp_roundtrip():
    now = datetime(2026, 10, 6, 12, 34, 56)
    name = make_bag_name('mastcam_bags', now)
    assert name == 'mastcam_bags_2026-10-06T12-34-56'
    assert parse_bag_timestamp(name) == now
    assert parse_bag_timestamp(name + '-2') == now  # de-dup suffix
    assert parse_bag_timestamp('no_timestamp') is None


def test_goal_id_hex_roundtrip():
    raw = list(range(16))
    assert hex_to_goal_id(goal_id_to_hex(raw)) == raw
    with pytest.raises(ValidationError):
        hex_to_goal_id('abcd')


# ------------------------------------------------------------------------------ profiles
FLAT = {
    'tf.topics': ['/tf', 'tf_static'],
    'cam.topics': ['/cam/image', '/cam/info'],
    'cam.include': ['tf'],
    'cam.exclude_regex': '.*/theora$',
    'imu.regex': '^/bno055/.*',
    'imu.extra_args': ['--max-cache-size', '1000'],
    'everything.all_topics': True,
    'combo.include': ['cam', 'imu'],
}


def test_profiles_parse_and_resolve_union():
    profiles = profiles_from_flat_params(FLAT)
    spec = resolve(profiles, ['combo'], extra_topics=['/extra', '/tf'], extra_regex='^/joy/.*')
    assert spec.profiles == ['combo', 'cam', 'tf', 'imu']
    assert spec.topics == ['/cam/image', '/cam/info', '/tf', '/tf_static', '/extra']
    assert spec.regex == '(?:^/bno055/.*)|(?:^/joy/.*)'
    assert spec.exclude_regex == '.*/theora$'
    assert spec.extra_args == ['--max-cache-size', '1000']
    assert not spec.all_topics


def test_resolve_errors():
    profiles = profiles_from_flat_params(FLAT)
    with pytest.raises(ValidationError, match='unknown profile'):
        resolve(profiles, ['nope'])
    with pytest.raises(ValidationError, match='nothing to record'):
        resolve(profiles, [])
    with pytest.raises(ValidationError):
        resolve(profiles, ['tf'], extra_topics=['bad topic'])
    # A goal with only explicit topics and no profiles is fine.
    assert resolve(profiles, [], extra_topics=['/x']).topics == ['/x']


def test_profile_config_errors():
    with pytest.raises(ValidationError, match='unknown key'):
        profiles_from_flat_params({'a.topic': ['/x']})
    with pytest.raises(ValidationError, match='unknown profile'):
        profiles_from_flat_params({'a.include': ['b']})
    with pytest.raises(ValidationError, match='cycle'):
        profiles_from_flat_params({'a.include': ['b'], 'b.include': ['a']})
    with pytest.raises(ValidationError, match='profiles.<name>.<key>'):
        profiles_from_flat_params({'flat': ['/x']})
    with pytest.raises(ValidationError, match="profile 'a'"):
        profiles_from_flat_params({'a.regex': '(bad'})


def test_build_record_command_shape():
    profiles = profiles_from_flat_params(FLAT)
    spec = resolve(profiles, ['combo'])
    cmd = build_record_command(spec, '/data/bag_1', 'rdc_x', 'ros2', ['--storage', 'mcap'])
    assert cmd[:8] == ['ros2', 'bag', 'record', '-o', '/data/bag_1',
                       '--disable-keyboard-controls', '--node-name', 'rdc_x']
    assert cmd[8:10] == ['--storage', 'mcap']
    # The variadic --topics list must come last.
    idx = cmd.index('--topics')
    assert cmd[idx + 1:] == spec.topics
    assert cmd[cmd.index('--regex') + 1] == spec.regex
    assert cmd[cmd.index('--exclude-regex') + 1] == '.*/theora$'
    all_spec = resolve(profiles, ['everything'])
    assert '--all-topics' in build_record_command(all_spec, '/d/b', 'n')
