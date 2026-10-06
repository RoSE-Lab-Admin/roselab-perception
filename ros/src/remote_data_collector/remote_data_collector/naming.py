"""
Validation and naming helpers shared by the collector, its clients and the tests.

Everything that arrives in an action goal or service request is validated here before it is
used to build a file path or a command line.
"""

from datetime import datetime
import re
from typing import Iterable, List, Optional

# Same timestamp format as the legacy capture services, so existing tooling keeps working.
TIME_FORMAT = '%Y-%m-%dT%H-%M-%S'

MAX_BAG_NAME_LEN = 128

# Bag folder names: no path separators, no wildcards, no leading dot or dash.
_BAG_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.\-]*$')
# ROS topic names (absolute or relative), no substitutions, no leading dash.
_TOPIC_RE = re.compile(r'^/?[A-Za-z_~][A-Za-z0-9_/]*$')


class ValidationError(ValueError):
    """Raised when a user-supplied name, topic or pattern is invalid."""


def validate_bag_name(name: str) -> str:
    """Return `name` if it is a safe bag folder name, else raise ValidationError."""
    if not name:
        raise ValidationError('bag name is empty')
    if len(name) > MAX_BAG_NAME_LEN:
        raise ValidationError(f'bag name longer than {MAX_BAG_NAME_LEN} characters: {name!r}')
    if name in ('.', '..') or not _BAG_NAME_RE.match(name):
        raise ValidationError(
            f'invalid bag name {name!r}: use letters, digits, "_", "-" and "." only, '
            'starting with a letter or digit')
    return name


def validate_topic(topic: str) -> str:
    """Return a normalized (absolute) topic name or raise ValidationError."""
    topic = topic.strip()
    if not topic or not _TOPIC_RE.match(topic) or '//' in topic or topic.endswith('/'):
        raise ValidationError(f'invalid topic name {topic!r}')
    if topic.startswith('~'):
        raise ValidationError(f'private topic names are not supported: {topic!r}')
    if not topic.startswith('/'):
        topic = '/' + topic
    return topic


def validate_regex(pattern: str) -> str:
    """Return `pattern` if it compiles and cannot be mistaken for a CLI flag."""
    if pattern.startswith('-'):
        raise ValidationError(f'regex may not start with "-": {pattern!r}')
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValidationError(f'invalid regex {pattern!r}: {exc}') from exc
    return pattern


def join_regexes(patterns: Iterable[str]) -> str:
    """Combine several regexes into one alternation (empty patterns are dropped)."""
    parts = [p for p in patterns if p]
    if not parts:
        return ''
    if len(parts) == 1:
        return parts[0]
    return '|'.join(f'(?:{p})' for p in parts)


def dedupe(items: Iterable[str]) -> List[str]:
    """Return items without duplicates, preserving first-seen order."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def timestamp(now: Optional[datetime] = None) -> str:
    """Return the bag timestamp suffix for `now` (default: current local time)."""
    return (now or datetime.now()).strftime(TIME_FORMAT)


def make_bag_name(prefix: str, now: Optional[datetime] = None) -> str:
    """Return '<prefix>_<timestamp>' after validating the prefix."""
    validate_bag_name(prefix)
    return validate_bag_name(f'{prefix}_{timestamp(now)}')


def parse_bag_timestamp(name: str) -> Optional[datetime]:
    """Return the timestamp encoded at the end of a bag name, or None."""
    suffix = name.rsplit('_', 1)[-1]
    # Tolerate the "-N" de-duplication suffix added when two bags start in the same second.
    for candidate in (suffix, suffix.rsplit('-', 1)[0]):
        try:
            return datetime.strptime(candidate, TIME_FORMAT)
        except ValueError:
            continue
    return None


def goal_id_to_hex(uuid_bytes) -> str:
    """Return a goal UUID (sequence of 16 ints/bytes) as 32 lowercase hex characters."""
    return bytes(bytearray(uuid_bytes)).hex()


def hex_to_goal_id(hex_str: str) -> List[int]:
    """Return the 16 integer UUID bytes encoded by a 32 character hex string."""
    raw = bytes.fromhex(hex_str.strip())
    if len(raw) != 16:
        raise ValidationError(f'goal id must be 32 hex characters, got {hex_str!r}')
    return list(raw)
