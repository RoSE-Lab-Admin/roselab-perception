"""
Topic profiles: named, composable topic selections defined in the collector's parameters.

A profile is declared with ROS parameters under the ``profiles.<name>.`` prefix::

    profiles:
      mastcam:
        description: "Front RealSense, compressed color + depth"
        topics: ["/MastCam/Front/color/image_raw/compressed", ...]
        regex: "^/MastCam/Front/extrinsics/.*"     # optional
        exclude_regex: ".*/theora$"                 # optional
        exclude_topics: ["/MastCam/Front/color/metadata"]  # optional
        include: ["tf"]                             # optional, other profiles to merge in
        all_topics: false                           # optional, record everything
        extra_args: ["--max-cache-size", "200000000"]  # optional, raw `ros2 bag record` flags

A goal selects one or more profiles and may add explicit topics and a regex. The union of all
of it is turned into a single ``ros2 bag record`` command line.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .naming import dedupe, join_regexes, validate_regex, validate_topic, ValidationError

PROFILE_KEYS = (
    'description', 'topics', 'regex', 'exclude_regex', 'exclude_topics', 'include',
    'all_topics', 'extra_args',
)


@dataclass
class Profile:
    """One named topic selection."""

    name: str
    description: str = ''
    topics: List[str] = field(default_factory=list)
    regex: str = ''
    exclude_regex: str = ''
    exclude_topics: List[str] = field(default_factory=list)
    include: List[str] = field(default_factory=list)
    all_topics: bool = False
    extra_args: List[str] = field(default_factory=list)


@dataclass
class RecordSpec:
    """Fully resolved recording request (what `ros2 bag record` will be asked to do)."""

    profiles: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    regex: str = ''
    exclude_regex: str = ''
    exclude_topics: List[str] = field(default_factory=list)
    all_topics: bool = False
    extra_args: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return True if nothing would be recorded."""
        return not (self.all_topics or self.topics or self.regex)

    def describe(self) -> str:
        """Return a short human-readable summary for logs."""
        parts = []
        if self.profiles:
            parts.append('profiles=' + ','.join(self.profiles))
        if self.all_topics:
            parts.append('ALL TOPICS')
        if self.topics:
            parts.append(f'{len(self.topics)} topics')
        if self.regex:
            parts.append(f'regex={self.regex!r}')
        if self.exclude_regex:
            parts.append(f'exclude_regex={self.exclude_regex!r}')
        if self.exclude_topics:
            parts.append(f'{len(self.exclude_topics)} excluded topics')
        return ' '.join(parts) or '(nothing)'


def _as_list(value: Any, key: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Allow a single string, or a comma separated list, for convenience in YAML / CLI.
        return [v.strip() for v in value.split(',') if v.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    raise ValidationError(f'{key}: expected a list of strings, got {type(value).__name__}')


def profile_from_dict(name: str, data: Mapping[str, Any]) -> Profile:
    """Build and validate a Profile from a plain mapping."""
    unknown = set(data) - set(PROFILE_KEYS)
    if unknown:
        raise ValidationError(
            f'profile {name!r}: unknown key(s) {sorted(unknown)}; allowed: {list(PROFILE_KEYS)}')
    where = f'profile {name!r}'
    try:
        profile = Profile(
            name=name,
            description=str(data.get('description', '') or ''),
            topics=[validate_topic(t) for t in _as_list(data.get('topics'), 'topics')],
            regex=str(data.get('regex', '') or ''),
            exclude_regex=str(data.get('exclude_regex', '') or ''),
            exclude_topics=[
                validate_topic(t) for t in _as_list(data.get('exclude_topics'), 'exclude_topics')],
            include=_as_list(data.get('include'), 'include'),
            all_topics=bool(data.get('all_topics', False)),
            extra_args=_as_list(data.get('extra_args'), 'extra_args'),
        )
        if profile.regex:
            validate_regex(profile.regex)
        if profile.exclude_regex:
            validate_regex(profile.exclude_regex)
    except ValidationError as exc:
        raise ValidationError(f'{where}: {exc}') from exc
    return profile


def profiles_from_flat_params(flat: Mapping[str, Any]) -> Dict[str, Profile]:
    """
    Build profiles from flattened parameters ``{'<name>.<key>': value}``.

    This is the shape returned by ``Node.get_parameters_by_prefix('profiles')`` once the values
    have been unwrapped.
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for dotted, value in flat.items():
        if '.' not in dotted:
            raise ValidationError(
                f'parameter "profiles.{dotted}" must look like profiles.<name>.<key>')
        name, key = dotted.split('.', 1)
        grouped.setdefault(name, {})[key] = value
    profiles = {name: profile_from_dict(name, data) for name, data in grouped.items()}
    for profile in profiles.values():
        for inc in profile.include:
            if inc not in profiles:
                raise ValidationError(
                    f'profile {profile.name!r} includes unknown profile {inc!r}')
    # Detect include cycles early so a bad config fails at startup, not at goal time.
    for name in profiles:
        _expand(profiles, [name])
    return profiles


def _expand(profiles: Mapping[str, Profile], names: Sequence[str]) -> List[Profile]:
    """Return the profiles named by `names` plus everything they include (depth first)."""
    out: List[Profile] = []
    seen = set()

    def visit(name: str, stack: List[str]):
        if name in stack:
            raise ValidationError('profile include cycle: ' + ' -> '.join(stack + [name]))
        if name not in profiles:
            raise ValidationError(
                f'unknown profile {name!r}; available: {sorted(profiles) or "(none)"}')
        if name in seen:
            return
        seen.add(name)
        out.append(profiles[name])
        for inc in profiles[name].include:
            visit(inc, stack + [name])

    for n in names:
        visit(n, [])
    return out


def resolve(
    profiles: Mapping[str, Profile],
    names: Iterable[str],
    extra_topics: Iterable[str] = (),
    extra_regex: str = '',
) -> RecordSpec:
    """
    Resolve profile names plus goal-level additions into a RecordSpec.

    Raises ValidationError for unknown profiles, invalid topics/regex, or an empty selection.
    """
    names = dedupe(n.strip() for n in names if n and n.strip())
    expanded = _expand(profiles, names)
    spec = RecordSpec(profiles=[p.name for p in expanded])
    spec.all_topics = any(p.all_topics for p in expanded)
    spec.topics = dedupe(
        [t for p in expanded for t in p.topics] + [validate_topic(t) for t in extra_topics])
    regexes = [p.regex for p in expanded]
    if extra_regex:
        regexes.append(validate_regex(extra_regex))
    spec.regex = join_regexes(regexes)
    spec.exclude_regex = join_regexes(p.exclude_regex for p in expanded)
    spec.exclude_topics = dedupe(t for p in expanded for t in p.exclude_topics)
    spec.extra_args = [a for p in expanded for a in p.extra_args]
    if spec.is_empty():
        raise ValidationError(
            'nothing to record: no topics, regex or all_topics in the selected profiles '
            f'({names or "none"}) or the goal')
    return spec


def build_record_command(
    spec: RecordSpec,
    output_path: str,
    node_name: str,
    ros2_executable: str = 'ros2',
    global_args: Sequence[str] = (),
) -> List[str]:
    """Return the argv for ``ros2 bag record`` (Jazzy CLI syntax)."""
    cmd = [ros2_executable, 'bag', 'record', '-o', output_path,
           '--disable-keyboard-controls', '--node-name', node_name]
    cmd += list(global_args)
    cmd += spec.extra_args
    if spec.all_topics:
        cmd.append('--all-topics')
    if spec.regex:
        cmd += ['--regex', spec.regex]
    if spec.exclude_regex:
        cmd += ['--exclude-regex', spec.exclude_regex]
    if spec.exclude_topics:
        cmd += ['--exclude-topics'] + spec.exclude_topics
    if spec.topics:
        # Keep the variadic --topics list last so nothing after it is mistaken for a topic.
        cmd += ['--topics'] + spec.topics
    return cmd
