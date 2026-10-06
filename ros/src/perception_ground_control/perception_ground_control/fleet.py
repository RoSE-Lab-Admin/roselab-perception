"""
Ground control fleet configuration: which remote collectors exist and how they are grouped.

Parameters (see config/ground_control.yaml)::

    remotes:
      mastcam_front:              # remote name used in targets
        node: /mastcam_front      # collector node name (default "/<remote name>")
        bag_name: mastcam_bags    # bag prefix (default: the collector's default_bag_name)
        profiles: [mastcam]       # topic profiles (default: the collector's default_profiles)
        topics: [/extra]          # extra topics (optional)
        regex: ""                 # extra regex (optional)
        duration: 0.0             # default seconds; 0 = until stopped
        download: true            # download bags when they finish
        delete_after_download: true
    groups:
      trial: [mastcam_front, ground_station]
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping

REMOTE_KEYS = ('node', 'bag_name', 'profiles', 'topics', 'regex', 'duration', 'download',
               'delete_after_download')


class FleetConfigError(ValueError):
    """Invalid ground control configuration or target list."""


@dataclass
class RemoteConfig:
    """How ground control uses one remote_data_collector."""

    name: str
    node: str
    bag_name: str = ''
    profiles: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    regex: str = ''
    duration: float = 0.0
    download: bool = True
    delete_after_download: bool = False

    def effective_duration(self, requested: float) -> float:
        """Resolve a StartCollection duration (0 = default, <0 = until stopped)."""
        if requested > 0:
            return float(requested)
        if requested < 0:
            return 0.0
        return float(self.duration)

    def bag_prefix(self, label: str) -> str:
        """Return the bag name prefix for a start request with optional label."""
        label = label.strip()
        if not label:
            return self.bag_name
        return f'{label}_{self.bag_name}' if self.bag_name else label


def _list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(',') if v.strip()]
    return [str(v) for v in value if str(v).strip()]


def _group_flat(flat: Mapping[str, Any], what: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for dotted, value in flat.items():
        if '.' not in dotted:
            raise FleetConfigError(f'parameter "{what}.{dotted}" must be {what}.<name>.<key>')
        name, key = dotted.split('.', 1)
        grouped.setdefault(name, {})[key] = value
    return grouped


def remotes_from_flat_params(flat: Mapping[str, Any]) -> Dict[str, RemoteConfig]:
    """Build RemoteConfigs from ``{'<remote>.<key>': value}`` (parameters under 'remotes.')."""
    remotes = {}
    for name, data in _group_flat(flat, 'remotes').items():
        unknown = set(data) - set(REMOTE_KEYS)
        if unknown:
            raise FleetConfigError(
                f'remote {name!r}: unknown key(s) {sorted(unknown)}; allowed: {list(REMOTE_KEYS)}')
        node = str(data.get('node') or f'/{name}').strip().rstrip('/')
        remotes[name] = RemoteConfig(
            name=name,
            node=node if node.startswith('/') else '/' + node,
            bag_name=str(data.get('bag_name', '') or ''),
            profiles=_list(data.get('profiles')),
            topics=_list(data.get('topics')),
            regex=str(data.get('regex', '') or ''),
            duration=float(data.get('duration', 0.0) or 0.0),
            download=bool(data.get('download', True)),
            delete_after_download=bool(data.get('delete_after_download', False)),
        )
    return remotes


def groups_from_params(flat: Mapping[str, Any],
                       remotes: Mapping[str, RemoteConfig]) -> Dict[str, List[str]]:
    """Build groups from ``{'<group>': [remote, ...]}`` (parameters under 'groups.')."""
    groups = {}
    for name, members in flat.items():
        if '.' in name:
            raise FleetConfigError(f'group name may not contain ".": {name!r}')
        if name in remotes:
            raise FleetConfigError(f'group {name!r} has the same name as a remote')
        members = _list(members)
        unknown = [m for m in members if m not in remotes]
        if unknown:
            raise FleetConfigError(f'group {name!r} references unknown remote(s) {unknown}')
        groups[name] = members
    return groups


def resolve_targets(targets: Iterable[str], remotes: Mapping[str, RemoteConfig],
                    groups: Mapping[str, List[str]], empty_means: List[str]) -> List[str]:
    """
    Expand remote/group names into an ordered, de-duplicated list of remote names.

    `empty_means` is returned when no targets are given. The special target "all" means
    every configured remote (unless a group named "all" exists).
    """
    targets = [t.strip() for t in targets if t and t.strip()]
    if not targets:
        return list(empty_means)
    out: List[str] = []
    unknown = []
    for t in targets:
        if t in remotes:
            members = [t]
        elif t in groups:
            members = groups[t]
        elif t == 'all':
            members = list(remotes)
        else:
            unknown.append(t)
            continue
        out += [m for m in members if m not in out]
    if unknown:
        raise FleetConfigError(
            f'unknown target(s) {unknown}; remotes: {sorted(remotes)}, groups: {sorted(groups)}')
    return out
