"""
The collector's data directory: bag folders, logs and bookkeeping.

Layout::

    <data_dir>/
        <bag_name>/                 one folder per bag (written by ros2 bag record)
            metadata.yaml
            <bag_name>_0.mcap
        .rdc/
            logs/<bag_name>.log     recorder output (also served over HTTP)
            logs/http_sidecar.log
            active/<bag_name>.json  marker while a recorder is running (crash recovery)
"""

from dataclasses import dataclass, field
from datetime import datetime
import fnmatch
import json
import os
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

from .naming import make_bag_name, validate_bag_name, ValidationError

STATE_DIR_NAME = '.rdc'


@dataclass
class BagRecord:
    """Plain description of a bag folder (converted to BagInfo.msg by the node)."""

    name: str
    path: Path
    files: List[str] = field(default_factory=list)
    file_sizes: List[int] = field(default_factory=list)
    complete: bool = False
    active: bool = False
    modified: float = 0.0

    @property
    def size_bytes(self) -> int:
        return sum(self.file_sizes)


class BagStore:
    """Filesystem operations on the data directory, restricted to direct child folders."""

    def __init__(self, data_dir):
        self.data_dir = Path(os.path.expanduser(str(data_dir))).resolve()
        self.state_dir = self.data_dir / STATE_DIR_NAME
        self.logs_dir = self.state_dir / 'logs'
        self.active_dir = self.state_dir / 'active'

    def ensure(self) -> None:
        """Create the directory layout if needed."""
        for d in (self.data_dir, self.logs_dir, self.active_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths & names
    def bag_path(self, name: str) -> Path:
        """Return the path of bag `name`, guaranteed to be a direct child of data_dir."""
        validate_bag_name(name)
        path = self.data_dir / name
        if path.parent != self.data_dir:
            raise ValidationError(f'bag {name!r} escapes the data directory')
        return path

    def log_path(self, name: str) -> Path:
        validate_bag_name(name)
        return self.logs_dir / f'{name}.log'

    def relative_url_path(self, path: Path) -> str:
        """Return `path` relative to data_dir with POSIX separators (for URLs)."""
        return Path(path).resolve().relative_to(self.data_dir).as_posix()

    def unique_bag_name(self, prefix: str, now: Optional[datetime] = None,
                        reserved: Iterable[str] = ()) -> str:
        """Return '<prefix>_<timestamp>' that does not collide with existing/reserved bags."""
        base = make_bag_name(prefix, now)
        reserved = set(reserved)
        name = base
        n = 1
        while name in reserved or (self.data_dir / name).exists() \
                or (self.logs_dir / f'{name}.log').exists():
            name = f'{base}-{n}'
            n += 1
        return validate_bag_name(name)

    # ------------------------------------------------------------------ queries
    def bag_record(self, name: str, active_names: Iterable[str] = ()) -> BagRecord:
        """Describe one bag folder. Raise FileNotFoundError if it does not exist."""
        path = self.bag_path(name)
        if not path.is_dir() or path.is_symlink():
            raise FileNotFoundError(f'no bag named {name!r} in {self.data_dir}')
        files, sizes = [], []
        latest = path.stat().st_mtime
        for root, dirs, names in os.walk(path):
            dirs.sort()
            for fname in sorted(names):
                fpath = Path(root) / fname
                try:
                    st = fpath.lstat()
                except OSError:
                    continue
                if not fpath.is_file() or fpath.is_symlink():
                    continue
                files.append(fpath.relative_to(path).as_posix())
                sizes.append(st.st_size)
                latest = max(latest, st.st_mtime)
        active = name in set(active_names)
        return BagRecord(
            name=name, path=path, files=files, file_sizes=sizes, active=active,
            complete=(path / 'metadata.yaml').is_file() and not active, modified=latest)

    def list_bags(self, pattern: str = '', active_names: Iterable[str] = ()) -> List[BagRecord]:
        """Return bag folders matching the shell-style `pattern`, newest first."""
        active_names = set(active_names)
        records = []
        if not self.data_dir.is_dir():
            return records
        for entry in self.data_dir.iterdir():
            if entry.name.startswith('.') or not entry.is_dir() or entry.is_symlink():
                continue
            if pattern and not fnmatch.fnmatchcase(entry.name, pattern):
                continue
            try:
                records.append(self.bag_record(entry.name, active_names))
            except (ValidationError, FileNotFoundError, OSError):
                continue  # not a folder we created (odd name) or vanished meanwhile
        records.sort(key=lambda r: r.modified, reverse=True)
        return records

    def disk_usage(self) -> Tuple[int, int]:
        """Return (free_bytes, total_bytes) of the filesystem holding data_dir."""
        usage = shutil.disk_usage(self.data_dir if self.data_dir.exists() else Path('/'))
        return usage.free, usage.total

    # ------------------------------------------------------------------ mutations
    def delete(self, name: str, active_names: Iterable[str] = ()) -> int:
        """
        Delete bag `name` (and its log). Return freed bytes.

        Raise ValidationError for bad names / active bags, FileNotFoundError if missing.
        """
        if name in set(active_names):
            raise ValidationError(f'bag {name!r} is being recorded; stop it before deleting')
        record = self.bag_record(name)
        shutil.rmtree(record.path)
        log = self.logs_dir / f'{name}.log'
        if log.exists():
            log.unlink()
        return record.size_bytes

    # ------------------------------------------------------------------ crash recovery
    def mark_active(self, name: str, info: Dict) -> None:
        """Persist a marker describing a running recording."""
        self.active_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.active_dir / f'.{name}.json.tmp'
        tmp.write_text(json.dumps(info))
        tmp.replace(self.active_dir / f'{name}.json')

    def clear_active(self, name: str) -> None:
        """Remove the marker for `name` (no error if missing)."""
        try:
            (self.active_dir / f'{name}.json').unlink()
        except FileNotFoundError:
            pass

    def active_markers(self) -> Dict[str, Dict]:
        """Return markers left on disk, keyed by bag name (e.g. after a crash)."""
        markers = {}
        if not self.active_dir.is_dir():
            return markers
        for marker in self.active_dir.glob('*.json'):
            try:
                markers[marker.stem] = json.loads(marker.read_text())
            except (OSError, ValueError):
                markers[marker.stem] = {}
        return markers
