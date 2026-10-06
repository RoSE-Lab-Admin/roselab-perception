"""
Download a bag folder from a collector's HTTP sidecar (pure Python, no wget needed).

Files are written into ``<dest>/<bag>.partial/`` and the folder is renamed to ``<dest>/<bag>/``
only after every file has been downloaded and its size verified, so a half-finished download
can never be mistaken for a complete bag.
"""

from pathlib import Path, PurePosixPath
import shutil
import time
from typing import Callable, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
import urllib.request

from .naming import validate_bag_name

CHUNK = 1 << 20  # 1 MiB

ProgressCallback = Callable[[int, int], None]  # (bytes_done, bytes_total)


class DownloadError(RuntimeError):
    """Raised when a bag cannot be downloaded completely."""


def _safe_relative(path: str) -> PurePosixPath:
    rel = PurePosixPath(path)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise DownloadError(f'refusing unsafe file path from server: {path!r}')
    return rel


def _fetch(url: str, dest: Path, expected_size: Optional[int], timeout: float,
           on_bytes: Callable[[int], None]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(dest, 'wb') as out:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            on_bytes(len(chunk))
    size = dest.stat().st_size
    if expected_size is not None and size != expected_size:
        raise DownloadError(f'{url}: got {size} bytes, expected {expected_size}')


def download_bag(
    download_url: str,
    files: Sequence[str],
    dest_dir,
    bag_name: str,
    file_sizes: Optional[Sequence[int]] = None,
    progress: Optional[ProgressCallback] = None,
    timeout: float = 30.0,
    retries: int = 3,
    overwrite: bool = False,
) -> Path:
    """
    Download every file of a bag into ``dest_dir/bag_name`` and return that path.

    `download_url` is the bag folder URL (BagInfo.download_url, ending in '/').
    If the destination already exists with identical file sizes it is reused (idempotent);
    if it differs, DownloadError is raised unless `overwrite` is set.
    """
    validate_bag_name(bag_name)
    if not download_url:
        raise DownloadError('no download URL (is the collector HTTP sidecar enabled?)')
    if not files:
        raise DownloadError(f'bag {bag_name!r} has no files to download')
    if not download_url.endswith('/'):
        download_url += '/'
    sizes = list(file_sizes) if file_sizes else [None] * len(files)
    if len(sizes) != len(files):
        raise DownloadError('files and file_sizes have different lengths')
    rels = [_safe_relative(f) for f in files]

    dest_dir = Path(dest_dir).expanduser()
    final = dest_dir / bag_name
    partial = dest_dir / f'{bag_name}.partial'

    if final.exists():
        if all(s is not None for s in sizes) and all(
                (final / rel).is_file() and (final / rel).stat().st_size == s
                for rel, s in zip(rels, sizes)):
            return final
        if not overwrite:
            raise DownloadError(f'{final} already exists and differs from the remote bag')
        shutil.rmtree(final)

    total = sum(s for s in sizes if s is not None)
    done = 0

    def on_bytes(n: int) -> None:
        nonlocal done
        done += n
        if progress:
            progress(done, total)

    partial.mkdir(parents=True, exist_ok=True)
    for rel, size in zip(rels, sizes):
        target = partial / rel
        if size is not None and target.is_file() and target.stat().st_size == size:
            on_bytes(size)  # already fetched by a previous, interrupted attempt
            continue
        url = download_url + quote(rel.as_posix())
        for attempt in range(1, retries + 1):
            before = done
            try:
                _fetch(url, target, size, timeout, on_bytes)
                break
            except (HTTPError, URLError, OSError, DownloadError) as exc:
                done = before
                if attempt == retries:
                    raise DownloadError(
                        f'failed to download {url} after {retries} attempts: {exc}') from exc
                time.sleep(min(2.0 * attempt, 10.0))
    partial.rename(final)
    return final


def format_bytes(n: float) -> str:
    """Return a human readable size, e.g. '1.5 GiB'."""
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(n) < 1024 or unit == 'TiB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.1f} {unit}'
        n /= 1024.0
    return f'{n:.1f} TiB'
