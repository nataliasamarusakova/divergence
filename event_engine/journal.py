"""Crash-aware rotation helpers for append-only JSONL journals.

Only standard-library modules are used so the workflow can rotate journals before
starting the market-data engine, and the engine can rotate again during a run.
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

# Keep margin under GitHub's 100 MiB hard per-file limit and 50 MiB warning.
DEFAULT_ACTION_JOURNAL_ROTATE_BYTES = 80 * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024
log = logging.getLogger(__name__)


def _sha256_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(_CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _sha256_file(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _sha256_gzip(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as stream:
        while True:
            chunk = stream.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Some operating systems/filesystems do not support directory fsync.
        pass


def _atomic_empty_file(path: Path) -> None:
    """Atomically replace path with a durable empty file."""
    empty_tmp = path.with_name(path.name + ".empty.tmp")
    try:
        with empty_tmp.open("wb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(empty_tmp, path)
        _fsync_directory(path.parent)
    finally:
        try:
            empty_tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _latest_matching_archive(
    archive_dir: Path,
    source_size: int,
    source_hash: str,
) -> Path | None:
    """Detect the crash window where a verified archive exists but source wasn't cleared."""
    archives = sorted(archive_dir.glob("actions-*.jsonl.gz"))
    if not archives:
        return None
    latest = archives[-1]
    try:
        archive_hash, archive_size = _sha256_gzip(latest)
    except (OSError, EOFError, gzip.BadGzipFile):
        return None
    if archive_size == source_size and archive_hash == source_hash:
        return latest
    return None


def rotate_jsonl_gzip(
    path: Path,
    *,
    threshold_bytes: int = DEFAULT_ACTION_JOURNAL_ROTATE_BYTES,
    incoming_bytes: int = 0,
) -> Path | None:
    """Rotate *path* to a verified gzip archive before it exceeds the limit.

    The original journal is untouched until a complete archive has been written
    and verified. The final archive is installed atomically, then the active file
    is atomically replaced with an empty file. If a process stops in the narrow
    interval between those operations, readers can detect the exact duplicate;
    the next rotation also reuses the matching archive instead of duplicating it.

    Returns the archive path when a rotation/recovery occurred, otherwise None.
    Raises OSError/ValueError if safe rotation cannot be completed; callers must
    not silently append past the configured maximum in that case.
    """
    path = Path(path)
    threshold_bytes = int(threshold_bytes)
    incoming_bytes = max(0, int(incoming_bytes))
    if threshold_bytes <= 0:
        raise ValueError("threshold_bytes must be positive")
    if incoming_bytes > threshold_bytes:
        raise ValueError("single journal record exceeds rotation threshold")
    if not path.exists():
        return None

    source_size = path.stat().st_size
    if source_size < threshold_bytes and source_size + incoming_bytes <= threshold_bytes:
        return None
    if source_size == 0:
        return None

    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    source_hash, confirmed_size = _sha256_file(path)
    if confirmed_size != source_size:
        raise OSError("journal changed while its size/hash were being checked")

    # Recover safely if the process previously installed the archive but stopped
    # before replacing the active journal with an empty file.
    existing = _latest_matching_archive(archive_dir, source_size, source_hash)
    if existing is not None:
        _atomic_empty_file(path)
        return existing

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = archive_dir / f"actions-{stamp}.jsonl.gz"
    suffix = 1
    while archive.exists() or archive.with_suffix(archive.suffix + ".tmp").exists():
        archive = archive_dir / f"actions-{stamp}-{suffix}.jsonl.gz"
        suffix += 1
    tmp = archive.with_suffix(archive.suffix + ".tmp")

    # Compute a hash as we compress. gzip mtime=0 keeps artifacts reproducible.
    digest = hashlib.sha256()
    copied_size = 0
    try:
        with path.open("rb") as source, tmp.open("wb") as raw_target:
            with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=raw_target, mtime=0) as target:
                while True:
                    chunk = source.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    target.write(chunk)
                    digest.update(chunk)
                    copied_size += len(chunk)
            raw_target.flush()
            os.fsync(raw_target.fileno())

        if copied_size != source_size or digest.hexdigest() != source_hash:
            raise OSError("journal changed during compression; active file preserved")

        archive_hash, archive_size = _sha256_gzip(tmp)
        if archive_size != source_size or archive_hash != source_hash:
            raise OSError("gzip archive verification failed; active file preserved")

        os.replace(tmp, archive)
        _fsync_directory(archive_dir)
        _atomic_empty_file(path)
        return archive
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _active_is_latest_archive(path: Path, archive_dir: Path) -> bool:
    """Return True only for the exact duplicate caused by an interrupted rotation."""
    archives = sorted(archive_dir.glob("actions-*.jsonl.gz"))
    if not archives or not path.exists():
        return False
    try:
        active_size = path.stat().st_size
        if active_size == 0:
            return False
        archive_hash, archive_size = _sha256_gzip(archives[-1])
        if archive_size != active_size:
            return False
        active_hash, verified_size = _sha256_file(path)
        return verified_size == active_size and active_hash == archive_hash
    except (OSError, EOFError, gzip.BadGzipFile):
        return False


def iter_jsonl_dicts(path: Path):
    """Yield JSON object records from JSONL; report malformed/truncated lines."""
    import json

    malformed = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    malformed += 1
                    continue
                if isinstance(value, dict):
                    yield value
                else:
                    malformed += 1
    except OSError as exc:
        log.error("[ACTION_JOURNAL] Could not read active JSONL %s: %s", path, exc)
        return
    if malformed:
        log.warning("[ACTION_JOURNAL] Skipped %d malformed/non-object row(s) in %s", malformed, path)


def iter_gzip_jsonl_dicts(path: Path):
    """Yield JSON object records from a compressed JSONL archive, reporting damage."""
    import json

    malformed = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    malformed += 1
                    continue
                if isinstance(value, dict):
                    yield value
                else:
                    malformed += 1
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        log.error("[ACTION_JOURNAL] Could not fully read compressed archive %s: %s", path, exc)
        return
    if malformed:
        log.warning("[ACTION_JOURNAL] Skipped %d malformed/non-object row(s) in %s", malformed, path)
