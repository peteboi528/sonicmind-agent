from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.config import settings


class LibraryBaseline(BaseModel):
    store_root: str
    resource_library: str
    media_root: str
    asset_count: int
    resource_track_count: int
    local_track_count: int
    resource_source_counts: dict[str, int]
    media_file_count: int
    asset_digest: str
    resource_digest: str
    media_digest: str
    overall_digest: str


def build_library_baseline(
    *,
    store_root: str | Path | None = None,
    resource_library: str | Path | None = None,
    media_root: str | Path | None = None,
) -> LibraryBaseline:
    store = Path(store_root or settings.store_root).resolve()
    resource = Path(resource_library or settings.resource_library_path).resolve()
    media = Path(media_root or settings.media_root).resolve()
    asset_files = sorted((store / "assets").glob("*.json")) if (store / "assets").exists() else []
    media_files = sorted(path for path in media.rglob("*") if path.is_file()) if media.exists() else []
    asset_digest = _file_set_digest(asset_files, store)
    media_digest = _file_set_digest(media_files, media)
    resource_count, resource_digest, source_counts = _resource_content_digest(resource)
    overall = _hash_json({
        "asset_count": len(asset_files),
        "resource_track_count": resource_count,
        "local_track_count": source_counts.get("local", 0),
        "resource_source_counts": source_counts,
        "media_file_count": len(media_files),
        "asset_digest": asset_digest,
        "resource_digest": resource_digest,
        "media_digest": media_digest,
    })
    return LibraryBaseline(
        store_root=str(store),
        resource_library=str(resource),
        media_root=str(media),
        asset_count=len(asset_files),
        resource_track_count=resource_count,
        local_track_count=source_counts.get("local", 0),
        resource_source_counts=source_counts,
        media_file_count=len(media_files),
        asset_digest=asset_digest,
        resource_digest=resource_digest,
        media_digest=media_digest,
        overall_digest=overall,
    )


def compare_library_baselines(before: LibraryBaseline, after: LibraryBaseline) -> dict[str, Any]:
    fields = (
        "asset_count", "resource_track_count", "local_track_count", "resource_source_counts", "media_file_count",
        "asset_digest", "resource_digest", "media_digest", "overall_digest",
    )
    changed = {
        field: {"before": getattr(before, field), "after": getattr(after, field)}
        for field in fields if getattr(before, field) != getattr(after, field)
    }
    return {"unchanged": not changed, "changed": changed}


def _file_set_digest(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _resource_content_digest(path: Path) -> tuple[int, str, dict[str, int]]:
    if not path.exists():
        return 0, hashlib.sha256(b"").hexdigest(), {}
    uri = f"file:{path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracks'"
        ).fetchone()
        if not exists:
            return 0, hashlib.sha256(b"").hexdigest(), {}
        rows = connection.execute(
            """SELECT title,artist,source,source_id,genre,mood,
                      COALESCE(playback_url,''),verified
               FROM tracks
               ORDER BY source,source_id,title,artist"""
        ).fetchall()
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row[2])
        source_counts[source] = source_counts.get(source, 0) + 1
    return len(rows), _hash_json([list(row) for row in rows]), dict(sorted(source_counts.items()))


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
