from __future__ import annotations

import sqlite3
from pathlib import Path

from app.models import Asset, DislikeRequest, ExternalTrack, ResourceTrack, utc_now_iso


class ResourceLibrary:
    """Local SQLite candidate/resource library.

    It stores verified online candidates, imported/local tracks, exposure counts
    and negative feedback without replacing the existing JSON store.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    artist TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL,
                    source_id TEXT NOT NULL DEFAULT '',
                    genre TEXT NOT NULL DEFAULT '',
                    mood TEXT NOT NULL DEFAULT '',
                    playback_url TEXT,
                    verified INTEGER NOT NULL DEFAULT 0,
                    last_seen TEXT NOT NULL,
                    exposure_count INTEGER NOT NULL DEFAULT 0,
                    ts_alpha REAL NOT NULL DEFAULT 1.0,
                    ts_beta REAL NOT NULL DEFAULT 1.0,
                    UNIQUE(source, source_id, title, artist)
                )
                """
            )
            # 迁移：为旧库补 Thompson Sampling 列。
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(tracks)")}
            if "ts_alpha" not in existing:
                conn.execute("ALTER TABLE tracks ADD COLUMN ts_alpha REAL NOT NULL DEFAULT 1.0")
            if "ts_beta" not in existing:
                conn.execute("ALTER TABLE tracks ADD COLUMN ts_beta REAL NOT NULL DEFAULT 1.0")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dislikes (
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    artist TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, title, artist, source, source_id)
                )
                """
            )

    def sync_assets(self, assets: list[Asset]) -> None:
        for asset in assets:
            self.upsert_asset(asset)

    def upsert_asset(self, asset: Asset) -> None:
        track = ResourceTrack(
            title=asset.title,
            artist=asset.artist or "",
            source="local",
            source_id=asset.asset_id,
            genre=asset.genre,
            mood=asset.mood,
            playback_url=asset.source_url,
            verified=True,
        )
        self.upsert_track(track)

    def upsert_external(self, track: ExternalTrack) -> None:
        self.upsert_track(
            ResourceTrack(
                title=track.title,
                artist=track.artist,
                source=track.source,
                source_id=track.external_id,
                genre=track.genre,
                mood=track.mood,
                playback_url=track.playback_url,
                verified=track.source in {"netease", "bilibili", "youtube"},
            )
        )

    def upsert_track(self, track: ResourceTrack) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tracks(title, artist, source, source_id, genre, mood, playback_url, verified, last_seen, exposure_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, source_id, title, artist) DO UPDATE SET
                    genre=excluded.genre,
                    mood=excluded.mood,
                    playback_url=excluded.playback_url,
                    verified=excluded.verified,
                    last_seen=excluded.last_seen
                """,
                (
                    track.title,
                    track.artist,
                    track.source,
                    track.source_id,
                    "|".join(track.genre),
                    "|".join(track.mood),
                    track.playback_url,
                    1 if track.verified else 0,
                    utc_now_iso(),
                    track.exposure_count,
                ),
            )

    def list_tracks(self, limit: int = 100) -> list[ResourceTrack]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tracks ORDER BY verified DESC, last_seen DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_track(row) for row in rows]

    def record_exposure(self, tracks: list[ExternalTrack | Asset]) -> None:
        with self._connect() as conn:
            for track in tracks:
                source = getattr(track, "source", "local")
                source_id = getattr(track, "external_id", "") or getattr(track, "asset_id", "")
                title = getattr(track, "title", "")
                artist = getattr(track, "artist", "") or ""
                conn.execute(
                    """
                    UPDATE tracks SET exposure_count=exposure_count+1, last_seen=?
                    WHERE source=? AND source_id=? AND title=? AND artist=?
                    """,
                    (utc_now_iso(), source, source_id, title, artist),
                )

    def sample_ts_scores(self, tracks: list[ExternalTrack | Asset]) -> dict[str, float]:
        """Thompson Sampling：为每个候选从 Beta(α,β) 采样一个探索分数 [0,1]。

        返回 {track_key: ts_score}，供精排在尾部候选中捞回高潜力冷门歌。
        未入库的候选用 Beta(1,1)=均匀分布兜底。
        """
        import random

        scores: dict[str, float] = {}
        with self._connect() as conn:
            for track in tracks:
                key, source, source_id, title, artist = self._identity(track)
                row = conn.execute(
                    "SELECT ts_alpha, ts_beta FROM tracks WHERE source=? AND source_id=? AND title=? AND artist=?",
                    (source, source_id, title, artist),
                ).fetchone()
                alpha = row["ts_alpha"] if row else 1.0
                beta = row["ts_beta"] if row else 1.0
                scores[key] = random.betavariate(max(alpha, 1e-6), max(beta, 1e-6))
        return scores

    def decay_exposure_ts(self, tracks: list[ExternalTrack | Asset]) -> None:
        """曝光衰减：被推荐即 ts_beta += 0.3（未被点击的先验逐步下降）。"""
        with self._connect() as conn:
            for track in tracks:
                _, source, source_id, title, artist = self._identity(track)
                conn.execute(
                    "UPDATE tracks SET ts_beta = ts_beta + 0.3 WHERE source=? AND source_id=? AND title=? AND artist=?",
                    (source, source_id, title, artist),
                )

    def update_ts_feedback(self, track: ExternalTrack | Asset, positive: bool, weight: float = 1.0) -> None:
        """在线学习反馈环（超越 SoulTuner 未完成项）：
        正反馈（听完/高分）→ ts_alpha += weight；负反馈（秒跳）→ ts_beta += weight。"""
        _, source, source_id, title, artist = self._identity(track)
        column = "ts_alpha" if positive else "ts_beta"
        with self._connect() as conn:
            conn.execute(
                f"UPDATE tracks SET {column} = {column} + ? WHERE source=? AND source_id=? AND title=? AND artist=?",
                (weight, source, source_id, title, artist),
            )

    @staticmethod
    def _identity(track: ExternalTrack | Asset) -> tuple[str, str, str, str, str]:
        source = getattr(track, "source", "local")
        source_id = getattr(track, "external_id", "") or getattr(track, "asset_id", "")
        title = getattr(track, "title", "")
        artist = getattr(track, "artist", "") or ""
        key = f"{source}|{source_id}|{title.lower()}|{artist.lower()}"
        return key, source, source_id, title, artist

    @staticmethod
    def track_key(track: ExternalTrack | Asset) -> str:
        return ResourceLibrary._identity(track)[0]

    def add_dislike(self, request: DislikeRequest) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO dislikes(user_id, title, artist, source, source_id, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.user_id,
                    request.title,
                    request.artist,
                    request.source,
                    request.source_id,
                    request.reason,
                    utc_now_iso(),
                ),
            )

    def list_dislikes(self, user_id: str) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT title, artist, source, source_id, reason FROM dislikes WHERE user_id=?",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def is_disliked(self, user_id: str, track: ExternalTrack | Asset) -> bool:
        title = getattr(track, "title", "")
        artist = getattr(track, "artist", "") or ""
        source = getattr(track, "source", "local")
        source_id = getattr(track, "external_id", "") or getattr(track, "asset_id", "")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM dislikes
                WHERE user_id=?
                  AND (
                    (source_id != '' AND source_id=?)
                    OR (lower(title)=lower(?) AND (artist='' OR lower(artist)=lower(?)))
                    OR (source != '' AND source=? AND lower(title)=lower(?))
                  )
                LIMIT 1
                """,
                (user_id, source_id, title, artist, source, title),
            ).fetchone()
        return row is not None

    @staticmethod
    def _row_to_track(row: sqlite3.Row) -> ResourceTrack:
        return ResourceTrack(
            title=row["title"],
            artist=row["artist"],
            source=row["source"],
            source_id=row["source_id"],
            genre=[item for item in row["genre"].split("|") if item],
            mood=[item for item in row["mood"].split("|") if item],
            playback_url=row["playback_url"],
            verified=bool(row["verified"]),
            last_seen=row["last_seen"],
            exposure_count=row["exposure_count"],
        )
