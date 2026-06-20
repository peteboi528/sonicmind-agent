from __future__ import annotations

import sqlite3
from pathlib import Path

from app.models import Asset, DislikeRequest, ExternalTrack, ResourceTrack, utc_now_iso


class ResourceLibrary:
    """Local SQLite candidate/resource library.

    It stores verified online candidates, imported/local tracks, exposure counts
    and negative feedback without replacing the existing JSON store.
    """

    def __init__(self, path: str | Path, *, max_tracks: int = 5000) -> None:
        self.path = Path(path)
        self.max_tracks = max_tracks
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._writes_since_prune = 0

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
        # 候选池只收真实可追溯来源。fallback/mock/llm 是联网不足时的降级假候选
        # （播不了、易污染推荐），坚决不入库——否则池子被假歌撑大，拖慢所有
        # 拉池子的操作（semantic_search/similar_artists），且污染后续推荐。
        source = (track.source or "").lower()
        if "fallback" in source or source in {"mock", "llm"}:
            return
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
        # 周期性裁剪（每 50 次写检查一次，摊销开销），把无界增长封顶。
        self._writes_since_prune += 1
        if self._writes_since_prune >= 50:
            self._writes_since_prune = 0
            self.prune()

    def prune(self) -> int:
        """池子超上限时淘汰最旧的、未被曝光过的外部候选，返回删除行数。

        保护 source='local'（用户真实导入库）和 exposure_count>0（曾被推荐过、
        有价值）的行；只淘汰"搜来但从没用上"的外部候选，按 last_seen 最旧优先。
        """
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
            if total <= self.max_tracks:
                return 0
            overflow = total - self.max_tracks
            cursor = conn.execute(
                """
                DELETE FROM tracks WHERE id IN (
                    SELECT id FROM tracks
                    WHERE source != 'local' AND exposure_count = 0
                    ORDER BY last_seen ASC
                    LIMIT ?
                )
                """,
                (overflow,),
            )
            return cursor.rowcount

    def purge_fallback_sources(self) -> int:
        """一次性清理：删除历史遗留的 fallback/mock/llm 假候选。

        老版本 upsert_external 无差别入库，污染了池子（netease-fallback 等）。
        新版已在入库口拦截，本方法清掉存量。返回删除行数。
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM tracks
                WHERE source LIKE '%fallback%' OR source IN ('mock', 'llm')
                """
            )
            return cursor.rowcount

    def purge_orphan_local(self, live_asset_ids: set[str]) -> int:
        """一次性清理：删除 source='local' 但对应 asset 已不存在的僵尸行。

        清空前端库（删 assets/*.json）时 SQLite 不会同步，留下指向已删 asset 的
        local 行。传入当前存活的 asset_id 集合，删掉不在其中的 local 行。返回删除行数。
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT id, source_id FROM tracks WHERE source='local'").fetchall()
            stale_ids = [row["id"] for row in rows if row["source_id"] not in live_asset_ids]
            if not stale_ids:
                return 0
            conn.executemany("DELETE FROM tracks WHERE id=?", [(i,) for i in stale_ids])
            return len(stale_ids)

    def list_tracks(self, limit: int = 100, *, verified_only: bool = False) -> list[ResourceTrack]:
        # verified_only 把过滤下推到 SQL，避免上层"拉 3000 行再丢掉未验证的"——
        # 池子大时这能少物化大量 ResourceTrack 对象（喂超时的开销之一）。
        where = "WHERE verified=1" if verified_only else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM tracks {where} ORDER BY verified DESC, last_seen DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_track(row) for row in rows]

    def semantic_search(self, query: str, limit: int = 5, pool_size: int = 300, min_score: float = 0.55) -> list[ResourceTrack]:
        """Dense fallback recall over verified resource-library tracks.

        This is intentionally opportunistic: if embeddings are unavailable or
        encoding fails, return [] and let callers keep their existing flow.
        """
        query = (query or "").strip()
        if not query:
            return []
        from app.retrieval import embeddings

        if not embeddings.embeddings_available():
            return []
        candidates = [track for track in self.list_tracks(pool_size) if track.verified]
        if not candidates:
            return []
        texts = [_resource_track_text(track) for track in candidates]
        scores = embeddings.semantic_scores(query, texts)
        if scores is None:
            return []
        ranked = [
            (score, track)
            for score, track in zip(scores, candidates, strict=False)
            if score >= min_score
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [track for _, track in ranked[:limit]]

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


def _resource_track_text(track: ResourceTrack) -> str:
    return " ".join([
        track.title or "",
        track.artist or "",
        " ".join(track.genre or []),
        " ".join(track.mood or []),
    ]).strip()
