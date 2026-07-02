from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.models import Asset, DislikeRequest, ExternalTrack, ResourceTrack, utc_now_iso
from app.recommend.hygiene import is_structural_reject


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
                    embedding TEXT NOT NULL DEFAULT '',
                    embed_dirty INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(source, source_id, title, artist)
                )
                """
            )
            # 迁移：为旧库补 Thompson Sampling 列 + embedding 预存列。
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(tracks)")}
            if "ts_alpha" not in existing:
                conn.execute("ALTER TABLE tracks ADD COLUMN ts_alpha REAL NOT NULL DEFAULT 1.0")
            if "ts_beta" not in existing:
                conn.execute("ALTER TABLE tracks ADD COLUMN ts_beta REAL NOT NULL DEFAULT 1.0")
            if "embedding" not in existing:
                # embedding：候选文本的归一化向量（JSON 数组），预存避免每次语义检索对全池现算。
                conn.execute("ALTER TABLE tracks ADD COLUMN embedding TEXT NOT NULL DEFAULT ''")
            if "embed_dirty" not in existing:
                # embed_dirty：文本变了需重算；1=待算（含迁移来的旧行，首次检索时补算并落库）。
                conn.execute("ALTER TABLE tracks ADD COLUMN embed_dirty INTEGER NOT NULL DEFAULT 1")
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
        # 结构性脏数据（教程/合集/串烧/功能音乐/mix）不入池——这是池子的唯一入口，9 个调用点
        # （web_music_search/search.py×4/handlers 等）共享这一道闸门，无需各自过滤。零 embedding
        # 成本（语义层判定仍只在推荐出口 classify_candidate 跑）。规则与推荐出口同源（hygiene）。
        if is_structural_reject(track):
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
                    last_seen=excluded.last_seen,
                    -- genre/mood 进了 embedding 文本，变了就需重算，标 dirty。
                    embed_dirty = CASE
                        WHEN excluded.genre IS NOT tracks.genre OR excluded.mood IS NOT tracks.mood
                        THEN 1 ELSE tracks.embed_dirty END
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

        性能关键：候选向量**预存**在 tracks.embedding 列。dirty 行（新写入或改了
        genre/mood）首次检索时补算并落库，之后命中预存向量——避免每次对全池现算
        embedding（1068 首冷启动曾耗时 21s，是 web_music_search 超时的真根因）。

        冷启动保护：若池里有大量 dirty 行且未预热，返回 [] 让调用方走零开销的词法
        兜底，而不是在请求路径里同步算几百个向量（会撞超时墙）。后台 warm_embeddings
        把向量算好后语义召回才启用。
        """
        query = (query or "").strip()
        if not query:
            return []
        from app.retrieval import embeddings

        if not embeddings.embeddings_available():
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, artist, genre, mood, source, source_id, playback_url,
                       verified, last_seen, exposure_count, embedding, embed_dirty
                FROM tracks WHERE verified=1
                ORDER BY last_seen DESC LIMIT ?
                """,
                (pool_size,),
            ).fetchall()
        if not rows:
            return []

        # 冷启动保护：未预热时别在请求路径里同步算大批向量，让位词法兜底。
        # 仅个别 dirty（增量写入）才补算——那是小批、可接受。
        dirty = [r for r in rows if r["embed_dirty"] or not r["embedding"]]
        if len(dirty) > 32:
            return []
        new_vectors = self._compute_and_persist_embeddings(dirty) if dirty else {}

        # query 向量走 encode LRU（同 query 不重算）。
        query_vec = embeddings.encode([query])
        query_vec = query_vec[0] if query_vec else None
        if query_vec is None:
            return []

        scored: list[tuple[float, ResourceTrack]] = []
        for r in rows:
            vec = new_vectors.get(r["id"])
            if vec is None:
                vec = self._decode_embedding(r["embedding"])
            if not vec:
                continue
            score = (embeddings.cosine_normalized(query_vec, vec) + 1.0) / 2.0
            if score >= min_score:
                scored.append((score, self._row_to_track(r)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [track for _, track in scored[:limit]]

    def warm_embeddings(self, batch: int = 200) -> int:
        """后台预热：批量补算所有 dirty 行的 embedding 并落库。

        在请求路径外（启动后台线程）跑，把冷启动的几十秒开销挪出用户等待。
        返回本次算好的行数。embedding 不可用时无操作。
        """
        from app.retrieval import embeddings

        if not embeddings.embeddings_available():
            return 0
        # 取全部 dirty 行的 id（不截断 batch），下面再分块算。batch 是单次 encode 的块大小。
        with self._connect() as conn:
            ids = [r["id"] for r in conn.execute(
                "SELECT id FROM tracks WHERE verified=1 AND embed_dirty=1"
            ).fetchall()]
        total = 0
        # 分块算（每块 batch 行），单块 encode 调用 + 批量落库，控制峰值内存。
        for start in range(0, len(ids), batch):
            chunk = ids[start:start + batch]
            if not chunk:
                continue
            with self._connect() as conn:
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    "SELECT id, title, artist, genre, mood, source, source_id, playback_url, "
                    "verified, last_seen, exposure_count "
                    f"FROM tracks WHERE id IN ({placeholders})",
                    chunk,
                ).fetchall()
            texts = [_resource_track_text(self._row_to_track(r)) for r in rows]
            vectors = embeddings.encode(texts)
            if vectors is None:
                break
            with self._connect() as conn:
                for row, vec in zip(rows, vectors, strict=False):
                    if not vec:
                        continue
                    conn.execute(
                        "UPDATE tracks SET embedding=?, embed_dirty=0 WHERE id=?",
                        (json.dumps(vec, separators=(",", ":")), row["id"]),
                    )
            total += len(chunk)
        return total

    def _compute_and_persist_embeddings(self, rows: list[sqlite3.Row]) -> dict[int, list[float]]:
        """批量算 dirty 行的 embedding，写回 SQLite，返回 {id: vector}。"""
        if not rows:
            return {}
        from app.retrieval import embeddings

        texts = [_resource_track_text(self._row_to_track(r)) for r in rows]
        vectors = embeddings.encode(texts)
        if vectors is None:
            return {}  # 编码失败（如模型临时不可用），这些行下次仍 dirty，不写入空值
        result: dict[int, list[float]] = {}
        with self._connect() as conn:
            for row, vec in zip(rows, vectors, strict=False):
                if not vec:
                    continue
                conn.execute(
                    "UPDATE tracks SET embedding=?, embed_dirty=0 WHERE id=?",
                    (json.dumps(vec, separators=(",", ":")), row["id"]),
                )
                result[row["id"]] = list(vec)
        return result

    @staticmethod
    def _decode_embedding(raw: str) -> list[float] | None:
        if not raw:
            return None
        try:
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, list) and decoded else None
        except (ValueError, TypeError):
            return None

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
