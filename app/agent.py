from __future__ import annotations

import hashlib
import json
import random
import urllib.parse
from typing import Any

from app.config import settings
from app.llm.client import build_llm
from app.llm.protocol import LLMProvider
from app.llm.structured import extract_json_dict, extract_json_list
from app.media.pipeline import MediaPipeline, netease_song_id
from app.memory import MemoryManager
from app.prompts import (
    AUTO_PLAYLIST_TEMPLATE,
    GENERATE_PLAYLIST_TEMPLATE,
    IDENTIFY_FROM_URL_TEMPLATE,
    LLM_SEARCH_TEMPLATE,
)
from app.models import (
    AgentAnswer,
    Asset,
    AssetStatus,
    DailyRecommendation,
    EnrichResponse,
    ExternalTrack,
    FeedbackRequest,
    MemoryUpdateRequest,
    Playlist,
    RagEvidence,
    RecommendedTrack,
    Segment,
    SearchResponse,
    SimilarAssetResult,
    SimilarSegmentResult,
    TasteProfile,
    UserMemory,
    utc_now_iso,
)
from app.react_loop import ReActLoop
from app.recommend.daily import DailyRecommender
from app.recommend.engine import RecommendEngine
from app.retrieval.vector_store import HybridRetriever
from app.similarity import AssetSimilarity
from app.sources.mock_source import MockSource
from app.sources.protocol import ExternalSource
from app.storage import JsonStore


class AudioVisualAgent:
    def __init__(self, store: JsonStore | None = None) -> None:
        self.store = store or JsonStore()
        self.media = MediaPipeline(self.store)
        self.memory = MemoryManager(self.store)
        self.similarity = AssetSimilarity(self.store)
        self.llm: LLMProvider = build_llm()
        self.source: ExternalSource = MockSource()
        self.engine = RecommendEngine()
        self.daily = DailyRecommender(self.engine, self.source, self.llm)
        self.react = ReActLoop(self)

    def ingest_video(self, url: str, force_refresh: bool = False) -> Asset:
        return self.media.ingest_video(url, force_refresh=force_refresh)

    def enrich_asset(self, asset_id: str, use_network: bool = False) -> EnrichResponse:
        asset = self.store.read_model("assets", asset_id, Asset)
        if asset is None:
            raise ValueError(f"Unknown asset_id: {asset_id}")

        if not use_network and not settings.enable_online_enrich:
            return EnrichResponse(
                asset=asset,
                enriched=False,
                mode="offline",
                note="Offline-first mode keeps enrich optional. Enable network explicitly to fetch title metadata.",
            )

        before = asset.model_dump(mode="json")
        # 网易云：API 直接给出精确 title + artist，只用 LLM 补 genre/mood
        if "163.com" in asset.source_url or "163cn.tv" in asset.source_url:
            song_id = _netease_song_id(asset.source_url)
            if song_id and self._enrich_from_netease(asset, song_id):
                self.store.write_model("assets", asset.asset_id, asset)
                after = asset.model_dump(mode="json")
                enriched = before != after
                return EnrichResponse(
                    asset=asset, enriched=enriched, mode="online",
                    note="Metadata enrichment completed." if enriched else "No new metadata was identified.",
                )
        video_title = self._fetch_video_title(asset.source_url)
        self._apply_title_artist_hint(asset, video_title)
        self._identify_from_url(asset, video_title)
        after = asset.model_dump(mode="json")
        enriched = before != after
        return EnrichResponse(
            asset=asset,
            enriched=enriched,
            mode="online",
            note="Metadata enrichment completed." if enriched else "No new metadata was identified.",
        )

    def _fetch_video_title(self, url: str) -> str | None:
        import json as _json
        import re as _re
        import urllib.parse as _parse
        import urllib.request as _req

        _headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

        # YouTube oEmbed
        if "youtube.com" in url or "youtu.be" in url:
            try:
                oembed = f"https://www.youtube.com/oembed?url={_parse.quote(url, safe='')}&format=json"
                with _req.urlopen(_req.Request(oembed, headers=_headers), timeout=10) as r:
                    return _json.loads(r.read().decode()).get("title")
            except Exception:
                pass

        # 网易云音乐 —— 优先调官方 API，退回 HTML <title>
        if "163.com" in url or "163cn.tv" in url:
            song_id = _netease_song_id(url)
            if song_id:
                # 方式 A：官方 API（无需登录，返回结构化 JSON）
                try:
                    api = f"https://music.163.com/api/song/detail/?id={song_id}&ids=[{song_id}]"
                    req = _req.Request(api, headers={**_headers, "Referer": "https://music.163.com/"})
                    with _req.urlopen(req, timeout=10) as r:
                        data = _json.loads(r.read().decode())
                    songs = data.get("songs") or []
                    if songs:
                        name = songs[0].get("name", "")
                        artists = "/".join(a.get("name", "") for a in songs[0].get("artists", []))
                        return f"{name} - {artists}" if artists else name
                except Exception:
                    pass
            # 方式 B：抓 HTML <title>
            try:
                page_url = f"https://music.163.com/song?id={song_id}" if song_id else url
                with _req.urlopen(_req.Request(page_url, headers=_headers), timeout=10) as r:
                    html = r.read().decode("utf-8", errors="ignore")[:20000]
                m = _re.search(r"<title[^>]*>(.+?)</title>", html, _re.IGNORECASE | _re.DOTALL)
                if m:
                    title = m.group(1).strip()
                    for suffix in [" - 单曲 - 网易云音乐", " - 网易云音乐"]:
                        title = title.replace(suffix, "")
                    return title.strip()
            except Exception:
                pass

        # B 站 HTML <title>
        if "bilibili.com" in url:
            try:
                with _req.urlopen(_req.Request(url, headers=_headers), timeout=10) as r:
                    html = r.read().decode("utf-8", errors="ignore")[:20000]
                m = _re.search(r"<title[^>]*>(.+?)</title>", html, _re.IGNORECASE | _re.DOTALL)
                if m:
                    title = m.group(1).strip()
                    for suffix in ["_哔哩哔哩_bilibili", " - 哔哩哔哩"]:
                        title = title.replace(suffix, "")
                    return title.strip()
            except Exception:
                pass

        # 通用兜底：yt-dlp（不强依赖 Chrome cookies）
        try:
            import subprocess
            result = subprocess.run(
                ["yt-dlp", "--get-title", "--no-download", "--no-warnings", url],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _enrich_from_netease(self, asset: Asset, song_id: str) -> bool:
        """从网易云 API 直接获取 title/artist，不走 LLM 猜测，再用 LLM 补 genre/mood。"""
        import json as _json
        import urllib.request as _req
        _headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Referer": "https://music.163.com/",
        }
        try:
            api = f"https://music.163.com/api/song/detail/?id={song_id}&ids=[{song_id}]"
            with _req.urlopen(_req.Request(api, headers=_headers), timeout=10) as r:
                data = _json.loads(r.read().decode())
            songs = data.get("songs") or []
            if not songs:
                return False
            song = songs[0]
            name = song.get("name", "").strip()
            if not name:
                return False

            # 拼完整歌名：主名 + alias（副标题/英文名）+ tns（翻译名）
            # 网易云 UI 显示格式：晴天（Sunny Day）
            extras: list[str] = []
            for alias in (song.get("alias") or []):
                a = alias.strip()
                if a and a != name:
                    extras.append(a)
            for t in (song.get("tns") or []):
                t = t.strip()
                if t and t != name and t not in extras:
                    extras.append(t)
            full_title = f"{name}（{'、'.join(extras)}）" if extras else name

            artist_items = song.get("artists") or song.get("ar") or []
            artists = [a.get("name", "").strip() for a in artist_items if a.get("name")]

            # 直接写入，不经过 LLM（保证准确性）
            asset.title = full_title
            if artists:
                asset.artist = "、".join(artists)
            album = song.get("album") or song.get("al") or {}
            if album.get("name"):
                asset.album = album["name"].strip()
            if album.get("picUrl"):
                asset.cover_url = album["picUrl"]
            duration_ms = song.get("duration") or song.get("dt")
            if duration_ms:
                asset.duration_seconds = max(1, int(duration_ms) // 1000)
            # 只用 LLM 推断 genre 和 mood（这两个 API 不返回）
            if not asset.genre or not asset.mood:
                artist_str = asset.artist or "未知"
                prompt = (
                    f"歌曲：《{full_title}》，歌手：{artist_str}\n"
                    f"请判断这首歌的风格和情绪，严格按格式回复：\n"
                    f"风格: xxx（如：流行、摇滚、电子、古典、R&B、说唱、爵士、民谣）\n"
                    f"情绪: xxx（如：欢快、治愈、励志、伤感、放松、激昂、浪漫）"
                )
                try:
                    result = self.llm.generate(prompt)
                    for line in result.strip().split("\n"):
                        if "风格" in line and ":" in line:
                            genre = line.split(":", 1)[1].strip()
                            if genre and genre != "未知":
                                asset.genre = [g.strip() for g in genre.replace("、", ",").split(",") if g.strip()]
                        if "情绪" in line and ":" in line:
                            mood = line.split(":", 1)[1].strip()
                            if mood and mood != "未知":
                                asset.mood = [m.strip() for m in mood.replace("、", ",").split(",") if m.strip()]
                except Exception:
                    pass
            return True
        except Exception:
            return False

    def _apply_title_artist_hint(self, asset: Asset, video_title: str | None) -> None:
        """Use deterministic title hints before asking the LLM to guess."""
        if not video_title:
            return
        title = video_title.strip()
        for suffix in [" - 单曲 - 网易云音乐", " - 网易云音乐"]:
            title = title.removesuffix(suffix).strip()
        for separator in (" - ", " — ", " – "):
            if separator not in title:
                continue
            name, artist = [part.strip() for part in title.split(separator, 1)]
            if name and (not asset.title or asset.title.startswith("网易云歌曲") or asset.title == "CineSonic Demo Asset"):
                asset.title = name
            if artist and not asset.artist:
                asset.artist = artist
            return
        if title and (not asset.title or asset.title.startswith("网易云歌曲") or asset.title == "CineSonic Demo Asset"):
            asset.title = title

    def _identify_from_url(self, asset: Asset, video_title: str | None = None) -> None:
        prompt = IDENTIFY_FROM_URL_TEMPLATE(
            url=asset.source_url,
            parsed_title=asset.title,
            video_title=video_title,
        )
        try:
            result = self.llm.generate(prompt)
            lines = result.strip().split("\n")
            for line in lines:
                if "歌名" in line and ":" in line:
                    name = line.split(":", 1)[1].strip().strip("《》\"'")
                    if name and name != "未知":
                        asset.title = name
                if "歌手" in line and ":" in line:
                    artist = line.split(":", 1)[1].strip()
                    if artist and artist != "未知":
                        asset.artist = artist
                if "风格" in line and ":" in line:
                    genre = line.split(":", 1)[1].strip()
                    if genre and genre != "未知":
                        asset.genre = [g.strip() for g in genre.replace("、", ",").split(",") if g.strip()]
                if "情绪" in line and ":" in line:
                    mood = line.split(":", 1)[1].strip()
                    if mood and mood != "未知":
                        asset.mood = [m.strip() for m in mood.replace("、", ",").split(",") if m.strip()]
            self.store.write_model("assets", asset.asset_id, asset)
        except Exception:
            pass

    def analyze_media(self, asset_id: str, force_refresh: bool = False) -> tuple[Asset, list[Segment]]:
        return self.media.analyze_media(asset_id, force_refresh=force_refresh)

    def _batch_classify_tracks(self, pairs: list[tuple[str, str]]) -> list[dict[str, list[str]]]:
        """批量让 LLM 判断一组 (歌名, 歌手) 的风格和情绪，一次调用处理多首。

        返回与输入等长的列表，每项 {"genre": [...], "mood": [...]}；失败则该项为空。
        """
        if not pairs:
            return []
        lines = "\n".join(f"{i}. 《{t}》- {a or '未知'}" for i, (t, a) in enumerate(pairs))
        prompt = (
            f"判断下面每首歌的风格和情绪。\n{lines}\n\n"
            f"严格输出 JSON 数组，每项对应一首（按序号），格式：\n"
            f'[{{"genre":"流行","mood":"治愈"}}]\n'
            f"风格可选：流行、摇滚、电子、古典、R&B、说唱、爵士、民谣、国风、金属。\n"
            f"情绪可选：欢快、治愈、励志、伤感、放松、激昂、浪漫、孤独。"
        )
        out: list[dict[str, list[str]]] = [{"genre": [], "mood": []} for _ in pairs]
        try:
            raw = extract_json_list(self.llm.generate(prompt)) or []
            for i, item in enumerate(raw[: len(pairs)]):
                if not isinstance(item, dict):
                    continue
                g = str(item.get("genre", "")).strip()
                m = str(item.get("mood", "")).strip()
                out[i] = {
                    "genre": [x.strip() for x in g.replace("、", ",").split(",") if x.strip()],
                    "mood": [x.strip() for x in m.replace("、", ",").split(",") if x.strip()],
                }
        except Exception:
            pass
        return out

    def import_netease_playlist(
        self,
        playlist_ref: str,
        cookie: str = "",
        user_id: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """把一个网易云歌单批量导入音乐库。

        playlist_ref 可以是歌单链接或纯 id。逐首转成 Asset 入库，
        title/artist/album/cover/duration 直接用网易云 API 的真实数据。
        返回 {"name","imported","skipped","total","tracks":[...]}。
        """
        from app.media.pipeline import netease_playlist_id
        from app.netease_auth import fetch_playlist_tracks

        pid = netease_playlist_id(playlist_ref)
        if not pid:
            raise ValueError("无法识别歌单链接，请确认是网易云歌单地址或 id。")

        data = fetch_playlist_tracks(pid, cookie=cookie, limit=limit)
        result: dict[str, Any] = {
            "name": data.get("name", ""),
            "imported": 0,
            "skipped": 0,
            "total": data.get("total", 0),
            "tracks": [],
        }
        tracks = data.get("tracks", [])
        # 批量让 LLM 判断 genre/mood（分块，每块 20 首，控制 prompt 长度）
        classifications: list[dict[str, list[str]]] = []
        for start in range(0, len(tracks), 20):
            chunk = tracks[start:start + 20]
            classifications.extend(
                self._batch_classify_tracks([(t.get("title", ""), t.get("artist", "")) for t in chunk])
            )

        existing_ids = {a.asset_id for a in self.list_assets()}
        for idx, t in enumerate(tracks):
            song_id = t.get("song_id")
            if not song_id:
                continue
            song_url = f"https://music.163.com/song?id={song_id}"
            asset = self.media.ingest_video(song_url)
            # 用歌单 API 的真实元数据覆盖占位标题
            asset.title = t.get("title") or asset.title
            if t.get("artist"):
                asset.artist = t["artist"]
            if t.get("album"):
                asset.album = t["album"]
            if t.get("cover"):
                asset.cover_url = t["cover"]
            if t.get("duration"):
                asset.duration_seconds = t["duration"]
            # 补全风格/情绪
            cls = classifications[idx] if idx < len(classifications) else {}
            if cls.get("genre"):
                asset.genre = cls["genre"]
            if cls.get("mood"):
                asset.mood = cls["mood"]
            # energy/tempo 用确定性兜底（与 analyzer 一致的做法），保证有值
            rng = random.Random(int(hashlib.sha1(asset.asset_id.encode()).hexdigest()[:8], 16))
            if not asset.tempo_bpm:
                asset.tempo_bpm = rng.randint(70, 160)
            if asset.energy_level is None:
                asset.energy_level = round(rng.uniform(0.2, 0.95), 2)
            # 关键：标记为已分析，否则推荐/歌单/品味会过滤掉这些歌
            asset.status = AssetStatus.ANALYZED
            asset.updated_at = utc_now_iso()
            self.store.write_model("assets", asset.asset_id, asset)
            if asset.asset_id in existing_ids:
                result["skipped"] += 1
            else:
                result["imported"] += 1
                existing_ids.add(asset.asset_id)
            result["tracks"].append(asset)

        # 导入后刷新品味档案，让推荐立即用上新歌
        if user_id and result["imported"]:
            library = [a for a in self.list_assets() if a.status == "analyzed"]
            self.memory.refresh_taste_profile(user_id, library)
        return result

    def list_assets(self) -> list[Asset]:
        keys = self.store.list_keys("assets")
        assets: list[Asset] = []
        for key in keys:
            asset = self.store.read_model("assets", key, Asset)
            if asset:
                assets.append(asset)
        return assets

    def delete_asset(self, asset_id: str, user_id: str | None = None) -> bool:
        deleted_asset = self.store.delete_key("assets", asset_id)
        deleted_segments = self.store.delete_key("segments", asset_id)
        deleted = deleted_asset or deleted_segments
        if deleted:
            self.memory.remove_asset_references(asset_id, user_id=user_id)
            if user_id:
                library = [a for a in self.list_assets() if a.status == "analyzed"]
                self.memory.refresh_taste_profile(user_id, library)
        return deleted

    def clear_cache(self, preserve_memory: bool = True) -> dict[str, int]:
        cleared = {
            "assets": self.store.clear_collection("assets"),
            "segments": self.store.clear_collection("segments"),
        }
        if not preserve_memory:
            cleared["memory"] = self.store.clear_collection("memory")
        return cleared

    # --- 推荐功能 ---

    def daily_recommend(
        self,
        user_id: str,
        time_of_day: str | None = None,
        count: int | None = None,
    ) -> DailyRecommendation:
        memory = self.memory.get_memory(user_id)
        library = [a for a in self.list_assets() if a.status == "analyzed"]
        if not memory.taste_profile:
            memory = self.memory.refresh_taste_profile(user_id, library)
        memory_query = self.memory.weighted_query(memory)
        query = f"{time_of_day or 'current'} {' '.join(memory.common_goals[-2:])} {memory_query}".strip()
        evidences = self.retrieve_library_evidence(query, top_k=4) if library else []
        trace = [
            f"memory_query={memory_query or 'none'}",
            f"time_bucket={time_of_day or 'auto'}",
            f"library_assets={len(library)}",
            f"evidence_chunks={len(evidences)}",
        ]
        return self.daily.generate(memory, library, time_of_day, count=count, evidences=evidences, trace=trace)

    def find_similar_assets(self, asset_id: str, top_k: int = 5) -> list[SimilarAssetResult]:
        return self.similarity.find_similar_assets(asset_id, top_k)

    def find_similar_segments(self, asset_id: str, segment_id: str, top_k: int = 5) -> list[SimilarSegmentResult]:
        return self.similarity.find_similar_segments(asset_id, segment_id, top_k)

    # --- 搜索 ---

    def search(self, user_id: str, query: str, include_external: bool = True, top_k: int = 20) -> SearchResponse:
        memory = self.memory.get_memory(user_id)
        memory_query = self.memory.weighted_query(memory)
        expanded_query = f"{query} {memory_query}".strip()
        local_results: list[Asset] = []
        query_lower = expanded_query.lower()
        for asset in self.list_assets():
            searchable = f"{asset.title} {asset.artist or ''} {' '.join(asset.genre)} {' '.join(asset.mood)}".lower()
            if any(term in searchable for term in query_lower.split()):
                local_results.append(asset)

        evidences = self.retrieve_library_evidence(expanded_query, top_k=min(top_k, 6))
        local_by_id = {asset.asset_id: asset for asset in local_results}
        for evidence in evidences:
            asset_id = str(evidence.metadata.get("asset_id", ""))
            asset = next((item for item in self.list_assets() if item.asset_id == asset_id), None)
            if asset is not None:
                local_by_id.setdefault(asset.asset_id, asset)

        external_results: list[ExternalTrack] = []
        if include_external:
            # 先从 mock 曲库搜
            external_results = self.source.search(query, limit=top_k)
            # 如果 mock 结果不够，用 LLM 补充
            if len(external_results) < 5:
                llm_results = self._llm_search(query, top_k - len(external_results))
                external_results.extend(llm_results)

        summary = self._safe_llm(
            f"搜索：{query}。结合用户记忆查询扩展为：{memory_query or '无'}。找到本地{len(local_by_id)}首，外部{len(external_results)}首。用中文简要总结搜索结果。",
            fallback=f"搜索完成：本地 {len(local_by_id)} 首，外部 {len(external_results)} 首。",
        )
        return SearchResponse(
            local=list(local_by_id.values())[:top_k],
            external=external_results[:top_k],
            summary=summary,
            evidences=evidences,
            agent_trace=[
                f"query={query}",
                f"memory_query={memory_query or 'none'}",
                f"local_hits={len(local_by_id)}",
                f"external_hits={len(external_results)}",
            ],
        )

    def _llm_search(self, query: str, limit: int) -> list[ExternalTrack]:
        prompt = LLM_SEARCH_TEMPLATE(query=query, limit=limit)
        try:
            result = self.llm.generate(prompt)
            raw = extract_json_list(result)
            if not raw:
                return []
            tracks: list[ExternalTrack] = []
            for i, item in enumerate(raw[:limit]):
                if not isinstance(item, dict):
                    continue
                tracks.append(ExternalTrack(
                    external_id=f"llm-search-{i:03d}",
                    title=item.get("title", ""),
                    artist=item.get("artist", ""),
                    genre=[item.get("genre", "")] if item.get("genre") else [],
                    mood=[item.get("mood", "")] if item.get("mood") else [],
                    source="llm",
                ))
            return tracks
        except Exception:
            return []

    # --- 收听记录 ---

    def record_listen(self, user_id: str, asset_id: str, duration: int, completed: bool, context: str | None = None) -> UserMemory:
        return self.memory.record_listen(user_id, asset_id, duration, completed, context)

    # --- 品味档案 ---

    # --- 评分 ---

    def rate_asset(self, user_id: str, asset_id: str, score: float) -> UserMemory:
        asset = self.store.read_model("assets", asset_id, Asset)
        if asset is None:
            raise ValueError(f"Unknown asset_id: {asset_id}")
        memory = self.memory.record_rating(user_id, asset, score)
        # 评分后立即刷新品味档案
        library = [a for a in self.list_assets() if a.status == "analyzed"]
        memory = self.memory.refresh_taste_profile(user_id, library)
        return memory

    # --- 品味档案 ---

    def get_taste_profile(self, user_id: str) -> TasteProfile:
        memory = self.memory.get_memory(user_id)
        if not memory.taste_profile:
            library = [a for a in self.list_assets() if a.status == "analyzed"]
            memory = self.memory.refresh_taste_profile(user_id, library)
        return memory.taste_profile or TasteProfile()

    # --- 对话 ---

    def chat(self, user_id: str, message: str, history: list[dict[str, Any]] | None = None) -> AgentAnswer:
        asset_id = self._resolve_asset_context(user_id, message)
        return self.react.run(user_id=user_id, asset_id=asset_id, query=message, top_k=5, history=history)

    # --- 记忆 ---

    def update_memory(self, request: MemoryUpdateRequest) -> tuple[UserMemory, bool]:
        return self.memory.update_memory(request)

    def record_feedback(self, request: FeedbackRequest) -> UserMemory:
        segments = []
        for key in self.store.list_keys("segments"):
            segments.extend(self.store.read_models("segments", key, Segment))
        target = next((s for s in segments if s.segment_id == request.segment_id), None)
        if target is None:
            raise ValueError(f"Unknown segment_id: {request.segment_id}")
        return self.memory.record_feedback(request.user_id, target, request.accepted)

    # --- 播放 ---

    def get_playback_url(self, track: Asset | ExternalTrack, netease_cookie: str = "") -> str | None:
        import urllib.parse
        if isinstance(track, Asset) and track.source_url:
            video_id = self._extract_youtube_id(track.source_url)
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0"
            bilibili = self._extract_bilibili_id(track.source_url)
            if bilibili:
                param, val = bilibili
                return f"https://player.bilibili.com/player.html?{param}={val}&autoplay=0&high_quality=1&danmaku=0"
            netease_id = _netease_song_id(track.source_url)
            if netease_id:
                audio = self._get_netease_audio_url(netease_id, netease_cookie)
                if audio:
                    return audio
            # NetEase source but no MP3 → try YouTube search as fallback
            video_id = self._search_youtube_video(
                f"{track.title} {track.artist or ''} official"
            )
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0"
            return track.source_url
        if isinstance(track, ExternalTrack):
            if track.playback_url and "listType=search" not in track.playback_url:
                return track.playback_url
            # Try NetEase MP3 first
            netease_id = self._search_netease(f"{track.title} {track.artist}")
            if netease_id:
                audio = self._get_netease_audio_url(netease_id, netease_cookie)
                if audio:
                    return audio
            # NetEase failed → YouTube fallback
            video_id = self._search_youtube_video(f"{track.title} {track.artist}")
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0"
            return None
        return None

    def get_external_url(self, track: Asset | ExternalTrack) -> str | None:
        import urllib.parse
        if isinstance(track, Asset) and track.source_url:
            return track.source_url
        if isinstance(track, ExternalTrack):
            query = urllib.parse.quote_plus(f"{track.title} {track.artist}")
            return f"https://music.163.com/#/search/m/?s={query}&type=1"
        return None

    # --- 两种播放模式：只听歌（音频） / 看 MV（视频） ---

    def get_audio_url(self, track: Asset | ExternalTrack, netease_cookie: str = "") -> str | None:
        """只听歌：只返回纯音频直链（网易云 MP3），不回退到 YouTube，避开机器人验证墙。"""
        if isinstance(track, Asset) and track.source_url:
            netease_id = _netease_song_id(track.source_url)
            if netease_id:
                return self._get_netease_audio_url(netease_id, netease_cookie)
            # 非网易云来源：按标题搜网易云拿音频
            netease_id = self._search_netease(f"{track.title} {track.artist or ''}")
            if netease_id:
                return self._get_netease_audio_url(netease_id, netease_cookie)
            return None
        if isinstance(track, ExternalTrack):
            netease_id = self._search_netease(f"{track.title} {track.artist}")
            if netease_id:
                return self._get_netease_audio_url(netease_id, netease_cookie)
            return None
        return None

    def get_mv_url(self, track: Asset | ExternalTrack) -> str | None:
        """看 MV：B 站优先（华语命中率高、嵌入不弹机器人验证），YouTube 仅作兜底。"""
        # 1) 已有来源链接里能直接提取 ID 的，直接用
        if isinstance(track, Asset) and track.source_url:
            bilibili = self._extract_bilibili_id(track.source_url)
            if bilibili:
                param, val = bilibili
                return f"https://player.bilibili.com/player.html?{param}={val}&autoplay=0&high_quality=1&danmaku=0"
            video_id = self._extract_youtube_id(track.source_url)
            if video_id:
                return f"https://www.youtube-nocookie.com/embed/{video_id}?autoplay=1&rel=0"

        title = track.title
        artist = getattr(track, "artist", "") or ""
        # 2) 主路径：搜 B 站
        bvid = self._search_bilibili_video(f"{title} {artist} MV".strip())
        if bvid:
            return f"https://player.bilibili.com/player.html?bvid={bvid}&autoplay=0&high_quality=1&danmaku=0"
        # 3) 兜底：搜 YouTube（nocookie 域）
        video_id = self._search_youtube_video(f"{title} {artist} MV official".strip())
        if video_id:
            return f"https://www.youtube-nocookie.com/embed/{video_id}?autoplay=1&rel=0"
        return None

    def _extract_youtube_id(self, url: str) -> str | None:
        import re
        patterns = [
            r"(?:v=|/embed/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        ]
        for p in patterns:
            m = re.search(p, url)
            if m:
                return m.group(1)
        return None

    def _extract_bilibili_id(self, url: str) -> tuple[str, str] | None:
        import re
        m = re.search(r"(BV[a-zA-Z0-9]+)", url)
        if m:
            return ("bvid", m.group(1))
        m = re.search(r"av(\d+)", url, re.IGNORECASE)
        if m:
            return ("aid", m.group(1))
        return None

    def _search_youtube_video(self, query: str) -> str | None:
        import json
        import re
        import urllib.parse
        import urllib.request
        search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        }
        try:
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as r:
                html = r.read().decode("utf-8")
        except Exception:
            return None
        m = re.search(r"var ytInitialData\s*=\s*(\{.+?\});\s*</script>", html, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                tabs = (
                    data.get("contents", {})
                    .get("twoColumnSearchResultsRenderer", {})
                    .get("primaryContents", {})
                    .get("sectionListRenderer", {})
                    .get("contents", [])
                )
                for tab in tabs:
                    for item in tab.get("itemSectionRenderer", {}).get("contents", []):
                        vid = item.get("videoRenderer", {}).get("videoId")
                        if vid:
                            return vid
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                pass
        ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
        return ids[0] if ids else None

    def _search_bilibili_video(self, query: str) -> str | None:
        """搜 B 站视频，返回 bvid。华语 MV 命中率高，嵌入不弹机器人验证。"""
        import json
        import urllib.parse
        import urllib.request
        search_url = (
            "https://api.bilibili.com/x/web-interface/search/type"
            f"?search_type=video&keyword={urllib.parse.quote(query)}"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.bilibili.com/",
            "Cookie": "buvid3=infoc;",  # B 站搜索需要一个种子 cookie
        }
        try:
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read().decode())
            if data.get("code") != 0:
                return None
            results = data.get("data", {}).get("result", []) or []
            for item in results:
                bvid = item.get("bvid")
                if bvid:
                    return bvid
        except Exception:
            pass
        return None

    def _search_netease(self, query: str) -> str | None:
        import json
        import urllib.parse
        import urllib.request
        search_url = (
            f"https://music.163.com/api/search/get/web"
            f"?s={urllib.parse.quote(query)}&type=1&limit=1&offset=0"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Referer": "https://music.163.com/",
        }
        try:
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read().decode())
            songs = data.get("result", {}).get("songs", [])
            if songs:
                return str(songs[0]["id"])
        except Exception:
            pass
        return None

    def _get_netease_audio_url(self, song_id: str, cookie: str = "") -> str | None:
        import json
        import urllib.request
        from app.netease_auth import _cookie_header
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Referer": "https://music.163.com/",
        }
        cookie_header = _cookie_header(cookie) if cookie else ""
        if cookie_header:
            headers["Cookie"] = cookie_header
        # 登录用户优先走 level 接口（支持会员音质 / 解锁 VIP 歌曲），
        # 未登录回退到旧的 enhance/player/url 接口。
        apis = []
        if cookie_header:
            apis.append(
                f"https://music.163.com/api/song/enhance/player/url/v1"
                f"?ids=[{song_id}]&level=exhigh&encodeType=aac"
            )
        apis.append(
            f"https://music.163.com/api/song/enhance/player/url"
            f"?id={song_id}&ids=[{song_id}]&br=320000"
        )
        for api in apis:
            try:
                req = urllib.request.Request(api, headers=headers)
                with urllib.request.urlopen(req, timeout=8) as r:
                    data = json.loads(r.read().decode())
                items = data.get("data", [])
                if items and items[0].get("url"):
                    return items[0]["url"].replace("http://", "https://", 1)
            except Exception:
                continue
        return None

    # --- 歌单 ---

    def generate_playlist(self, user_id: str, instruction: str) -> Playlist:
        import hashlib
        library = self.list_assets()
        lib_desc = "\n".join([f"- {a.asset_id}: {a.title} - {a.artist or '?'} ({', '.join(a.genre)}, {', '.join(a.mood)}, energy={a.energy_level})" for a in library])

        prompt = GENERATE_PLAYLIST_TEMPLATE(
            instruction=instruction, library_size=len(library), lib_desc=lib_desc,
        )
        result = self.llm.generate(prompt)
        data = extract_json_dict(result)
        if not data:
            return self._fallback_playlist(user_id, instruction, library)
        asset_map = {a.asset_id: a for a in library}
        tracks: list[Asset | ExternalTrack] = []
        for item in data.get("tracks", []):
            aid = item.get("asset_id")
            if aid and aid in asset_map:
                tracks.append(asset_map[aid])
            else:
                tracks.append(ExternalTrack(
                    external_id=hashlib.sha1(f"{item['title']}-{item['artist']}".encode()).hexdigest()[:10],
                    title=item.get("title", ""), artist=item.get("artist", ""),
                    genre=[], mood=[], source="llm",
                    playback_url=None,
                ))

        playlist = Playlist(
            playlist_id=hashlib.sha1(f"{user_id}-{instruction}".encode()).hexdigest()[:8],
            user_id=user_id, name=data.get("name", instruction),
            description=data.get("description", ""), tracks=tracks, generated_by="llm",
        )
        self.save_playlist(user_id, playlist)
        return playlist

    def auto_playlists(self, user_id: str) -> list[Playlist]:
        import hashlib
        library = self.list_assets()
        if not library:
            return []
        lib_desc = "\n".join([f"- {a.asset_id}: {a.title} - {a.artist or '?'} ({', '.join(a.genre)}, {', '.join(a.mood)})" for a in library])

        prompt = AUTO_PLAYLIST_TEMPLATE(library_size=len(library), lib_desc=lib_desc)
        result = self.llm.generate(prompt)
        raw = extract_json_list(result)
        if not raw:
            return self._fallback_auto_playlists(user_id, library)
        asset_map = {a.asset_id: a for a in library}
        playlists: list[Playlist] = []
        for item in raw:
            tracks = [asset_map[tid] for tid in item.get("track_ids", []) if tid in asset_map]
            pl = Playlist(
                playlist_id=hashlib.sha1(f"{user_id}-{item.get('name','')}".encode()).hexdigest()[:8],
                user_id=user_id, name=item.get("name", ""),
                description=item.get("description", ""), tracks=tracks, generated_by="auto",
            )
            self.save_playlist(user_id, pl)
            playlists.append(pl)
        return playlists

    def save_playlist(self, user_id: str, playlist: Playlist) -> None:
        key = f"{user_id}_{playlist.playlist_id}"
        self.store.write_model("playlists", key, playlist)

    def list_playlists(self, user_id: str) -> list[Playlist]:
        keys = self.store.list_keys("playlists")
        playlists: list[Playlist] = []
        for key in keys:
            if key.startswith(f"{user_id}_"):
                pl = self.store.read_model("playlists", key, Playlist)
                if pl:
                    playlists.append(pl)
        return playlists

    def delete_playlist(self, user_id: str, playlist_id: str) -> bool:
        return self.store.delete_key("playlists", f"{user_id}_{playlist_id}")

    def _fallback_playlist(self, user_id: str, instruction: str, library: list[Asset]) -> Playlist:
        keywords = instruction.lower().split()
        matched = [
            asset
            for asset in library
            if any(
                term in f"{asset.title} {asset.artist or ''} {' '.join(asset.genre)} {' '.join(asset.mood)}".lower()
                for term in keywords
            )
        ]
        if not matched:
            matched = sorted(
                library,
                key=lambda asset: (asset.energy_level or 0.0, asset.updated_at),
                reverse=True,
            )[:8]
        playlist = Playlist(
            playlist_id=hashlib.sha1(f"{user_id}-{instruction}".encode()).hexdigest()[:8],
            user_id=user_id,
            name=instruction or "Agent 歌单",
            description="离线回退歌单：根据你的音乐库和指令自动整理。",
            tracks=matched[:12],
            generated_by="fallback",
        )
        self.save_playlist(user_id, playlist)
        return playlist

    def _fallback_auto_playlists(self, user_id: str, library: list[Asset]) -> list[Playlist]:
        grouped: dict[str, list[Asset]] = {}
        for asset in library:
            bucket = asset.genre[0] if asset.genre else "未分类"
            grouped.setdefault(bucket, []).append(asset)
        playlists: list[Playlist] = []
        for genre, tracks in list(grouped.items())[:4]:
            playlist = Playlist(
                playlist_id=hashlib.sha1(f"{user_id}-{genre}".encode()).hexdigest()[:8],
                user_id=user_id,
                name=f"{genre}精选",
                description=f"按 {genre} 风格自动整理的离线歌单。",
                tracks=tracks[:10],
                generated_by="fallback-auto",
            )
            self.save_playlist(user_id, playlist)
            playlists.append(playlist)
        return playlists

    # --- RAG（保留兼容） ---

    def retrieve_evidence(self, asset_id: str, query: str, top_k: int = 5) -> list[RagEvidence]:
        segments = self._require_segments(asset_id)
        evidences = HybridRetriever(segments).search(query=query, top_k=top_k)
        asset = self.store.read_model("assets", asset_id, Asset)
        title = asset.title if asset else asset_id
        for evidence in evidences:
            evidence.metadata["asset_id"] = asset_id
            evidence.metadata["asset_title"] = title
        return evidences

    def retrieve_library_evidence(self, query: str, top_k: int = 5) -> list[RagEvidence]:
        ranked: list[RagEvidence] = []
        for asset in self.list_assets():
            if asset.status != "analyzed":
                continue
            ranked.extend(self.retrieve_evidence(asset.asset_id, query, top_k=min(3, top_k)))
        ranked.sort(key=lambda evidence: evidence.similarity, reverse=True)
        return ranked[:top_k]

    def recommend_with_memory(self, asset_id: str, user_id: str, goal: str, top_k: int = 3) -> AgentAnswer:
        memory = self.memory.get_memory(user_id)
        memory_query = self.memory.weighted_query(memory)
        evidences = self.retrieve_evidence(asset_id, f"{goal} {memory_query}".strip(), top_k=max(top_k * 2, top_k))
        segment_map = {segment.segment_id: segment for segment in self._require_segments(asset_id)}
        segments: list[Segment] = []
        seen: set[str] = set()
        for evidence in evidences:
            if evidence.segment_id in seen:
                continue
            segment = segment_map.get(evidence.segment_id)
            if segment is not None:
                seen.add(evidence.segment_id)
                segments.append(segment)
        lines = [
            f"{index}. {segment.timestamp} - {segment.scene_summary}"
            for index, segment in enumerate(segments[:top_k], start=1)
        ]
        answer = "基于你的记忆和当前素材，我优先推荐这些片段：\n" + "\n".join(lines) if lines else "当前素材里没有足够明显的高匹配片段。"
        return AgentAnswer(
            answer=answer,
            evidences=evidences[:top_k],
            recommended_segments=segments[:top_k],
            agent_trace=[
                f"goal={goal}",
                f"memory_query={memory_query or 'none'}",
                f"evidence_chunks={len(evidences)}",
            ],
        )

    def generate_report(self, asset_id: str) -> dict[str, Any]:
        asset = self.store.read_model("assets", asset_id, Asset)
        if asset is None:
            raise ValueError(f"Unknown asset_id: {asset_id}")
        segments = self._require_segments(asset_id)
        evidences = self.retrieve_evidence(asset_id, "high energy climax mood genre summary", top_k=4)
        return {
            "asset": asset.model_dump(mode="json"),
            "summary": f"{asset.title} 已拆分为 {len(segments)} 个片段，可用于风格检索、推荐解释和相似内容分析。",
            "top_evidences": [evidence.model_dump(mode="json") for evidence in evidences],
            "fingerprint": {
                "genre": asset.genre,
                "mood": asset.mood,
                "tempo_bpm": asset.tempo_bpm,
                "energy_level": asset.energy_level,
            },
        }

    def summarize_taste(self, user_id: str) -> str:
        memory = self.memory.get_memory(user_id)
        if not memory.taste_profile:
            library = [asset for asset in self.list_assets() if asset.status == "analyzed"]
            memory = self.memory.refresh_taste_profile(user_id, library)
        taste = memory.taste_profile or TasteProfile()
        genres = [genre for genre, _ in taste.top_genres[:4]]
        moods = [mood for mood, _ in taste.top_moods[:4]]
        prefs = memory.preferences[-3:]
        genre_text = "、".join(genres) if genres else "未形成稳定风格"
        mood_text = "、".join(moods) if moods else "暂无明显偏好"
        pref_text = "；".join(prefs) if prefs else "暂无"
        return (
            f"你的品味目前更偏向 {genre_text}，"
            f"情绪上常出现 {mood_text}，"
            f"显式表达过的偏好包括 {pref_text}。"
        )

    def recommend_for_query(self, user_id: str, goal: str, top_k: int = 5) -> DailyRecommendation:
        time_bucket = self._infer_time_bucket(goal)
        return self.daily_recommend(user_id, time_of_day=time_bucket, count=top_k)

    def _resolve_asset_context(self, user_id: str, query: str) -> str | None:
        lowered = query.lower()
        asset_sensitive = any(token in lowered for token in ["片段", "segment", "video", "素材", "场景", "镜头", "similar"])
        memory = self.memory.get_memory(user_id)
        if memory.listening_history:
            recent_asset_id = memory.listening_history[-1].asset_id
            if recent_asset_id:
                return recent_asset_id if asset_sensitive else None
        assets = self.list_assets()
        if asset_sensitive and assets:
            assets.sort(key=lambda asset: asset.updated_at, reverse=True)
            return assets[0].asset_id
        if len(assets) == 1 and asset_sensitive:
            return assets[0].asset_id
        return None

    def _infer_time_bucket(self, text: str) -> str | None:
        lowered = text.lower()
        hints = {
            "morning": ["morning", "早晨", "清晨", "早餐"],
            "focus": ["focus", "专注", "工作", "学习"],
            "afternoon": ["afternoon", "午后", "白天"],
            "evening": ["evening", "晚上", "傍晚", "通勤"],
            "night": ["night", "深夜", "睡前", "夜晚"],
        }
        for bucket, keywords in hints.items():
            if any(keyword in lowered for keyword in keywords):
                return bucket
        return None

    def _safe_llm(self, prompt: str, fallback: str) -> str:
        result = self.llm.generate(prompt)
        if result.startswith("LLM 请求失败"):
            return fallback
        return result

    def _require_segments(self, asset_id: str) -> list[Segment]:
        segments = self.media.get_segments(asset_id)
        if not segments:
            _, segments = self.analyze_media(asset_id)
        return segments


def _netease_song_id(url: str) -> str | None:
    """从各种网易云 URL 格式中提取 song id。

    支持：
      https://music.163.com/song?id=186016
      https://music.163.com/#/song?id=186016
      https://y.music.163.com/m/song/186016
      https://163cn.tv/AbCdEf  （短链，无法直接解析 id，返回 None）
    """
    return netease_song_id(url)


# 向后兼容别名
CineSonicAgent = AudioVisualAgent
