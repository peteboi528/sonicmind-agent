"""LibraryService —— 音乐库内容与资产生命周期编排。

从 `AudioVisualAgent` 抽离的库操作层：入库（ingest/enrich/analyze）、网易云歌单
批量导入（import_netease_playlist）、曲目分类兜底（_batch_classify_tracks /
_ensure_track_tags）、资产读写与进程内缓存（list_assets / _invalidate_assets_cache）、
资产删除/清缓存/候选池清理，以及资源库 track 读取（list_resource_tracks）。

依赖通过构造注入（store/media/memory/library/llm），与其它 service 一致；agent 侧
保留同名薄委托，外部 `agent.ingest_video` / `agent.list_assets` / 测试 monkeypatch
不受影响。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from app.config import settings
from app.library import ResourceLibrary
from app.llm.protocol import LLMProvider
from app.llm.structured import extract_json_list
from app.media.pipeline import MediaPipeline
from app.memory import MemoryManager
from app.models import Asset, AssetStatus, EnrichResponse, utc_now_iso
from app.prompts import IDENTIFY_FROM_URL_TEMPLATE
from app.rules.recommend import _netease_song_id
from app.sources import bilibili as bilibili_source
from app.sources import netease as netease_source
from app.sources import youtube as youtube_source
from app.storage import JsonStore

logger = logging.getLogger(__name__)


class LibraryService:
    def __init__(
        self,
        store: JsonStore,
        media: MediaPipeline,
        memory: MemoryManager,
        library: ResourceLibrary,
        llm_provider: Callable[[], LLMProvider],
    ) -> None:
        self.store = store
        self.media = media
        self.memory = memory
        self.library = library
        # 动态取 llm：agent 侧 monkeypatch self.llm 后立即生效，保持搬家前行为
        # （agent._batch_classify_tracks 等委托到这里，用的一直是 agent 当前 llm）。
        self._llm_provider = llm_provider
        # 进程内缓存状态（从 agent 搬来）。构造期不缓存：那时 store 常为空，缓存空快照会
        # 污染后续请求。agent 构造末尾调 enable_cache() 开启，第一个真实请求才填充。
        self._assets_cache: list[Asset] | None = None
        self._assets_synced_dirty: bool = True
        self._caching_enabled: bool = False

    @property
    def llm(self) -> LLMProvider:
        return self._llm_provider()

    def enable_cache(self) -> None:
        """构造完成后开启 list_assets 缓存。"""
        self._caching_enabled = True

    def ingest_video(self, url: str, force_refresh: bool = False) -> Asset:
        asset = self.media.ingest_video(url, force_refresh=force_refresh)
        self._invalidate_assets_cache()
        return asset

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
                self._invalidate_assets_cache()
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
        if "youtube.com" in url or "youtu.be" in url:
            title = youtube_source.fetch_youtube_title(url)
            if title:
                return title

        if "163.com" in url or "163cn.tv" in url:
            song_id = _netease_song_id(url)
            title = netease_source.fetch_netease_title(url, song_id)
            if title:
                return title

        if "bilibili.com" in url:
            title = bilibili_source.fetch_bilibili_title(url)
            if title:
                return title

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
            logger.debug("yt-dlp title fallback failed for url=%s", url, exc_info=True)
        return None

    def _enrich_from_netease(self, asset: Asset, song_id: str) -> bool:
        """从网易云 API 直接获取 title/artist，不走 LLM 猜测，再用 LLM 补 genre/mood。"""
        try:
            detail = netease_source.fetch_netease_song_detail(song_id)
            if not detail:
                return False
            song = detail.get("raw") or {}
            name = (detail.get("title") or "").strip()
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

            artists = [a.strip() for a in (detail.get("artist") or "").split("、") if a.strip()]

            # 直接写入，不经过 LLM（保证准确性）
            asset.title = full_title
            if artists:
                asset.artist = "、".join(artists)
            if detail.get("album"):
                asset.album = detail["album"]
            if detail.get("cover"):
                asset.cover_url = detail["cover"]
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
                    logger.debug("LLM genre/mood inference failed for song_id=%s", song_id, exc_info=True)
            return True
        except Exception:
            logger.debug("NetEase enrichment failed for song_id=%s", song_id, exc_info=True)
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
            self._invalidate_assets_cache()
        except Exception:
            logger.debug("URL identity inference failed for asset_id=%s", asset.asset_id, exc_info=True)

    def analyze_media(self, asset_id: str, force_refresh: bool = False) -> tuple[Asset, list]:
        asset, segments = self.media.analyze_media(asset_id, force_refresh=force_refresh)
        self._invalidate_assets_cache()
        self.library.upsert_asset(asset)
        return asset, segments

    _VALID_GENRES = {"流行", "摇滚", "电子", "古典", "R&B", "说唱", "爵士", "民谣", "国风", "金属"}

    # 网易云歌单 tags → 本系统曲风词表的映射（歌单级 tags 是导入时唯一可靠的曲风线索）
    _NETEASE_TAG_TO_GENRE = {
        "R&B/Soul": "R&B", "R&B": "R&B", "Soul": "R&B", "蓝调": "R&B",
        "摇滚": "摇滚", "Rock": "摇滚", "金属": "金属", "Metal": "金属", "朋克": "摇滚",
        "电子": "电子", "Electronic": "电子", "House": "电子", "Techno": "电子", "EDM": "电子",
        "说唱": "说唱", "Rap": "说唱", "Hip-Hop": "说唱", "嘻哈": "说唱",
        "爵士": "爵士", "Jazz": "爵士", "布鲁斯": "爵士",
        "古典": "古典", "Classical": "古典", "纯音乐": "古典",
        "民谣": "民谣", "Folk": "民谣", "乡村": "民谣",
        "流行": "流行", "Pop": "流行",
        "国风": "国风", "古风": "国风", "中国风": "国风",
    }

    def _playlist_tags_to_genres(self, tags: list[str]) -> list[str]:
        """把网易云歌单 tags 映射成本系统曲风（用作整单兜底）。无映射则返回空。"""
        genres: list[str] = []
        for tag in tags:
            g = self._NETEASE_TAG_TO_GENRE.get(tag)
            if g and g not in genres:
                genres.append(g)
        return genres

    def _batch_classify_tracks(self, pairs: list[tuple[str, str]]) -> list[dict[str, list[str]]]:
        """批量让 LLM 判断一组 (歌名, 歌手) 的风格和情绪，一次调用处理多首。

        返回与输入等长的列表，每项 {"genre": [...], "mood": [...]}；失败则该项为空。
        会做一次重试：首次解析后仍为空的项，重新发一个只含这些歌的小批再问一次，
        减少落到「中性默认」兜底的数量（提升 R&B 等英文歌名的分类命中率）。
        """
        if not pairs:
            return []
        out = self._classify_once(pairs)
        # 重试：收集首轮没拿到 genre 的项，单独再问一次
        missing = [i for i, r in enumerate(out) if not r.get("genre")]
        if missing:
            retry_pairs = [pairs[i] for i in missing]
            retried = self._classify_once(retry_pairs)
            for slot, r in zip(missing, retried, strict=False):
                if r.get("genre"):
                    out[slot] = r
        return out

    def _classify_once(self, pairs: list[tuple[str, str]]) -> list[dict[str, list[str]]]:
        lines = "\n".join(f"{i}. 《{t}》- {a or '未知'}" for i, (t, a) in enumerate(pairs))
        prompt = (
            f"判断下面每首歌的风格（genre）和情绪（mood）。\n"
            f"歌手名是判断风格的重要线索：看歌手名是否包含或暗示特定风格。\n\n"
            f"{lines}\n\n"
            f"严格输出 JSON 数组，每项对应一首（按序号），格式：\n"
            f'[{{"genre":"说唱","mood":"激昂"}}]\n\n'
            f"风格可选：流行、摇滚、电子、古典、R&B、说唱、爵士、民谣、国风、金属。\n"
            f"情绪可选：欢快、治愈、励志、伤感、放松、激昂、浪漫、孤独、律动、慵懒、热血、暗黑。\n\n"
            f"判断指南：\n"
            f"- 歌手名含 Ft./feat./× 或多位歌手 → 可能是说唱/R&B 合作曲\n"
            f"- 英文歌名 + 中文歌手 → 可能是 R&B/说唱/独立\n"
            f"- 歌手名含 Gem/Trap/Lil/K/制作人代号 → 倾向说唱\n"
            f"- 歌手名含 keshi/Dean/Crush/Zion.T/SZA/The Weeknd → R&B\n"
            f"- 摇滚/金属/朋克相关关键词 → 摇滚\n"
            f"- DJ/Remix/电音/Beat → 电子\n"
            f"- 不确定的宁可标「流行」也不要瞎猜小众风格\n"
        )
        out: list[dict[str, list[str]]] = [{"genre": [], "mood": []} for _ in pairs]
        try:
            raw = extract_json_list(self.llm.generate(prompt)) or []
            for i, item in enumerate(raw[: len(pairs)]):
                if not isinstance(item, dict):
                    continue
                g = str(item.get("genre", "")).strip()
                m = str(item.get("mood", "")).strip()
                genres = [x.strip() for x in g.replace("、", ",").split(",") if x.strip()]
                # 只保留在合法集合内的风格，过滤 LLM 偶发的自由发挥
                genres = [x for x in genres if x in self._VALID_GENRES]
                out[i] = {
                    "genre": genres,
                    "mood": [x.strip() for x in m.replace("、", ",").split(",") if x.strip()],
                }
        except Exception:
            logger.debug("Track classification failed for %s tracks", len(pairs), exc_info=True)
        return out

    # 确定性兜底词表（与 tag_rules 的可选值保持一致）
    _FALLBACK_GENRES = ["流行", "摇滚", "电子", "古典", "R&B", "说唱", "爵士", "民谣", "国风", "金属"]
    _FALLBACK_MOODS = ["欢快", "治愈", "励志", "伤感", "放松", "激昂", "浪漫", "宁静", "律动", "梦幻", "暗黑", "性感"]

    def _ensure_track_tags(
        self,
        title: str,
        artist: str,
        genre: list[str],
        mood: list[str],
        playlist_genres: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """按可靠性逐层补全 genre/mood：
        1) LLM 分类结果（传入的 genre/mood）——最准；
        2) 关键词规则从歌名+歌手推断（tag_rules extract_genre/mood）；
        3) 歌手名→风格映射表（tag_rules extract_genre_from_artist）——覆盖已知艺人；
        4) 歌单级 tags 映射的曲风（playlist_genres）——整单线索；
        5) 仍为空 → genre 标「未分类」，绝不用 hash 随机或假「流行」污染品味画像。
        """
        from app.graph.tag_rules import extract_genre, extract_genre_from_artist, extract_mood

        text = f"{title} {artist}"
        if not genre:
            genre = extract_genre(text)
        if not mood:
            mood = extract_mood(text)
        # 歌手名映射兜底：比歌单 tags 更精准，覆盖 keshi/Drake/The Weeknd 等知名艺人
        if not genre and artist:
            genre = extract_genre_from_artist(artist)
        # 用歌单整单曲风兜底（网易云 tags 映射结果）
        if not genre and playlist_genres:
            genre = list(playlist_genres)
        # 仍为空 → 如实标「未分类」，不猜
        if not genre:
            genre = ["未分类"]
        if not mood:
            mood = ["放松"]
        return genre, mood

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
        # 歌单级 tags 映射成曲风，作为整单兜底（网易云歌曲级无曲风，歌单 tags 是唯一可靠线索）
        playlist_genres = self._playlist_tags_to_genres(data.get("tags", []))
        # 批量让 LLM 判断 genre/mood（每块 8 首：20 首会让 DeepSeek 超时整批失败）
        classifications: list[dict[str, list[str]]] = []
        for start in range(0, len(tracks), 8):
            chunk = tracks[start:start + 8]
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
            # 补全风格/情绪：LLM → 关键词规则 → 确定性兜底，三层保证永不为空
            cls = classifications[idx] if idx < len(classifications) else {}
            genre, mood = self._ensure_track_tags(
                asset.title, asset.artist or "", cls.get("genre") or [], cls.get("mood") or [],
                playlist_genres=playlist_genres,
            )
            asset.genre = genre
            asset.mood = mood
            # 诚实化：tempo/energy 无真实测量时保持 None（下游 score_track 用默认值兜底），
            # 不再用 rng 随机伪造具体数值（与 pipeline.analyze_media 一致）。
            # genre/mood 已由上方 _ensure_track_tags 基于真实曲名/歌手推断，这里不重复伪造。
            # 关键：标记为已分析，否则推荐/歌单/品味会过滤掉这些歌
            asset.status = AssetStatus.ANALYZED
            asset.updated_at = utc_now_iso()
            self.store.write_model("assets", asset.asset_id, asset)
            self._invalidate_assets_cache()
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
        # 进程内缓存：list_assets 是 O(库大小) 的逐文件磁盘读+反序列化，且一次请求内
        # 被多处反复调用（search/summarize_taste/list_resource_tracks/rerank…）。
        # 资产只在 ingest/enrich/analyze/delete/clear 时变动——这些点显式失效缓存。
        # 库大时这一项是超时主因之一，缓存把"每请求 ×N 次全量读"压成一次。
        # 构造期（_caching_enabled=False）不缓存：那时 store 常为空，缓存空快照会污染
        # 后续请求。enable_cache() 在 agent 构造末尾开启，第一个真实请求才填充。
        cached = self._assets_cache
        if cached is not None:
            return list(cached)
        keys = self.store.list_keys("assets")
        assets: list[Asset] = []
        for key in keys:
            asset = self.store.read_model("assets", key, Asset)
            if asset:
                assets.append(asset)
        if self._caching_enabled:
            self._assets_cache = assets
        return list(assets)

    def _invalidate_assets_cache(self) -> None:
        """资产写入/删除/清空后调用，确保下次 list_assets 读到最新，并标记需重新同步到 SQLite。"""
        self._assets_cache = None
        self._assets_synced_dirty = True

    def delete_asset(self, asset_id: str, user_id: str | None = None) -> bool:
        deleted_asset = self.store.delete_key("assets", asset_id)
        deleted_segments = self.store.delete_key("segments", asset_id)
        deleted = deleted_asset or deleted_segments
        if deleted:
            self._invalidate_assets_cache()
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
        self._invalidate_assets_cache()
        if not preserve_memory:
            cleared["memory"] = self.store.clear_collection("memory")
        # 专辑详情是纯性能缓存（非用户数据），主动清缓存时一并清掉，确保下次点击重新取最新。
        try:
            from app.sources.netease import clear_album_detail_cache
            cleared["album_detail"] = clear_album_detail_cache()
        except Exception:
            logger.debug("clear_album_detail_cache failed", exc_info=True)
        return cleared

    def cleanup_resource_library(self) -> dict[str, int]:
        """清理候选池污染：删历史 fallback/mock 假候选 + 指向已删 asset 的僵尸 local 行。

        新代码已在入库口拦截 fallback，本方法清存量；可由启动钩子或 /cache 主动触发。
        """
        live_ids = {asset.asset_id for asset in self.list_assets()}
        removed_fallback = self.library.purge_fallback_sources()
        removed_orphan = self.library.purge_orphan_local(live_ids)
        if removed_fallback or removed_orphan:
            logger.info(
                "候选池清理：删除 fallback 假候选 %d 行、僵尸 local %d 行",
                removed_fallback, removed_orphan,
            )
        return {"fallback": removed_fallback, "orphan_local": removed_orphan}

    def list_resource_tracks(self, limit: int = 100):
        # sync_assets 把 JSON 资产同步进 SQLite，是 O(库大小) 的写。资产没变时重复同步
        # 纯属浪费（similar_artists 每次拉 2500 就触发一次全量 re-upsert）。只在资产
        # 实际变动后同步一次。
        if self._assets_synced_dirty:
            self.library.sync_assets(self.list_assets())
            self._assets_synced_dirty = False
        return self.library.list_tracks(limit)
