from __future__ import annotations

import os

import pytest

# Tests must be deterministic even when production behavior is online-first and
# a developer has a real LLM key in .env. Runtime behavior is unchanged outside
# pytest.
os.environ["LLM_API_KEY"] = ""
os.environ.setdefault("LLM_TIMEOUT_SECONDS", "1")
os.environ.setdefault("RESOURCE_LIBRARY_PATH", "data/test_resource_library.sqlite")
# 外部源默认已改为真实网易云源（NeteaseSource）；测试需离线确定，强制用 mock 目录。
os.environ["EXTERNAL_SOURCE"] = "mock"


@pytest.fixture(autouse=True)
def fake_online_music_search(monkeypatch):
    from app.agent import AudioVisualAgent
    from app.models import ExternalTrack

    def _fake_search_web_music(self, query: str, top_k: int = 5, relevance_query: str = "", offset: int = 0, **_):
        # offset>0 模拟翻页：返回一批"更深位次"的不同曲目，让延续指令去重测试
        # 能真正拿到新歌（offset=0 仍是原来那批最热曲目，兼容既有用例）。
        if offset:
            return [
                ExternalTrack(
                    external_id=f"netease-paged-{offset}-{idx}",
                    title=f"Paged Track {offset}-{idx}",
                    artist="Demo Artist",
                    genre=["R&B"],
                    mood=["放松"],
                    source="netease",
                    playback_url=f"https://music.163.com/song?id={offset}0{idx}",
                )
                for idx in range(1, top_k + 1)
            ]
        seeds = [
            ExternalTrack(
                external_id="netease-real-1",
                title="Online Focus One",
                artist="Demo Artist",
                genre=["R&B"],
                mood=["放松"],
                source="netease",
                playback_url="https://music.163.com/song?id=1",
            ),
            ExternalTrack(
                external_id="bili-real-1",
                title="Online Chill Two",
                artist="Demo Creator",
                genre=["R&B"],
                mood=["放松"],
                source="bilibili",
                playback_url="https://player.bilibili.com/player.html?bvid=BV1",
            ),
            ExternalTrack(
                external_id="yt-real-1",
                title="Online Work Three",
                artist="Demo Channel",
                genre=["电子"],
                mood=["专注"],
                source="youtube",
                playback_url="https://www.youtube.com/embed/demo",
            ),
        ]
        if len(seeds) < top_k:
            seeds.extend(
                ExternalTrack(
                    external_id=f"mock-fallback-{idx}",
                    title=f"Fallback Track {idx}",
                    artist="Mock Artist",
                    genre=["流行"],
                    mood=["放松"],
                    source="mock-fallback",
                )
                for idx in range(1, top_k - len(seeds) + 1)
            )
        return seeds[:top_k]

    monkeypatch.setattr(AudioVisualAgent, "search_web_music", _fake_search_web_music)

    async def _fake_search_web_music_async(
        self, query: str, top_k: int = 5, relevance_query: str = "",
        include_video_sources: bool = False, offset: int = 0, variants=None,
    ):
        return self.search_web_music(
            query, top_k=top_k, relevance_query=relevance_query,
            include_video_sources=include_video_sources, offset=offset, variants=variants,
        )

    monkeypatch.setattr(AudioVisualAgent, "search_web_music_async", _fake_search_web_music_async)

    async def _fake_search_videos_async(self, query: str, top_k: int = 5):
        return self.search_videos(query, top_k=top_k)

    async def _fake_search_artist_info_async(self, query: str):
        return self.search_artist_info(query)

    async def _fake_recommend_artist_albums_async(self, user_id: str, artist: str, limit: int = 6):
        return self.recommend_artist_albums(user_id, artist, limit)

    monkeypatch.setattr(AudioVisualAgent, "search_videos_async", _fake_search_videos_async)
    monkeypatch.setattr(AudioVisualAgent, "search_artist_info_async", _fake_search_artist_info_async)
    monkeypatch.setattr(AudioVisualAgent, "recommend_artist_albums_async", _fake_recommend_artist_albums_async)


@pytest.fixture(autouse=True)
def _reset_netease_album_cache():
    """专辑详情缓存是进程级全局的，测试间共享会互相污染（A 缓存的 "18893" 让 B 拿不到
    它打桩的 urlopen）。每个测试前清空，保证隔离。"""
    from app.sources.netease import clear_album_detail_cache
    clear_album_detail_cache()
    yield
