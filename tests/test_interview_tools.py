from __future__ import annotations

from types import SimpleNamespace

from app.models import Asset, ExternalTrack, ListeningEvent, TasteProfile, UserMemory
from app.services.fact_check import run_music_fact_check
from app.services.playlist_repair import analyze_playlist_repair
from app.services.profile_shift import detect_taste_shift
from app.services.recommend_explainer import build_recommend_explanation
from app.graph import nodes


class _Memory:
    def __init__(self, memory: UserMemory) -> None:
        self._memory = memory

    def get_memory(self, _user_id: str) -> UserMemory:
        return self._memory


class _Agent:
    def __init__(self, memory: UserMemory, assets: list[Asset]) -> None:
        self.memory = _Memory(memory)
        self._assets = assets

    def list_assets(self) -> list[Asset]:
        return list(self._assets)

    def search(self, _user_id: str, _query: str, include_external: bool = False, top_k: int = 8):
        return SimpleNamespace(local=self._assets[:top_k], external=[] if not include_external else [])


def test_playlist_repair_detects_duplicates_and_jumps():
    tracks = [
        ExternalTrack(external_id="1", title="Night Drive", artist="A", source="netease", genre=["Synthwave"], mood=["深夜"]),
        ExternalTrack(external_id="2", title="Night Drive", artist="A", source="netease", genre=["Synthwave"], mood=["深夜"]),
        ExternalTrack(external_id="3", title="Battle Cry", artist="B", source="netease", genre=["Metal"], mood=["热血"]),
    ]
    payload = analyze_playlist_repair(
        agent=_Agent(UserMemory(user_id="u1"), []),
        user_id="u1",
        query="修一下这个歌单",
        instruction="更适合深夜",
        target="上一轮歌单",
        prior_results=[{"type": "playlist", "playlist": SimpleNamespace(tracks=tracks)}],
    )
    kinds = {item["kind"] for item in payload["issues"]}
    assert "duplicate_tracks" in kinds
    assert "style_jump" in kinds or "energy_gap" in kinds


def test_taste_shift_detector_finds_emerging_genre():
    assets = [
        Asset(asset_id="old-1", source_url="https://x/1", title="Old Jazz", artist="Jazzer", duration_seconds=180, genre=["Jazz"], mood=["放松"], status="analyzed"),
        Asset(asset_id="new-1", source_url="https://x/2", title="New House", artist="House Hero", duration_seconds=180, genre=["House"], mood=["律动"], status="analyzed"),
    ]
    memory = UserMemory(
        user_id="u1",
        listening_history=[
            ListeningEvent(asset_id="old-1", timestamp="2026-03-01T00:00:00+00:00", completed=True),
            ListeningEvent(asset_id="new-1", timestamp="2026-06-20T00:00:00+00:00", completed=True),
            ListeningEvent(asset_id="new-1", timestamp="2026-06-21T00:00:00+00:00", completed=True),
        ],
    )
    payload = detect_taste_shift(agent=_Agent(memory, assets), user_id="u1", recent_days=30, baseline_days=120)
    assert any(item["name"] == "House" for item in payload["shift_signals"])
    assert "House" in payload["emerging_genres"]


def test_music_fact_check_marks_year_verified():
    class _FactAgent(_Agent):
        pass

    payload = run_music_fact_check(
        agent=_FactAgent(UserMemory(user_id="u1"), []),
        query="Frank Ocean 的 Blonde 是 2016 年的专辑吗",
        claims_text="Frank Ocean 的 Blonde 是 2016 年的专辑",
        plan={"intent": "music_fact_check"},
    )
    assert payload["claims"]
    assert payload["verified_claims"] or payload["uncertain_claims"]


def test_recommend_explainer_uses_recent_recommendation():
    memory = UserMemory(
        user_id="u1",
        taste_profile=TasteProfile(top_genres=[("R&B", 3.0)], top_moods=[("放松", 2.0)]),
    )
    track = ExternalTrack(external_id="1", title="Nikes", artist="Frank Ocean", source="netease", genre=["R&B"], mood=["放松"])
    recommendation = SimpleNamespace(
        tracks=[SimpleNamespace(asset=track, reason="贴近你最近常听的 R&B 和放松氛围")],
    )
    payload = build_recommend_explanation(
        agent=_Agent(memory, []),
        user_id="u1",
        query="为什么推荐这些歌",
        prior_results=[{"type": "daily_recommend", "recommendation": recommendation}],
    )
    assert payload["global_reasons"]
    assert payload["per_track_reasons"][0]["title"] == "Nikes"


def test_compose_deterministic_answers_for_new_tool_types():
    concert = nodes._compose_deterministic_answer(
        [{"type": "concert_events", "artist": "The Weeknd", "city": "", "events": [{"title": "After Hours Tour", "source_url": "https://example.com"}]}],
        SimpleNamespace(intent="concert_events"),
    )
    repair = nodes._compose_deterministic_answer(
        [{"type": "playlist_repair", "issues": [{"summary": "存在重复曲目"}], "repair_actions": [], "suggested_replacements": []}],
        SimpleNamespace(intent="playlist_repair"),
    )
    assert "The Weeknd" in concert
    assert "重复曲目" in repair


def test_concert_answer_separates_verified_events_from_weak_sources():
    concert = nodes._compose_deterministic_answer(
        [{
            "type": "concert_events",
            "artist": "The Weeknd",
            "city": "",
            "events": [{
                "title": "After Hours Til Dawn Tour",
                "date_text": "2026-10-01",
                "city": "Hong Kong",
                "venue": "启德体育园",
                "source_name": "theweeknd.com",
                "source_url": "https://www.theweeknd.com/tour",
            }],
            "unverified_sources": [{
                "title": "演出曲目单：The Weeknd 巡演",
                "source_name": "music.apple.com",
                "source_url": "https://music.apple.com/example",
            }],
        }],
        SimpleNamespace(intent="concert_events"),
    )
    assert "After Hours Til Dawn Tour" in concert
    assert "启德体育园" in concert
    assert "Apple Music" not in concert
    assert "未纳入已确认场次" in concert


def test_music_compare_returns_honest_message_when_entity_missing():
    text = nodes._compose_deterministic_answer(
        [{
            "type": "music_compare",
            "message": "我只稳定识别到《Drake》，另一侧比较对象没有解析稳，这轮先不硬做比较。",
            "entities": [{"name": "Drake", "type": "artist"}],
        }],
        SimpleNamespace(intent="music_compare"),
    )
    assert "Drake" in text
    assert "不硬做比较" in text


def test_music_compare_prefers_structured_compare_answer():
    text = nodes._compose_deterministic_answer(
        [{
            "type": "music_compare",
            "answer": "Drake 和 Future 的区别在于旋律结构 vs trap 状态，先听 Jumpman。",
            "dossier": {"entity": {"name": "Drake", "type": "artist"}},
        }],
        SimpleNamespace(intent="music_compare"),
    )

    assert "Jumpman" in text
    assert "trap 状态" in text


def test_concert_events_filters_stale_and_weak_sources():
    from app.tools.actions import _concert_events

    class Agent:
        def search_artist_info(self, _query):
            return [
                {
                    "title": "It's All a Blur Tour - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/It%27s_All_a_Blur_Tour",
                    "content": "Past tour information.",
                },
                {
                    "title": "Drake concert | Melbourne的日期及行程",
                    "url": "https://tw.trip.com/events/drake-concert-20241209",
                    "content": "Drake concert 2024-12-09 Melbourne.",
                },
                {
                    "title": "Drake Tickets, 2026 Concert Tour Dates | Ticketmaster",
                    "url": "https://www.ticketmaster.com/drake-tickets/artist/1319371",
                    "content": "Drake 2026 concert tour dates and tickets.",
                },
            ]

    payload, _summary = _concert_events(Agent(), {"artist": "Drake"})

    titles = [item["title"] for item in payload["events"]]
    weak_titles = [item["title"] for item in payload["unverified_sources"]]
    assert titles == ["Drake Tickets, 2026 Concert Tour Dates | Ticketmaster"]
    assert "It's All a Blur Tour - Wikipedia" in weak_titles
    assert "Drake concert | Melbourne的日期及行程" in weak_titles


def test_concert_events_prioritize_concrete_event_over_tour_page():
    from app.tools.actions import _concert_events

    class Agent:
        def search_artist_info(self, _query):
            return [
                {
                    "title": "The Weeknd Tour | Official Site",
                    "url": "https://www.theweeknd.com/tour",
                    "content": "Official tour page for The Weeknd.",
                },
                {
                    "title": "The Weeknd: After Hours Til Dawn Tour - 启德体育园",
                    "url": "https://www.kaitaksportspark.com.hk/tc/events-tickets/the-weeknd-after-hours-til-dawn-tour",
                    "content": "2026-10-01 Hong Kong 启德体育园 concert event page.",
                },
            ]

    payload, _summary = _concert_events(Agent(), {"artist": "The Weeknd"})

    assert len(payload["events"]) == 2
    assert payload["events"][0]["title"] == "The Weeknd: After Hours Til Dawn Tour - 启德体育园"
    assert payload["events"][0]["kind"] == "event"
    assert payload["events"][1]["kind"] == "tour_page"
