from app.agent import CineSonicAgent
from app.memory import score_entries
from app.models import FeedbackRequest, MemoryEntry, MemoryUpdateRequest
from app.storage import JsonStore


def test_structured_preferences_created(tmp_path):
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))

    agent.update_memory(MemoryUpdateRequest(
        user_id="test-user",
        event="I like cinematic tension and strong bass.",
    ))
    memory = agent.memory.get_memory("test-user")
    assert memory.structured_preferences
    assert memory.structured_preferences[0].text == "cinematic tension and strong bass"
    assert memory.structured_preferences[0].frequency == 1


def test_frequency_bumps_on_repeat(tmp_path):
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))

    agent.update_memory(MemoryUpdateRequest(user_id="test-user", event="I like strong bass."))
    agent.update_memory(MemoryUpdateRequest(user_id="test-user", event="I like strong bass."))
    memory = agent.memory.get_memory("test-user")
    entry = next(e for e in memory.structured_preferences if "strong bass" in e.text)
    assert entry.frequency == 2


def test_decay_scoring():
    fresh = MemoryEntry(text="fresh", frequency=2, last_used="2026-05-26T00:00:00+00:00")
    old = MemoryEntry(text="old", frequency=2, last_used="2026-04-01T00:00:00+00:00")
    scored = score_entries([fresh, old])
    assert scored[0][0].text == "fresh"
    assert scored[0][1] > scored[1][1]


def test_feedback_reinforces_preference(tmp_path):
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))

    asset = agent.ingest_video("https://example.com/feedback-test")
    _, segments = agent.analyze_media(asset.asset_id)

    agent.update_memory(MemoryUpdateRequest(
        user_id="test-user",
        event="I like cinematic tension.",
    ))

    seg_with_tension = None
    for seg in segments:
        if any("cinematic" in t or "tension" in t for t in seg.audio_tags):
            seg_with_tension = seg
            break

    if seg_with_tension:
        agent.record_feedback(FeedbackRequest(
            user_id="test-user",
            segment_id=seg_with_tension.segment_id,
            accepted=True,
        ))
        memory = agent.memory.get_memory("test-user")
        entry = next(e for e in memory.structured_preferences if "cinematic tension" in e.text)
        assert entry.frequency >= 2


def test_weighted_query_includes_taste_profile(tmp_path):
    """回归：上传歌曲算出的品味档案必须进入推荐查询。

    历史 bug：weighted_query 只读 structured_preferences，无视 taste_profile，
    导致用户上传一堆摇滚后，在线推荐查询完全不含'摇滚'，只能返回泛化垃圾。
    """
    from app.models import Asset, TasteProfile, UserMemory

    agent = CineSonicAgent(JsonStore(tmp_path / "store"))
    memory = UserMemory(
        user_id="u1",
        taste_profile=TasteProfile(
            top_genres=[("摇滚", 12.0), ("英伦摇滚", 12.0)],
            top_moods=[("励志", 12.0)],
        ),
    )
    query = agent.memory.weighted_query(memory)
    assert "摇滚" in query
    assert "英伦摇滚" in query
    assert "励志" in query


def test_weighted_query_empty_without_any_signal(tmp_path):
    """无偏好无品味时返回空串（不崩）。"""
    from app.models import UserMemory

    agent = CineSonicAgent(JsonStore(tmp_path / "store"))
    assert agent.memory.weighted_query(UserMemory(user_id="u2")) == ""
