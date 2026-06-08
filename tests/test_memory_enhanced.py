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
