from app.agent import CineSonicAgent
from app.memory import score_entries
from app.models import ExternalTrack, FeedbackRequest, MemoryEntry, MemoryUpdateRequest, SavedAlbum
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
    from app.models import TasteProfile, UserMemory

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


def test_weighted_query_no_repeat_and_cross_source_dedup(tmp_path):
    """回归：偏好不应被重复 N 次（旧 int(weight*2) 把高频歌手堆 7-8 次，纯噪声、
    还污染搜索/LLM 上下文）；且 structured_preferences 与 taste_profile.top_artists
    同一歌手必须去重，不能各塞一份。"""
    from app.models import MemoryEntry, TasteProfile, UserMemory

    agent = CineSonicAgent(JsonStore(tmp_path / "store"))
    memory = UserMemory(
        user_id="u1",
        structured_preferences=[
            MemoryEntry(text="The Weeknd", frequency=10, source="test"),
        ],
        taste_profile=TasteProfile(top_artists=[("the weeknd", 12.0)]),
    )
    query = agent.memory.weighted_query(memory)
    # "The Weeknd" 整串只出现一次（既不因高频重复，也不因 taste_profile 再加一份）
    assert query.lower().count("the weeknd") == 1


def test_weighted_query_can_keep_artist_memory_soft_for_generic_tasks(tmp_path):
    from app.models import TasteProfile, UserMemory

    agent = CineSonicAgent(JsonStore(tmp_path / "store"))
    memory = UserMemory(
        user_id="u-soft",
        taste_profile=TasteProfile(
            top_genres=[("R&B", 10.0)],
            top_moods=[("暗黑", 8.0)],
            top_artists=[("The Weeknd", 12.0)],
        ),
    )

    query = agent.memory.weighted_query(memory, include_artists=False)

    assert "R&B" in query
    assert "暗黑" in query
    assert "The Weeknd" not in query


def test_saved_albums_contribute_to_taste_profile(tmp_path):
    from app.agent import AudioVisualAgent

    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    saved = SavedAlbum(
        album_id="blonde",
        user_id="u-album",
        name="Blonde",
        artist="Frank Ocean",
        tags=["alternative r&b", "art pop"],
        tracks=[
            ExternalTrack(external_id="1", title="Nikes", artist="Frank Ocean", source="netease"),
            ExternalTrack(external_id="2", title="Self Control", artist="Frank Ocean", source="netease"),
        ],
    )

    agent.save_album("u-album", saved)
    taste = agent.get_taste_profile("u-album")

    assert any(genre == "alternative r&b" for genre, _ in taste.top_genres)
    assert any(artist == "frank ocean" for artist, _ in taste.top_artists)


def test_refresh_preserves_genre_mood_when_library_has_no_tags(tmp_path):
    """曲库曲目无 genre/mood 标签时，refresh 不该清空已积累的曲风/情绪（治「每次更新库消失」）。"""
    from app.agent import AudioVisualAgent
    from app.models import Asset, AssetStatus

    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    # 1) 有 genre/mood 标签的库 → 建立 taste
    tagged = [Asset(
        asset_id="t1", source_url="x", title="t", duration_seconds=200,
        status=AssetStatus.ANALYZED, genre=["说唱"], mood=["治愈"],
    )]
    agent.memory.refresh_taste_profile("u", tagged)
    saved_genres = [g for g, _ in agent.memory.get_memory("u").taste_profile.top_genres]
    saved_moods = [m for m, _ in agent.memory.get_memory("u").taste_profile.top_moods]
    assert saved_genres and saved_moods

    # 2) 换成无 genre/mood 标签的库（如网易云导入缺失）→ 不该清空
    untagged = [Asset(
        asset_id="u1", source_url="x", title="u", duration_seconds=200,
        status=AssetStatus.ANALYZED,
    )]
    agent.memory.refresh_taste_profile("u", untagged)
    after = agent.memory.get_memory("u").taste_profile
    assert [g for g, _ in after.top_genres] == saved_genres
    assert [m for m, _ in after.top_moods] == saved_moods
