from app.agent import CineSonicAgent
from app.storage import JsonStore


def test_find_similar_assets(tmp_path):
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))

    a1 = agent.ingest_video("https://example.com/cinematic-trailer-one")
    agent.analyze_media(a1.asset_id)

    a2 = agent.ingest_video("https://example.com/cinematic-trailer-two")
    agent.analyze_media(a2.asset_id)

    a3 = agent.ingest_video("https://example.com/ambient-nature-doc")
    agent.analyze_media(a3.asset_id)

    similar = agent.find_similar_assets(a1.asset_id, top_k=5)
    assert isinstance(similar, list)
    assert all(r.asset_id != a1.asset_id for r in similar)
    if similar:
        assert similar[0].score > 0
        assert similar[0].shared_tags


def test_find_similar_segments(tmp_path):
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))

    asset = agent.ingest_video("https://example.com/test-segments")
    _, segments = agent.analyze_media(asset.asset_id)

    assert len(segments) >= 3
    target = segments[0]
    similar = agent.find_similar_segments(asset.asset_id, target.segment_id, top_k=3)
    assert isinstance(similar, list)
    assert all(r.segment.segment_id != target.segment_id for r in similar)
    if similar:
        assert similar[0].score > 0


def test_list_assets(tmp_path):
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))

    agent.ingest_video("https://example.com/video-a")
    agent.ingest_video("https://example.com/video-b")

    assets = agent.list_assets()
    assert len(assets) == 2
