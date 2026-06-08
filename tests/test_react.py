from app.agent import CineSonicAgent
from app.react_loop import ActionType, ReActLoop
from app.storage import JsonStore


def test_react_classify_similar(tmp_path):
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))
    loop = ReActLoop(agent)

    actions, _ = loop._think("find similar videos in my library", "some-id", history=None)
    similar_actions = {ActionType.SIMILAR_CROSS, ActionType.SIMILAR_INTRA}
    assert any(a in similar_actions for a in actions)


def test_react_classify_recommend(tmp_path):
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))
    loop = ReActLoop(agent)

    actions, _ = loop._think("recommend trailer highlights", "some-id", history=None)
    assert ActionType.RECOMMEND in actions


def test_react_fallback_to_retrieve(tmp_path):
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))
    loop = ReActLoop(agent)

    actions, _ = loop._think("what happens at the 2 minute mark?", "some-id", history=None)
    assert ActionType.RETRIEVE in actions


def test_react_classify_taste(tmp_path):
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))
    loop = ReActLoop(agent)

    actions, _ = loop._think("分析我的品味", None, history=None)
    assert ActionType.TASTE in actions


def test_react_full_flow(tmp_path):
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))

    asset = agent.ingest_video("https://example.com/react-test")
    agent.analyze_media(asset.asset_id)

    result = agent.chat("demo-user", "推荐一些适合晚上听的音乐")
    assert result.answer
    assert result.agent_trace
    assert len(result.agent_trace) >= 2
    assert any("recommend" in step for step in result.agent_trace)
