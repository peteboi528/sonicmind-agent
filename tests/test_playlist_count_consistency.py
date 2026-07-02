"""playlist 数量一致性单测：脏数据被剔后，文案数量 = 真实展示数量。

对应线上问题：文案说"20 首歌单"，但实际展示 < 20（教程/合集混入后被剔）。
"""
from __future__ import annotations

from app.graph.nodes import _compose_deterministic_answer, _planned_arguments
from app.models import AgentPlan, ExternalTrack, Playlist
from app.tools.contracts import ToolContext
from app.tools.handlers import _playlist


def _t(title: str, artist: str = "Demo", source: str = "netease") -> ExternalTrack:
    return ExternalTrack(external_id=title, title=title, artist=artist, source=source)


class _Agent:
    def __init__(self, tracks):
        self._tracks = tracks

    def generate_playlist(self, user_id, instruction, seed_tracks=None, target_count=None):
        return Playlist(
            playlist_id="p1", user_id=user_id, name="跑步歌单",
            description="英文流行", tracks=list(self._tracks),
        )


def test_playlist_keeps_only_real_songs_and_recounts():
    """12 首正常 + 3 条 B站教程 + 2 条网易云歌单内容 → 最终只剩 12 首，数量口径一致。"""
    tracks = [
        _t("Ditto", "NewJeans"), _t("ETA", "NewJeans"), _t("Firework", "Katy Perry"),
        _t("Blinding Lights", "The Weeknd"), _t("Levitating", "Dua Lipa"),
        _t("As It Was", "Harry Styles"), _t("Anti-Hero", "Taylor Swift"),
        _t("Flowers", "Miley Cyrus"), _t("Unholy", "Sam Smith"),
        _t("Calm Down", "Rema"), _t("Cruel Summer", "Taylor Swift"), _t("Vampire", "Olivia Rodrigo"),
        # 脏数据
        _t("编曲技巧:怎么做一首流行R&B风格的音乐？", "UP主", "bilibili"),
        _t("独立流行音乐真的好难做", "UP主", "bilibili"),
        _t("跑步英文流行歌单大合集", "UP主", "bilibili"),
        _t("独立流行摇滚弹跳全集", "UP主", "bilibili"),
        _t("纯音乐合集", "UP主", "bilibili"),
    ]
    agent = _Agent(tracks)
    ctx = ToolContext(
        thread_id="t", user_id="u", query="给我 20 首跑步英文流行歌",
        agent=agent, plan={"target_count": 20},
    )
    result = _playlist({"instruction": "20 首跑步英文流行歌", "target_count": 20}, ctx)

    titles = [c["title"] for c in result.cards]
    # 脏数据全部剔除
    assert "编曲技巧:怎么做一首流行R&B风格的音乐？" not in titles
    assert not any("合集" in t or "教程" in t or "全集" in t for t in titles)
    # 只剩 12 首真实歌曲
    assert len(result.cards) == 12
    # summary 里的数量与真实展示一致（不是 20）
    assert "12" in result.summary
    assert "20" not in result.summary


def test_playlist_hygiene_report_counts_filtering_cost():
    """ResultHygieneReport：raw/cleaned/removed_invalid 计数正确，可观测过滤成本。"""
    tracks = [
        _t("Ditto", "NewJeans"), _t("Firework", "Katy Perry"), _t("ETA", "NewJeans"),
        _t("编曲技巧:怎么做一首流行R&B", "UP主", "bilibili"),
        _t("纯音乐合集", "UP主", "bilibili"),
    ]
    agent = _Agent(tracks)
    ctx = ToolContext(thread_id="t", user_id="u", query="流行歌", agent=agent, plan={"target_count": 10})
    result = _playlist({"instruction": "流行歌", "target_count": 10}, ctx)

    h = result.data["hygiene"]
    assert h["requested_count"] == 10
    assert h["raw_count"] == 5
    assert h["cleaned_count"] == 3
    assert h["removed_invalid_tracks"] == 2  # 2 条脏数据被 hygiene 剔
    assert h["raw_count"] - h["cleaned_count"] == 2


def test_playlist_planned_arguments_omit_none_target_count():
    """未指定数量时不要传 target_count=None，否则工具参数校验会把歌单链路打成 error。"""
    plan = AgentPlan(intent="playlist", tools_needed=["playlist"], target_count=None)

    args = _planned_arguments("playlist", "帮我做一个跑步歌单", plan, top_k=5)

    assert args == {"instruction": "帮我做一个跑步歌单"}


def test_journey_answer_mentions_phases():
    """音乐旅程的确定性文案必须显式分阶段，避免退化成普通曲目列表。"""
    plan = AgentPlan(intent="journey", tools_needed=["journey"])
    text = _compose_deterministic_answer(
        [{
            "type": "journey",
            "journey": {
                "instruction": "从热身到冲刺",
                "phases": [
                    {"name": "热身", "goal": "轻快进入状态", "tracks": [{"title": "Warm Up"}]},
                    {"name": "冲刺", "goal": "高能量峰值", "tracks": [{"title": "Sprint"}]},
                ],
            },
        }],
        plan,
    )

    assert "阶段 1" in text
    assert "阶段 2" in text
