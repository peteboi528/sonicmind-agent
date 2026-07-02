"""曲风词表扩充（10 → ~30 一级标签 + 子风格分层 + 一首多标签上限 3）。

验证：
1. genres 单一事实来源自洽（父类映射、netease/Last.fm 映射的值都在词表内）；
2. tag_rules 的歌手映射产出细分曲风（中文说唱/欧美说唱/英伦摇滚…）且都是合法标签；
3. parent_genre 能把子风格上卷回一级，库匹配可粗/细两用。
"""

from __future__ import annotations

from app.genres import (
    GENRE_PARENT,
    GENRE_TO_LASTFM_EN,
    NETEASE_TAG_TO_GENRE,
    VALID_GENRE_SET,
    VALID_GENRES,
    parent_genre,
)
from app.graph.tag_rules import extract_genre_from_artist


def test_vocab_expanded_and_self_consistent():
    # 扩到至少 30 个一级标签，原 10 个仍在（向后兼容）。
    assert len(VALID_GENRES) >= 30
    for legacy in ["流行", "摇滚", "电子", "古典", "R&B", "说唱", "爵士", "民谣", "国风", "金属"]:
        assert legacy in VALID_GENRE_SET
    # 关键细分必须在词表内。
    for fine in ["中文说唱", "欧美说唱", "英伦摇滚", "独立摇滚", "另类R&B", "City Pop", "盯鞋"]:
        assert fine in VALID_GENRE_SET


def test_parent_map_points_into_vocab():
    # 子风格的父类必须是合法一级标签；子风格本身也在词表内。
    for sub, parent in GENRE_PARENT.items():
        assert sub in VALID_GENRE_SET, sub
        assert parent in VALID_GENRE_SET, parent
    assert parent_genre("中文说唱") == "说唱"
    assert parent_genre("英伦摇滚") == "摇滚"
    assert parent_genre("说唱") == "说唱"  # 一级原样返回
    assert parent_genre("未知风格X") == "未知风格X"  # 未知不崩


def test_netease_and_lastfm_maps_target_valid_genres():
    for tag, g in NETEASE_TAG_TO_GENRE.items():
        assert g in VALID_GENRE_SET, f"netease tag {tag!r} → 非法曲风 {g!r}"
    for g in GENRE_TO_LASTFM_EN:
        assert g in VALID_GENRE_SET, f"Last.fm 映射键 {g!r} 不在词表"


def test_artist_hints_emit_fine_genres():
    # 中英文说唱分流：这是用户最在意的细分。
    assert extract_genre_from_artist("Kendrick Lamar") == ["欧美说唱"]
    assert extract_genre_from_artist("宝石Gem") == ["中文说唱"]
    assert "中文说唱" in extract_genre_from_artist("GAI")
    # 英伦/独立摇滚分流。
    assert extract_genre_from_artist("Oasis") == ["英伦摇滚"]
    assert extract_genre_from_artist("Arctic Monkeys") == ["独立摇滚"]
    # 另类 R&B。
    assert "另类R&B" in extract_genre_from_artist("Frank Ocean")


def test_artist_hints_all_valid():
    """歌手映射表产出的每个曲风都必须是合法标签，否则会被 _ensure_track_tags 的
    VALID 过滤静默丢掉，等于没分类。"""
    from app.graph.tag_rules import _ARTIST_GENRE_HINTS

    for artist, genres in _ARTIST_GENRE_HINTS.items():
        for g in genres:
            assert g in VALID_GENRE_SET, f"{artist} → 非法曲风 {g!r}（不在 genres.VALID_GENRES）"


def test_classify_caps_three_genres(tmp_path):
    """分类层把 LLM 给的多曲风裁到最多 3 个、去重保序、过滤非法值。"""
    from app.agent import AudioVisualAgent
    from app.storage import JsonStore

    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    class _StubLLM:
        def generate(self, prompt, system=None, temperature=0.7, thinking=None):
            # 给 5 个（含重复 + 1 个非法），期望裁成前 3 个合法去重。
            return '[{"genre":"中文说唱,Trap,说唱,中文说唱,瞎编风格","mood":"激昂,热血,愤怒"}]'

    agent.library_svc._llm_provider = lambda: _StubLLM()
    out = agent.library_svc._classify_once([("某说唱", "某歌手")])
    assert out[0]["genre"] == ["中文说唱", "Trap", "说唱"]  # 去重、保序、裁 3、去非法
    assert len(out[0]["mood"]) <= 2  # mood 裁 2
