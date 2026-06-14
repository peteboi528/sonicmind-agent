"""锁住语言加权推荐：按曲库中/英文分布偏好同语言候选，但不排斥任一语言。

用户需求(2026-06)："曲库中英文歌比较多就多推荐英文歌，但不是完全不推荐中文歌曲。"
"""
from __future__ import annotations

from types import SimpleNamespace

from app.recommend.rerank import (
    _language_multiplier,
    detect_language,
    language_distribution,
)


def _track(title: str, artist: str = "") -> SimpleNamespace:
    return SimpleNamespace(title=title, artist=artist)


class TestDetectLanguage:
    def test_english_track(self):
        assert detect_language(_track("Blinding Lights", "The Weeknd")) == "en"

    def test_chinese_track(self):
        assert detect_language(_track("晴天", "周杰伦")) == "zh"

    def test_mixed_chinese_dominant(self):
        # 中文字符多于拉丁字母 → zh
        assert detect_language(_track("特别的人", "方大同")) == "zh"

    def test_chinese_artist_english_title(self):
        # "Lemon" 全英文标题，歌手中文 → 拉丁字母多 → en
        assert detect_language(_track("Lemon", "米津玄师")) == "en"

    def test_empty_defaults_english(self):
        assert detect_language(_track("", "")) == "en"


class TestLanguageDistribution:
    def test_all_english(self):
        tracks = [_track("Nights", "Frank Ocean"), _track("Adorn", "Miguel")]
        dist = language_distribution(tracks)
        assert dist["en"] == 1.0
        assert dist["zh"] == 0.0

    def test_mixed(self):
        tracks = [
            _track("Nights", "Frank Ocean"),
            _track("Adorn", "Miguel"),
            _track("晴天", "周杰伦"),
            _track("十年", "陈奕迅"),
        ]
        dist = language_distribution(tracks)
        assert dist["en"] == 0.5
        assert dist["zh"] == 0.5

    def test_empty_library_balanced(self):
        """空库返回均衡分布，避免冷启动把任一语言压到 0。"""
        dist = language_distribution([])
        assert dist == {"zh": 0.5, "en": 0.5}


class TestLanguageMultiplier:
    def test_english_preferred_but_chinese_not_zeroed(self):
        """英文偏好库下，英文乘子更高，但中文乘子仍 > 0（不排斥）。"""
        pref = {"en": 0.7, "zh": 0.3}
        en_mult = _language_multiplier(_track("Nights", "Frank Ocean"), pref)
        zh_mult = _language_multiplier(_track("晴天", "周杰伦"), pref)
        assert en_mult > zh_mult
        assert zh_mult > 0.8  # 中文仍保留可观权重

    def test_no_pref_neutral(self):
        """未传 lang_pref 时乘子为 1（不影响排序）。"""
        assert _language_multiplier(_track("Nights", "Frank Ocean"), None) == 1.0

    def test_multiplier_gap_is_gentle(self):
        """占比 0 vs 1 的乘子差距温和（约 35%），不是一刀切。"""
        full = _language_multiplier(_track("Nights", "Frank Ocean"), {"en": 1.0, "zh": 0.0})
        none = _language_multiplier(_track("晴天", "周杰伦"), {"en": 1.0, "zh": 0.0})
        assert abs(full - 1.15) < 0.01
        assert abs(none - 0.85) < 0.01


class TestRerankLanguageWeighting:
    def test_english_lib_ranks_english_higher(self):
        """同等其他条件下，英文偏好库应把英文候选排得更靠前——但中文仍在结果里。"""
        from app.recommend.rerank import rerank_candidates

        tracks = [
            SimpleNamespace(title="晴天", artist="周杰伦", genre=["流行"], mood=["浪漫"], external_id="zh1", source="mock"),
            SimpleNamespace(title="Nights", artist="Frank Ocean", genre=["流行"], mood=["浪漫"], external_id="en1", source="mock"),
        ]
        lang_pref = {"en": 0.8, "zh": 0.2}
        ranked = rerank_candidates(
            "推荐", tracks, taste=None, lang_pref=lang_pref, top_k=5, apply_mmr=False
        )
        titles = [getattr(t, "title", "") for t, _ in ranked]
        # 英文候选排第一，但中文候选仍在结果中（未被排除）
        assert titles[0] == "Nights"
        assert "晴天" in titles
