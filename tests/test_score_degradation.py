"""score_track 优雅降级：tempo/energy 缺失时即时估算，仍缺失则丢弃该项并重归一化。

验证四点：
1. 全空标签（估算不出 energy/tempo）→ 丢弃两项、重归一化 genre/mood/novelty，不制造常数。
2. 可估算标签 → energy 维度恢复区分度（两首不同 mood 的歌分数不同）。
3. 全维度有值时重归一化是恒等（remaining=1.0，与旧公式向后兼容）。
4. 行为奖惩保持加性（不参与重归一化，保留 BaRT 语义）。
"""

from __future__ import annotations

from app.models import Asset, AssetStatus, TasteProfile
from app.recommend.engine import score_track


def _track(aid, genre=None, mood=None, tempo=None, energy=None):
    return Asset(
        asset_id=aid,
        source_url=f"x/{aid}",
        title=aid,
        duration_seconds=200,
        status=AssetStatus.ANALYZED,
        genre=genre or [],
        mood=mood or [],
        tempo_bpm=tempo,
        energy_level=energy,
    )


def _taste():
    # preferred_energy 默认 0.5，preferred_tempo_range 默认 [80,140]→center 110
    return TasteProfile(top_genres=[("流行", 3.0)], top_moods=[("欢快", 3.0)])


def test_unmappable_features_degrade_without_constant():
    # genre/mood 都「未分类」→ energy/tempo 估算不出 → 丢弃两项，重归一化
    t = _track("x", genre=["未分类"], mood=["未分类"])
    s = score_track(t, _taste(), ["欢快"], set())
    # genre_match=0, mood_match=0, novelty=0.2；remaining=0.65 → base = 0.10*0.2/0.65 ≈ 0.0308
    assert 0.02 < s < 0.05


def test_estimated_energy_restores_discrimination():
    # 两首 genre 相同（流行→tempo 115 估算一致），仅 mood 不同 → energy 不同 → 分数不同
    high = _track("h", genre=["流行"], mood=["激昂"])  # energy≈0.85
    low = _track("l", genre=["流行"], mood=["放松"])  # energy≈0.30
    sh = score_track(high, _taste(), ["欢快"], set())
    sl = score_track(low, _taste(), ["欢快"], set())
    assert sh != sl


def test_full_features_renormalization_is_identity():
    # 所有维度有值 → remaining=1.0，base=terms（与旧公式一致，向后兼容）
    t = _track("f", genre=["流行"], mood=["欢快"], tempo=110, energy=0.5)
    s = score_track(t, _taste(), ["欢快"], set())
    # 0.30*1 + 0.25*1 + 0.20*1 + 0.15*1 + 0.10*0.2 = 0.92
    assert abs(s - 0.92) < 1e-9


def test_behavior_reward_remains_additive():
    t = _track("hit", genre=["流行"], mood=["欢快"], tempo=110, energy=0.5)
    base = score_track(t, _taste(), ["欢快"], set())
    boosted = score_track(t, _taste(), ["欢快"], set(), {"hit": 3.0})
    assert boosted > base
    # 加性：差值恰好是 behavior 权重 * 满信号（3.0→reward 1.0）
    assert abs((boosted - base) - 0.25) < 1e-9
