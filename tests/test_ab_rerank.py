"""A/B 三锚 profile 逻辑的快速确定性测试（无 LLM，进 CI）。

验证：四个 profile 都能产出 top-5、diversity 落在 [0,1]、纯行为锚会把有行为信号的
曲目顶上来、settings 权重切换对 rerank 立即生效。
"""
from __future__ import annotations

import pytest

from app.config import settings
from tests.eval.ab_rerank import BEHAVIOR, CANDIDATES, PROFILES, apply_profile, run_profile
from tests.eval.metrics import intra_list_diversity


@pytest.fixture(autouse=True)
def _restore_weights():
    """A/B 会改全局 settings 权重；每个用例后还原，避免污染其它测试。"""
    saved = (
        settings.tri_anchor_w_semantic,
        settings.tri_anchor_w_personal,
        settings.tri_anchor_w_behavior,
    )
    yield
    settings.tri_anchor_w_semantic, settings.tri_anchor_w_personal, settings.tri_anchor_w_behavior = saved


def test_profiles_produce_rankings():
    for name in PROFILES:
        ranked, div = run_profile(name)
        assert len(ranked) == 5, f"{name} 应返回 top-5"
        assert 0.0 <= div <= 1.0


def test_pure_behavior_surfaces_behavioral_tracks():
    """纯行为锚应把有正向收听信号的曲目（c1/c5）排到前列。"""
    ranked, _ = run_profile("pure_behavior")
    top3_ids = {t.asset_id for t, _ in ranked[:3]}
    assert top3_ids & set(BEHAVIOR), "纯行为 top-3 应包含有行为信号的曲目"
    # c7 是秒跳（负分），不应出现在纯行为的 top-5
    assert all(t.asset_id != "c7" for t, _ in ranked), "秒跳曲 c7 不该进纯行为 top-5"


def test_settings_mutation_visible_to_rerank():
    """rerank 调用时读 settings，apply_profile 后立即生效。"""
    apply_profile("pure_behavior")
    assert settings.tri_anchor_w_behavior == 1.0
    apply_profile("three_anchor")
    assert settings.tri_anchor_w_behavior == 0.25


def test_diversity_metric_range():
    assert intra_list_diversity([]) == 1.0
    assert intra_list_diversity([CANDIDATES[0]]) == 1.0
    div = intra_list_diversity(CANDIDATES[:4])
    assert 0.0 <= div <= 1.0
