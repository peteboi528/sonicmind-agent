"""app/recommend/features.py 标签估算 + LibraryService.backfill_estimated_features 回填。

验证：估算值确定性正确、空/不可映射标签返回 None、tempo 取最快子风格并四舍五入到 5、
energy 取 mood 均值；回填只填 None 不覆盖已有值、置 features_source='estimated'。
离线隔离：用 tmp_path 构造独立 JsonStore，绝不触碰真实 data/store（见 conftest 的 STORE_ROOT 隔离）。
"""
from __future__ import annotations

from app.agent import AudioVisualAgent
from app.models import Asset, AssetStatus
from app.recommend.features import estimate_energy, estimate_features, estimate_tempo
from app.storage import JsonStore


# ── 纯函数：估算确定性 ────────────────────────────────────────────────────────
def test_tempo_takes_fastest_subgenre():
    # 欧美说唱(95) + Trap(140) → 取最快的 Trap=140（tempo 跟随 arousal，贴合 Trap 介入后的听感）
    assert estimate_tempo(["欧美说唱", "Trap"]) == 140


def test_tempo_rounds_to_nearest_five():
    assert estimate_tempo(["电子"]) == 125   # 124 → 125
    assert estimate_tempo(["House"]) == 125  # 124 → 125
    assert estimate_tempo(["流行"]) == 115   # 已是 5 的倍数


def test_tempo_empty_or_unmappable_is_none():
    assert estimate_tempo([]) is None
    assert estimate_tempo(["未分类"]) is None
    assert estimate_tempo(None) is None


def test_energy_single_and_mean():
    assert estimate_energy(["暗黑"]) == 0.55
    # 律动(0.70)+梦幻(0.40) 均值 = 0.55（取整无歧义）
    assert estimate_energy(["律动", "梦幻"]) == 0.55


def test_energy_empty_or_unmappable_is_none():
    assert estimate_energy([]) is None
    assert estimate_energy(["未分类"]) is None
    assert estimate_energy(None) is None


def test_estimate_features_tuple():
    tempo, energy = estimate_features(["欧美说唱", "Trap"], ["暗黑"])
    assert tempo == 140
    assert energy == 0.55
    assert estimate_features([], []) == (None, None)


# ── 回填：填 None、不覆盖、置 source ──────────────────────────────────────────
def test_backfill_fills_none_sets_source_and_does_not_clobber(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    # a：无 tempo/energy 且可估算（Trap+暗黑）→ 应被填
    a = Asset(asset_id="a", source_url="x", title="a", duration_seconds=200,
              status=AssetStatus.ANALYZED, genre=["Trap"], mood=["暗黑"])
    # b：已有值（暂视作真实测量，source=None）→ 不应被覆盖
    b = Asset(asset_id="b", source_url="x", title="b", duration_seconds=200,
              status=AssetStatus.ANALYZED, genre=["民谣"], mood=["放松"],
              tempo_bpm=88, energy_level=0.2)
    # c：无可映射标签 → 保持 None
    c = Asset(asset_id="c", source_url="x", title="c", duration_seconds=200,
              status=AssetStatus.ANALYZED, genre=["未分类"], mood=["未分类"])
    for ast in (a, b, c):
        agent.store.write_model("assets", ast.asset_id, ast)

    result = agent.backfill_estimated_features()
    assert result == {"updated": 1, "skipped": 1, "unchanged": 1}

    a2 = agent.store.read_model("assets", "a", Asset)
    assert a2.tempo_bpm == 140
    assert a2.energy_level == 0.55
    assert a2.features_source == "estimated"

    b2 = agent.store.read_model("assets", "b", Asset)
    assert b2.tempo_bpm == 88       # 未覆盖
    assert b2.energy_level == 0.2   # 未覆盖
    assert b2.features_source is None  # 未动

    c2 = agent.store.read_model("assets", "c", Asset)
    assert c2.tempo_bpm is None
    assert c2.energy_level is None
    assert c2.features_source is None
