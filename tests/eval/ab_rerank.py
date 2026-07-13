"""A/B 三锚精排：在同一候选集上对比不同权重 profile 的排序与多样性。

隔离 MMR（apply_mmr=False），纯看三锚权重差异——这是 ranking 策略 A/B 的正确方法论；
真实管线在三锚之后还叠 MMR，那是另一层，不混进本次对比。

Usage:
    python -m tests.eval.ab_rerank
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.models import Asset, TasteProfile
from app.recommend.rerank import rerank_candidates
from tests.eval.metrics import intra_list_diversity

# (semantic, personal, behavior) —— 三锚权重 profile
PROFILES: dict[str, tuple[float, float, float]] = {
    "three_anchor": (0.45, 0.30, 0.25),
    "pure_semantic": (1.0, 0.0, 0.0),
    "pure_personal": (0.0, 1.0, 0.0),
    "pure_behavior": (0.0, 0.0, 1.0),
}

QUERY = "适合跑步的高能量歌"

CANDIDATES: list[Asset] = [
    Asset(
        asset_id="c1",
        source_url="u1",
        title="Run It Up",
        artist="A",
        duration_seconds=200,
        genre=["电子"],
        mood=["激昂"],
        tempo_bpm=140,
        energy_level=0.9,
        status="analyzed",
    ),
    Asset(
        asset_id="c2",
        source_url="u2",
        title="Night Drive",
        artist="B",
        duration_seconds=210,
        genre=["电子", "合成器"],
        mood=["激昂", "律动"],
        tempo_bpm=128,
        energy_level=0.85,
        status="analyzed",
    ),
    Asset(
        asset_id="c3",
        source_url="u3",
        title="晨跑节拍",
        artist="C",
        duration_seconds=190,
        genre=["流行"],
        mood=["欢快"],
        tempo_bpm=130,
        energy_level=0.8,
        status="analyzed",
    ),
    Asset(
        asset_id="c4",
        source_url="u4",
        title="Slow Coffee",
        artist="D",
        duration_seconds=240,
        genre=["爵士"],
        mood=["放松"],
        tempo_bpm=80,
        energy_level=0.3,
        status="analyzed",
    ),
    Asset(
        asset_id="c5",
        source_url="u5",
        title="Pump Iron",
        artist="E",
        duration_seconds=180,
        genre=["摇滚"],
        mood=["热血"],
        tempo_bpm=150,
        energy_level=0.95,
        status="analyzed",
    ),
    Asset(
        asset_id="c6",
        source_url="u6",
        title="霓虹冲刺",
        artist="F",
        duration_seconds=195,
        genre=["电子"],
        mood=["热血"],
        tempo_bpm=135,
        energy_level=0.88,
        status="analyzed",
    ),
    Asset(
        asset_id="c7",
        source_url="u7",
        title="Ballad Cry",
        artist="G",
        duration_seconds=300,
        genre=["流行"],
        mood=["伤感"],
        tempo_bpm=70,
        energy_level=0.2,
        status="analyzed",
    ),
    Asset(
        asset_id="c8",
        source_url="u8",
        title="Sprint Mix",
        artist="H",
        duration_seconds=185,
        genre=["说唱", "电子"],
        mood=["激昂"],
        tempo_bpm=145,
        energy_level=0.92,
        status="analyzed",
    ),
]

# 行为信号：用户听完 c1/c5（喜欢），秒跳 c7。让 behavior 锚有区分度。
BEHAVIOR: dict[str, float] = {"c1": 2.5, "c5": 1.8, "c7": -1.0}


def make_taste() -> TasteProfile:
    """模拟一个平时爱听爵士/放松的用户——口味与「跑步高能量」查询故意错配，
    让 pure_personal（偏爱爵士/放松）与查询语义显著分叉，A/B 才有看头。"""
    return TasteProfile(
        top_genres=[("爵士", 1.0), ("民谣", 0.5)],
        top_moods=[("放松", 1.0), ("治愈", 0.5)],
        top_artists=[],
        preferred_energy=0.3,
        preferred_tempo_range=[70, 100],
    )


def semantic_mode() -> str:
    """语义锚当前模式：dense(embeddings) 还是 TF 回退（弱锚）。"""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return "TF 回退（弱锚；无 embeddings 时 pure_semantic 会退化为个性化）"
    return "dense（embeddings 可用，pure_semantic 独立有效）"


def apply_profile(name: str) -> None:
    """切换 settings 的三锚权重（rerank 在调用时读取，故立即生效）。"""
    sem, per, beh = PROFILES[name]
    settings.tri_anchor_w_semantic = sem
    settings.tri_anchor_w_personal = per
    settings.tri_anchor_w_behavior = beh


def run_profile(name: str) -> tuple[list[tuple[Any, Any]], float]:
    """返回 (top-5 (track, breakdown), 列内多样性)。"""
    apply_profile(name)
    ranked = rerank_candidates(
        QUERY,
        CANDIDATES,
        make_taste(),
        behavior_scores=BEHAVIOR,
        top_k=5,
        apply_mmr=False,
    )
    tracks = [t for t, _ in ranked]
    return ranked, intra_list_diversity(tracks)


def print_comparison() -> None:
    print(f"Query: {QUERY!r}   候选 {len(CANDIDATES)} 首（隔离 MMR，纯三锚对比）")
    print(f"用户口味: 爵士/放松（与「跑步高能量」查询错配）   语义锚: {semantic_mode()}\n")
    print(f"{'profile':<16} {'diversity':>9}   top-5")
    print("-" * 78)
    for name in PROFILES:
        sem, per, beh = PROFILES[name]
        ranked, div = run_profile(name)
        titles = " > ".join(f"{t.title}[{'/'.join(t.genre + t.mood)}]" for t, _ in ranked)
        print(f"{name:<16} {div:>9.3f}   {titles}")
        print(f"{'':<16} {'':>9}   weights sem={sem}/per={per}/beh={beh}")
    print()
    print("解读：用户口味=爵士/放松，但查询=跑步高能量——锚点分叉：")
    print("  pure_personal 错配地把爵士/放松顶上来（c4 Slow Coffee）；")
    print("  pure_behavior 顶有正向收听的 c1/c5；three_anchor 在口味↔行为间折中；")
    print("  pure_semantic 匹配跑步/高能量（无 embeddings 时退化为个性化，见上方语义锚模式）。")


if __name__ == "__main__":
    print_comparison()
