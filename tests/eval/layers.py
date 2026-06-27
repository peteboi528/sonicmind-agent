"""候选管线层级 eval：直接测 hygiene / scene-vibe / local-ratio 三个子系统。

互补于 run.py / regress.py（端到端 chat 质量）：那套测「整条对话输出」，但候选来自
conftest 的固定 mock，测不到「假歌来了 hygiene 挡没挡」「深夜 vibe 分得开分不开」。
这里直接喂**构造的候选集**给各层函数，确定性度量——把用户真实踩过的坑固化成回归门禁。

不依赖 LLM、不依赖真网络（scene-vibe 依赖本地 embedding 模型，缺失时该维度 skip 并诚实标注）。

Usage:
    python -m tests.eval.layers                  # 跑 + 打印 + 对比 layers_baseline.json
    python -m tests.eval.layers --update-baseline
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from app.intents import extract_content_negations
from app.models import ExternalTrack
from app.recommend.hygiene import filter_music_tracks
from app.recommend.scene_vibe import scene_vibe_penalty
from tests.offline_fakes import configure_offline_env

BASELINE_PATH = Path(__file__).parent / "layers_baseline.json"


# ── hygiene 层：构造「假歌 + 真歌」候选集，验 hygiene 挡假歌、留真歌 ──────────
@dataclass
class HygieneCase:
    case_id: str
    query: str
    junk: list[tuple[str, str]]          # (title, artist) —— 必须被滤掉
    legit: list[tuple[str, str]] = field(default_factory=list)  # 必须保留


HYGIENE_CASES: list[HygieneCase] = [
    HygieneCase(
        case_id="deep-night-functional-and-mood",
        query="推荐几首适合深夜的歌",
        junk=[
            ("慵懒的午后", "轻音乐钢琴曲"),                 # 功能艺人
            ("Sunny Afternoon(慵懒午后)", "瑶啊瑶、极音阁"),  # 括号 mood 描述
            ("雨爱 - R&B氛围男声", "小仓鼠要早睡"),           # 破折号 mood 描述
        ],
        legit=[
            ("Sober", "Yo Trane"),
            ("If You Let Me", "Alina Baraz"),
        ],
    ),
    HygieneCase(
        case_id="cover-instrumental",
        query="推荐几首歌",
        junk=[
            ("夜曲(钢琴曲)(原唱：周杰伦)", "翻唱账号"),       # 括号 + 钢琴曲 + 原唱
            ("某歌 - 助眠版", "网名用户"),                    # 破折号 + 助眠
            ("Beat - 8D Audio", "user"),                     # 8d 改版
        ],
        legit=[
            ("Ditto", "NewJeans"),
            ("Vampire - Live", "Olivia Rodrigo"),            # 合法破折号后缀，须保留
        ],
    ),
]


def _tracks(pairs: list[tuple[str, str]]) -> list[ExternalTrack]:
    return [
        ExternalTrack(external_id=f"{t}|{a}", title=t, artist=a, source="netease")
        for t, a in pairs
    ]


def run_hygiene() -> dict:
    """每个 case：junk 全被滤掉、legit 全留下 → pass。返回 per-case + pass_rate。"""
    results = []
    patcher = (
        patch("app.recommend.hygiene.embeddings_available", lambda: False)
        if os.getenv("ENABLE_EMBEDDINGS", "").lower() == "false"
        else None
    )
    ctx = patcher if patcher is not None else nullcontext()
    with ctx:
        for case in HYGIENE_CASES:
            tracks = _tracks(case.junk) + _tracks(case.legit)
            accepted, _ = filter_music_tracks(tracks, case.query, allow_maybe=False)
            accepted_titles = {t.title for t in accepted}
            junk_survived = [t for t in case.junk if t[0] in accepted_titles]
            legit_dropped = [t for t in case.legit if t[0] not in accepted_titles]
            passed = not junk_survived and not legit_dropped
            results.append({
                "case_id": case.case_id,
                "passed": passed,
                "junk_survived": junk_survived,
                "legit_dropped": legit_dropped,
            })
    passed_n = sum(1 for r in results if r["passed"])
    return {
        "cases": results,
        "pass_rate": round(passed_n / len(results), 3) if results else None,
    }


# ── scene-vibe 层：验「深夜 vibe」与「下午 vibe」能被分开 ──────────────────────
@dataclass
class SceneVibeCase:
    case_id: str
    scene: str
    higher: str   # 期望契合度更高的文本
    lower: str    # 期望契合度更低的文本


SCENE_VIBE_CASES: list[SceneVibeCase] = [
    # 描述性探针（含场景词，基本能分开——验证机制本身）
    SceneVibeCase("night-vs-afternoon", "深夜",
                  higher="深夜 伤感 慵懒 R&B 慢板 内省",
                  lower="Sunny Afternoon 慵懒午后 明亮 轻快"),
    SceneVibeCase("afternoon-vs-night", "下午",
                  higher="午后 阳光 慵懒 轻快 明亮",
                  lower="深夜 伤感 迷幻 慢板 内省"),
    SceneVibeCase("morning-vs-night", "早晨",
                  higher="清晨 阳光 清爽 轻快 元气",
                  lower="深夜 伤感 迷幻 慢板"),
    # 真实短标题（用户实际场景：标题露馅的下午曲在深夜 query 里应低分）。
    # 这才是 scene-vibe 对用户痛点的真实考验；标题里没时段词的纯 vibe 区分仍是已知天花板。
    SceneVibeCase("real-afternoon-title-in-night", "深夜",
                  higher="Sober",            # Yo Trane，无时段词
                  lower="Sunny Afternoon"),  # 标题明写 afternoon——深夜里必须更低
    SceneVibeCase("real-night-title-in-afternoon", "下午",
                  higher="Sunny Afternoon",
                  lower="夜曲"),             # 周杰伦「夜曲」——下午场景里应低于 Sunny Afternoon
]


def run_scene_vibe() -> dict:
    """每个 case：higher 的（对比式）契合度 > lower → pass，并报告 margin。
    margin < 0.05 视为「弱区分」（方向对但落在 embedding 噪声内）——诚实标注。
    embedding 不可用 → 整维 skip。"""
    patcher = (
        patch("app.recommend.scene_vibe.embeddings_available", lambda: False)
        if os.getenv("ENABLE_EMBEDDINGS", "").lower() == "false"
        else None
    )
    ctx = patcher if patcher is not None else nullcontext()
    with ctx:
        probe, _ = scene_vibe_penalty([SCENE_VIBE_CASES[0].higher], SCENE_VIBE_CASES[0].scene)
    if probe is None:
        return {"skipped": True, "reason": "embedding 模型不可用", "cases": [], "discrimination_rate": None}
    results = []
    weak_n = 0
    with ctx:
        for case in SCENE_VIBE_CASES:
            fits, _ = scene_vibe_penalty([case.higher, case.lower], case.scene)
            hi, lo = (fits or [None, None])[:2]
            margin = round(hi - lo, 3) if (hi is not None and lo is not None) else None
            passed = hi is not None and lo is not None and hi > lo
            weak = passed and margin is not None and margin < 0.05
            if weak:
                weak_n += 1
            results.append({
                "case_id": case.case_id, "scene": case.scene,
                "higher_fit": round(hi, 3) if hi is not None else None,
                "lower_fit": round(lo, 3) if lo is not None else None,
                "margin": margin, "passed": passed, "weak": weak,
            })
    passed_n = sum(1 for r in results if r["passed"])
    return {
        "skipped": False,
        "cases": results,
        "discrimination_rate": round(passed_n / len(results), 3) if results else None,
        "weak_count": weak_n,  # 方向对但 margin<0.05（噪声内）的个数
        "avg_margin": round(sum(r["margin"] for r in results if r["margin"] is not None) / max(1, len(results)), 3),
    }


# ── local-ratio 层：验「不要/减少 local」检测 ───────────────────────────────
@dataclass
class LocalRatioCase:
    case_id: str
    query: str
    expect: float


LOCAL_RATIO_CASES: list[LocalRatioCase] = [
    LocalRatioCase("no-local", "推荐几首，不要local", 0.0),
    LocalRatioCase("no-local-alt", "不要本地歌曲，全要线上的", 0.0),
    LocalRatioCase("no-local-natural", "推荐几首适合放松的歌，不要本地库里的", 0.0),
    LocalRatioCase("reduce-local", "减少local 推荐几首", 0.15),
    LocalRatioCase("reduce-local-alt", "少推本地", 0.15),
    LocalRatioCase("prefer-online", "多用线上结果", 0.15),
    LocalRatioCase("no-signal", "推荐几首适合跑步的歌", 0.4),  # 默认值
]


def run_local_ratio() -> dict:
    from app.agent import _local_ratio_from_query  # 局部导入，避免拉起整个 agent 依赖
    results = []
    for case in LOCAL_RATIO_CASES:
        ratio = _local_ratio_from_query(case.query, default=0.4)
        results.append({
            "case_id": case.case_id, "query": case.query,
            "expected": case.expect, "actual": ratio,
            "passed": abs(ratio - case.expect) < 1e-9,
        })
    passed_n = sum(1 for r in results if r["passed"])
    return {"cases": results, "accuracy": round(passed_n / len(results), 3) if results else None}


# ── content-negation 层：验“不要中文/越南/日语”约束抽取是否稳定 ────────────────
@dataclass
class ContentNegationCase:
    case_id: str
    query: str
    expected: list[str]


CONTENT_NEGATION_CASES: list[ContentNegationCase] = [
    ContentNegationCase("zh-negation", "不要中文歌曲", ["中文"]),
    ContentNegationCase("vi-negation", "不要越南语歌曲", ["越南"]),
    ContentNegationCase("ja-negation", "别放日本语歌", ["日语"]),
    ContentNegationCase("ko-negation", "排除韩文音乐", ["韩语"]),
    ContentNegationCase("cantonese-negation", "no Cantonese music", ["粤语"]),
]


def run_content_negation() -> dict:
    results = []
    for case in CONTENT_NEGATION_CASES:
        actual = extract_content_negations(case.query)
        results.append({
            "case_id": case.case_id,
            "query": case.query,
            "expected": case.expected,
            "actual": actual,
            "passed": actual == case.expected,
        })
    passed_n = sum(1 for r in results if r["passed"])
    return {"cases": results, "accuracy": round(passed_n / len(results), 3) if results else None}


# ── 汇总 / baseline ───────────────────────────────────────────────────────
def _summary(hygiene: dict, scene: dict, local: dict, negation: dict) -> dict:
    return {
        "hygiene_pass_rate": hygiene["pass_rate"],
        "scene_vibe_discrimination_rate": scene["discrimination_rate"],
        "scene_vibe_avg_margin": scene.get("avg_margin"),
        "scene_vibe_weak_count": scene.get("weak_count"),
        "scene_vibe_skipped": scene.get("skipped", False),
        "local_ratio_accuracy": local["accuracy"],
        "content_negation_accuracy": negation["accuracy"],
    }


def _print_report(hygiene: dict, scene: dict, local: dict, negation: dict, summary: dict) -> None:
    print("\n=== Hygiene（假歌须滤掉、真歌须留下）===")
    for r in hygiene["cases"]:
        flag = "PASS" if r["passed"] else "FAIL"
        extra = []
        if r["junk_survived"]:
            extra.append(f"junk 漏过: {r['junk_survived']}")
        if r["legit_dropped"]:
            extra.append(f"真歌误杀: {r['legit_dropped']}")
        print(f"  [{flag}] {r['case_id']}" + (f"  ({'; '.join(extra)})" if extra else ""))
    print(f"  hygiene_pass_rate = {summary['hygiene_pass_rate']}")

    print("\n=== Scene-vibe（深夜/下午 vibe 能否分开；对比式打分）===")
    if scene.get("skipped"):
        print(f"  SKIPPED: {scene['reason']}")
    else:
        for r in scene["cases"]:
            flag = "PASS" if r["passed"] else "FAIL"
            weak = "  ⚠️弱区分(margin<0.05)" if r.get("weak") else ""
            print(f"  [{flag}] {r['case_id']} (scene={r['scene']}: "
                  f"higher={r['higher_fit']} > lower={r['lower_fit']}, margin={r['margin']}){weak}")
        print(f"  discrimination_rate = {summary['scene_vibe_discrimination_rate']}"
              f"  avg_margin = {summary['scene_vibe_avg_margin']}"
              f"  weak_count = {summary['scene_vibe_weak_count']}")

    print("\n=== Local-ratio（不要/减少 local 检测）===")
    for r in local["cases"]:
        flag = "PASS" if r["passed"] else "FAIL"
        print(f"  [{flag}] {r['case_id']}: '{r['query']}' → {r['actual']} (expect {r['expected']})")
    print(f"  local_ratio_accuracy = {summary['local_ratio_accuracy']}")

    print("\n=== Content-negation（硬排除项抽取）===")
    for r in negation["cases"]:
        flag = "PASS" if r["passed"] else "FAIL"
        print(f"  [{flag}] {r['case_id']}: '{r['query']}' → {r['actual']} (expect {r['expected']})")
    print(f"  content_negation_accuracy = {summary['content_negation_accuracy']}")

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="候选管线层级 eval")
    parser.add_argument("--update-baseline", action="store_true", help="用当前结果覆写 baseline")
    args = parser.parse_args()

    configure_offline_env()
    hygiene = run_hygiene()
    scene = run_scene_vibe()
    local = run_local_ratio()
    negation = run_content_negation()
    summary = _summary(hygiene, scene, local, negation)
    _print_report(hygiene, scene, local, negation, summary)

    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\n[baseline 已写入 {BASELINE_PATH}]")
        return 0

    if BASELINE_PATH.exists():
        prev = json.loads(BASELINE_PATH.read_text())
        print(f"\n[对比 baseline {BASELINE_PATH.name}]")
        regressed = False
        for k, v in summary.items():
            old = prev.get(k)
            mark = ""
            if old is not None and v is not None and isinstance(v, (int, float)):
                if v < old:
                    mark = f"  ⚠️ 退化 ({old} → {v})"
                    regressed = True
                elif v > old:
                    mark = f"  ✅ 改善 ({old} → {v})"
            print(f"  {k}: {v}  (baseline {old}){mark}")
        return 1 if regressed else 0

    print("\n[无 baseline，建议跑 `python -m tests.eval.layers --update-baseline` 立基线]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
