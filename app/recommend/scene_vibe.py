"""场景-情绪 vibe 自动判官（M2 基础件）。

用 embedding 场景原型自动判别候选与场景的 vibe 契合度——不靠人工标注、不靠逐首 LLM。
确定性（给定 embedding 模型）、embedding 不可用时安全降级（返回 None，调用方跳过）。

为什么需要它：网易云歌曲无 valence/energy 等音频特征，系统分不清「深夜 R&B」和「下午 R&B」
（标签都叫放松/慵懒）。这里用「场景 vibe 原型」的语义距离做代理：深夜 query 下，与「深夜
内省氛围」原型语义近的候选加分、远的（如 Sunny Afternoon 这种下午向）降权。

vibe 判别的真实效果由 P1 eval 的 scene_relevance 指标度量，不在此处手工断言。
"""
from __future__ import annotations

from app.retrieval.embeddings import cosine_normalized, embeddings_available, encode

# 场景/时段 → vibe 原型（多语言，覆盖中英 + 风格词）。embedding 对齐用。
# 选词强调该场景的**区分性 vibe**（深夜=内省/迷幻/慢；下午=明亮/轻松/阳光），而非泛「放松」。
_SCENE_VIBE_PROTOTYPES: dict[str, list[str]] = {
    "深夜": [
        "深夜一个人听的内省氛围音乐",
        "nocturnal introspective moody late-night music",
        "深夜 伤感 迷幻 氛围 慢板 慵懒",
        "late night melancholic ambient slow R&B",
    ],
    "下午": [
        "下午轻松惬意明亮的音乐",
        "afternoon light relaxing bright sunny music",
        "午后 阳光 慵懒 轻快 治愈 明亮",
    ],
    "早晨": [
        "早晨清新活力的音乐",
        "morning fresh energetic bright music",
        "清晨 阳光 清爽 轻快 元气",
    ],
    "运动": [
        "运动健身高能量强节奏音乐",
        "workout high energy driving beat music",
        "跑步 动感 强节奏 快板 律动 燃",
    ],
    "学习": [
        "学习工作专注的平静背景音乐",
        "focus study calm instrumental ambient music",
        "专注 平静 纯音乐 效率 沉浸",
    ],
    "睡眠": [
        "睡前助眠安静舒缓的音乐",
        "sleep calm soothing quiet music",
        "助眠 安静 舒缓 白噪音 入睡",
    ],
}

# query → 场景。时段优先（深夜/下午/早晨是用户最常抱怨跑偏的），再覆盖运动/学习/睡眠。
# 注意：tag_rules._SCENARIO_RULES 没有「深夜/夜晚」，这里补上。
_SCENE_DETECT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("深夜", ("深夜", "夜晚", "夜里", "半夜", "午夜", "night", "late night", "midnight")),
    ("下午", ("下午", "午后", "afternoon")),
    ("早晨", ("早晨", "清晨", "早上", "morning")),
    ("运动", ("跑步", "运动", "健身", "workout", "running", "gym")),
    ("学习", ("学习", "工作", "专注", "focus", "study", "考研")),
    ("睡眠", ("睡前", "助眠", "睡眠", "sleep", "晚安")),
]


def detect_scene_vibe(query: str) -> str | None:
    """从 query 识别场景/时段；无则 None（非场景请求，不启用 vibe 判别）。"""
    q = (query or "").lower()
    for scene, keywords in _SCENE_DETECT_RULES:
        if any(kw.lower() in q for kw in keywords):
            return scene
    return None


def scene_vibe_scores(track_texts: list[str], scene: str) -> list[float] | None:
    """每个候选文本对场景的 vibe 契合度 ∈ [0,1]（越高越契合）。

    取候选向量与该场景所有原型的最大余弦（映射到 [0,1]）。embedding 不可用/无原型/空 → None，
    调用方据此跳过 vibe 判别、行为不变。
    """
    protos = _SCENE_VIBE_PROTOTYPES.get(scene)
    if not protos or not track_texts:
        return None
    if not embeddings_available():
        return None
    vecs = encode([*protos, *track_texts])
    if vecs is None or len(vecs) != len(protos) + len(track_texts):
        return None
    proto_vecs = vecs[: len(protos)]
    track_vecs = vecs[len(protos):]
    return [
        max((cosine_normalized(tv, pv) + 1.0) / 2.0 for pv in proto_vecs)
        for tv in track_vecs
    ]


def scene_vibe_score(track_text: str, scene: str) -> float | None:
    """单个候选的便捷封装。"""
    scores = scene_vibe_scores([track_text], scene)
    return None if scores is None else scores[0]


# 时段场景的「反场景」——用于对比式打分。深夜↔下午 是用户最常抱怨跑偏的一对。
# 多语言 embedding 对短歌名的正向 fit 都挤在 0.6~0.9 高位窄区间（Sober 0.672 vs Sunny
# Afternoon 0.668，差 0.004=噪声），绝对阈值分不开；改用 fit_scene − fit_anti_scene，
# 偏 anti-scene 的候选（深夜 query 里的下午向曲）会被推到负值，跨过阈值被降权。
_SCENE_ANTI: dict[str, str] = {"深夜": "下午", "下午": "深夜", "早晨": "深夜"}


def scene_vibe_penalty(texts: list[str], scene: str) -> tuple[list[float] | None, float]:
    """返回 (per_track_fit, threshold)：fit < threshold 的候选应在 rerank 里降权。

    - 有 anti-scene 的时段场景（深夜/下午/早晨）：用对比式 fit_scene − fit_anti_scene ∈ [-1,1]，
      threshold≈0.0（负=偏 anti-scene，如深夜 query 里的下午向曲）。
    - 其余场景（运动/学习/睡眠）：无干净反义，用正向 fit ∈ [0,1]，threshold≈0.45。
    - embedding 不可用 → (None, None)，调用方跳过。
    """
    fit = scene_vibe_scores(texts, scene)
    if fit is None:
        return None, 0.0
    anti = _SCENE_ANTI.get(scene)
    if not anti:
        return fit, 0.45
    anti_fit = scene_vibe_scores(texts, anti)
    if anti_fit is None or len(anti_fit) != len(fit):
        return fit, 0.45  # anti 算失败 → 安全退回正向 fit
    return [f - a for f, a in zip(fit, anti_fit, strict=False)], 0.0
