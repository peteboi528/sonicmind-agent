"""基于曲风/情绪标签的音频特征估算（deterministic tag-derived estimate，非真实测量）。

背景：网易云 API 不返回 BPM/energy，本地无音频文件可分析（DemoAnalyzer 是占位实现，
按项目「绝不随机伪造」原则保持 tempo/energy=None）。结果是 ``score_track`` 里 energy_prox
与 tempo_fit 对所有候选恒为常数——推荐 35% 的权重（energy 0.20 + tempo 0.15）零区分度，
tempo_range 的 p25-p75 修复也因 tempo_values 恒空而失效。

本模块给出**确定性的、保守的**特征估算：从已有 genre/mood 标签映射出粗粒度 BPM/energy 区间，
作为推荐/品味的可用信号。它**不是测量值**——返回值统一四舍五入到 5 BPM、energy 取标签均值，
刻意读作「一个区间」而非「精确数字」。来源用 ``Asset.features_source='estimated'`` 显式标注，
绝不冒充 measured（见 app/tools/actions.py:_audio_features 的诚实化处理）。

设计：
- tempo 来自 genre：取标签中最快的子风格（tempo 跟随 arousal，Trap 介入的说唱取 140 而非 95）。
  未命中时回退到父类（parent_genre），再未命中返回 None。
- energy 来自 mood：取各 mood energy 的均值（mood 本身就是情绪/能量描述，语义上最可信）。
  无可映射 mood 返回 None（由 score_track 优雅降级）。
- 与 genres.py 的分工：genres.py 是「曲风词表」单一事实来源；这里是「特征估算」逻辑，
  仅 import parent_genre 做父类回退，词表本身仍在 genres.py 维护。
"""

from __future__ import annotations

from app.genres import parent_genre

# ── genre → 代表性 tempo（BPM）──────────────────────────────────────────────
# 覆盖 VALID_GENRES 全集；粗粒度、四舍五入到 5，读作「区间」而非精确值。
GENRE_TEMPO_BPM: dict[str, int] = {
    # 流行
    "流行": 115,
    "华语流行": 112,
    "欧美流行": 118,
    "韩流": 120,
    "日系流行": 122,
    "City Pop": 105,
    "合成器流行": 118,
    # 摇滚
    "摇滚": 125,
    "英伦摇滚": 130,
    "独立摇滚": 130,
    "朋克": 170,
    "后摇": 120,
    "盯鞋": 110,
    "硬摇滚": 140,
    "后朋克": 145,
    # 电子
    "电子": 124,
    "House": 124,
    "Techno": 130,
    "氛围电子": 85,
    "未来贝斯": 150,
    "synthwave": 110,
    # 说唱
    "说唱": 95,
    "中文说唱": 92,
    "欧美说唱": 95,
    "Trap": 140,
    "Drill": 145,
    # R&B / soul
    "R&B": 90,
    "新灵魂": 88,
    "另类R&B": 92,
    "放克": 105,
    # 其他常见
    "独立": 110,
    "氛围": 80,
    "lo-fi": 82,
    "原声": 95,
    "世界音乐": 100,
    "拉丁": 100,
    "雷鬼": 75,
    "蓝调": 90,
    "乡村": 100,
    "古风": 85,
    "实验": 110,
    "爵士": 100,
    "金属": 150,
    "古典": 90,
    "民谣": 95,
    "国风": 85,
}

# ── mood → energy（0..1）───────────────────────────────────────────────────
# mood 本身编码了情绪/能量，因此语义上最可信。覆盖观测到的 mood + _MOOD_RULES 全集。
MOOD_ENERGY: dict[str, float] = {
    "激昂": 0.85,
    "热血": 0.88,
    "律动": 0.70,
    "欢快": 0.65,
    "励志": 0.62,
    "性感": 0.55,
    "暗黑": 0.55,
    "浪漫": 0.50,
    "梦幻": 0.40,
    "治愈": 0.35,
    "伤感": 0.30,
    "孤独": 0.28,
    "放松": 0.30,
    "慵懒": 0.25,
    "宁静": 0.22,
}


def estimate_tempo(genres: list[str] | None) -> int | None:
    """从 genre 标签估算代表性 BPM（取最快子风格；未命中回退父类）。

    tempo 跟随 arousal：一首同时标了「欧美说唱(95)+Trap(140)」的歌取 140，更贴合 Trap 介入后的实际听感。
    结果四舍五入到 5 BPM，刻意读作「区间」。无可映射 genre 返回 None。
    """
    vals: list[int] = []
    for g in genres or []:
        v = GENRE_TEMPO_BPM.get(g)
        if v is None:
            v = GENRE_TEMPO_BPM.get(parent_genre(g))
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return int(round(max(vals) / 5.0) * 5)


def estimate_energy(moods: list[str] | None) -> float | None:
    """从 mood 标签估算 energy（0..1，取均值）。无可映射 mood 返回 None。"""
    vals = [MOOD_ENERGY.get(m) for m in moods or []]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def estimate_features(genres: list[str] | None, moods: list[str] | None) -> tuple[int | None, float | None]:
    """一次性估算 (tempo_bpm, energy_level)。任一维度无信号时对应位置为 None。"""
    return estimate_tempo(genres), estimate_energy(moods)
