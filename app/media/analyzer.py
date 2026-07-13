"""音视频分段分析（demo / 离线演示）。

诚实说明：当前默认实现 ``DemoAnalyzer`` 是**确定性的占位分析器**——按 asset_id 哈希成种，
从固定的视觉/音频/转写/摘要池采样，生成 6 个结构正确的占位 Segment，目的是让
「入库 → 证据库 → 检索」整条 RAG 链路在没有真实 ASR/CV 时也能跑通、稳定演示
（见 docs/EXPLAINER.md §6 离线优先）。它**不**做真实语音转写或视觉理解。

真实分析（Whisper 转写 / CLIP 视觉标签 / ffmpeg BPM 抽取）是后续扩展点：实现一个新的
``MediaAnalyzer``（遵循下方 Protocol）并在 app/media/pipeline.py 替换即可，Segment 数据结构
与下游检索无需改动。asset.genre/mood/tempo/energy 的真实来源是 tag_rules 规则映射与各数据源
元数据（enrich）；未识别时 pipeline 标「未分类」、tempo/energy 保持 None，绝不随机伪造。
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Protocol

from app.models import Asset, Segment


class MediaAnalyzer(Protocol):
    """分段分析协议：把 Asset 切成带模态证据（transcript/视觉/音频/摘要）的 Segment 列表。

    真实实现应基于 media_path 做音视频处理；当前默认注册的是下方确定性占位实现 DemoAnalyzer。
    """

    def analyze(self, asset: Asset, media_path: Path | None) -> list[Segment]: ...


VISUAL_TAG_POOL = [
    ["wide shot", "city lights", "establishing scene"],
    ["medium shot", "motion", "human subject"],
    ["fast cuts", "contrast lighting", "dynamic camera"],
    ["hero shot", "dramatic gesture", "high contrast"],
    ["close up", "slower motion", "warm light"],
    ["closing frame", "logo-safe space", "fade out"],
    ["aerial view", "landscape", "natural light"],
    ["handheld", "intimate framing", "shallow depth"],
    ["tracking shot", "neon signs", "urban texture"],
    ["static frame", "symmetry", "minimal composition"],
]

AUDIO_TAG_POOL = [
    ["ambient", "low energy", "soft texture"],
    ["steady beat", "rising rhythm", "clean mix"],
    ["rising energy", "drums", "cinematic tension"],
    ["climax", "strong bass", "wide dynamics"],
    ["release", "melodic", "medium energy"],
    ["outro", "resolved harmony", "low energy"],
    ["synth pad", "atmospheric", "reverb heavy"],
    ["percussive", "staccato", "tight mix"],
    ["orchestral swell", "brass", "epic build"],
    ["acoustic", "fingerpicking", "intimate"],
]

TRANSCRIPT_POOL = [
    "The opening introduces the location with calm ambience and a slow visual rhythm.",
    "A character or performer begins moving through the scene while the beat becomes clearer.",
    "The music gains momentum and the edit becomes faster, creating a trailer-like build.",
    "The strongest emotional peak appears with dense sound, dramatic motion, and memorable imagery.",
    "The pace relaxes and the content gives space for reflection after the peak.",
    "The ending resolves the theme and leaves a clean outro for credits or a call to action.",
    "An unexpected shift in tone introduces a new visual motif and rhythmic pattern.",
    "Layered textures build gradually as the camera reveals more of the environment.",
    "A solo instrument carries the melody while the visuals focus on detail and texture.",
    "The energy plateaus at a sustained intensity before the final resolution.",
]

SUMMARY_POOL = [
    "A quiet opening scene that gives viewers spatial context and emotional baseline.",
    "The story begins to move from atmosphere into intention.",
    "A high-potential build-up moment for promotional editing.",
    "The best candidate for a trailer climax or short-form highlight.",
    "A useful transition section after the main impact moment.",
    "A clean ending that works for final branding or recap.",
    "An atmospheric interlude that resets the viewer's expectations.",
    "A rhythmically driven passage suited for montage editing.",
    "A contemplative moment that deepens emotional engagement.",
    "A sustained peak that maintains tension without resolution.",
]


class DemoAnalyzer:
    """确定性占位分析器（demo / 离线演示用）。

    输出由 asset_id 哈希决定（同一素材每次结果一致），便于测试与稳定演示；
    但内容来自固定池，**不代表真实音视频内容**。替换为真实 MediaAnalyzer 即可接入真实分析。
    """

    def analyze(self, asset: Asset, media_path: Path | None) -> list[Segment]:
        seed = int(hashlib.sha1(asset.asset_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        num_segments = 6
        duration = asset.duration_seconds
        chunk = duration // num_segments

        visual_choices = rng.sample(range(len(VISUAL_TAG_POOL)), num_segments)
        audio_choices = rng.sample(range(len(AUDIO_TAG_POOL)), num_segments)
        transcript_choices = rng.sample(range(len(TRANSCRIPT_POOL)), num_segments)
        summary_choices = rng.sample(range(len(SUMMARY_POOL)), num_segments)

        segments: list[Segment] = []
        for i in range(num_segments):
            start = i * chunk
            end = min((i + 1) * chunk, duration)
            segments.append(
                Segment(
                    segment_id=f"{asset.asset_id}-{i + 1:02d}",
                    asset_id=asset.asset_id,
                    start_seconds=start,
                    end_seconds=end,
                    transcript=TRANSCRIPT_POOL[transcript_choices[i]],
                    keyframe_path=None,
                    visual_tags=VISUAL_TAG_POOL[visual_choices[i]],
                    audio_tags=AUDIO_TAG_POOL[audio_choices[i]],
                    scene_summary=SUMMARY_POOL[summary_choices[i]],
                )
            )
        return segments
