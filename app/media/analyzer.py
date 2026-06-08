from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Protocol

from app.models import Asset, Segment


class MediaAnalyzer(Protocol):
    def analyze(self, asset: Asset, media_path: Path | None) -> list[Segment]: ...


GENRE_POOL = ["流行", "摇滚", "电子", "古典", "R&B", "说唱", "爵士", "民谣", "后摇", "独立"]
MOOD_POOL = ["欢快", "伤感", "放松", "激昂", "浪漫", "忧郁", "治愈", "热血", "宁静", "梦幻"]


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
            segments.append(Segment(
                segment_id=f"{asset.asset_id}-{i + 1:02d}",
                asset_id=asset.asset_id,
                start_seconds=start,
                end_seconds=end,
                transcript=TRANSCRIPT_POOL[transcript_choices[i]],
                keyframe_path=None,
                visual_tags=VISUAL_TAG_POOL[visual_choices[i]],
                audio_tags=AUDIO_TAG_POOL[audio_choices[i]],
                scene_summary=SUMMARY_POOL[summary_choices[i]],
            ))
        return segments
