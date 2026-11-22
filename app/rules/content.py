from __future__ import annotations

import re
from typing import Any, TypeVar

from app.intents import expand_content_negation, normalize_content_negation

Track = TypeVar("Track")


def filter_content_exclusions(tracks: list[Track], exclusions: list[str]) -> list[Track]:
    """Hard-filter rewritten negative constraints from every result-producing path."""
    if not tracks or not exclusions:
        return tracks
    from app.recommend.rerank import detect_language

    canonical = {normalize_content_negation(item) for item in exclusions if item.strip()}
    aliases = {alias.lower() for item in canonical for alias in expand_content_negation(item) if alias.strip()}

    def blocked(track: Any) -> bool:
        text = " ".join(
            [
                str(getattr(track, "title", "") or ""),
                str(getattr(track, "artist", "") or ""),
                str(getattr(track, "album", "") or ""),
                *[str(item) for item in getattr(track, "genre", []) or []],
                *[str(item) for item in getattr(track, "mood", []) or []],
            ]
        ).lower()
        if any(alias in text for alias in aliases):
            return True
        if "中文" in canonical and detect_language(track) == "zh":
            return True
        if "英文" in canonical and detect_language(track) == "en":
            return True
        if "日语" in canonical and re.search(r"[\u3040-\u30ff]", text):
            return True
        if "韩语" in canonical and re.search(r"[\uac00-\ud7af]", text):
            return True
        if "越南" in canonical and re.search(
            r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", text
        ):
            return True
        return False

    return [track for track in tracks if not blocked(track)]
