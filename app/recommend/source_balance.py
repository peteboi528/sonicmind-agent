from __future__ import annotations

from typing import Any, Callable


def balance_recommendation_sources(
    ranked: list[tuple[Any, Any]],
    top_k: int,
    *,
    local_ratio: float = 0.4,
    is_local_track: Callable[[Any], bool],
) -> list[tuple[Any, Any]]:
    """Interleave local and online recommendations under a soft source budget."""
    if not ranked or top_k <= 0:
        return []
    local = [item for item in ranked if is_local_track(item[0])]
    online = [item for item in ranked if not is_local_track(item[0])]
    if local_ratio <= 0:
        return online[:top_k]
    if not online:
        return local[:top_k]
    if not local:
        return online[:top_k]

    local_target = min(len(local), max(1, round(top_k * local_ratio)))
    online_target = min(len(online), top_k - local_target)
    remaining = top_k - local_target - online_target
    if remaining > 0:
        extra_online = min(remaining, len(online) - online_target)
        online_target += extra_online
        remaining -= extra_online
    if remaining > 0:
        local_target += min(remaining, len(local) - local_target)

    total = local_target + online_target
    selected: list[tuple[Any, Any]] = []
    local_used = 0
    online_used = 0
    for position in range(total):
        should_have_local = round((position + 1) * local_target / total)
        if local_used < should_have_local and local_used < local_target:
            selected.append(local[local_used])
            local_used += 1
        elif online_used < online_target:
            selected.append(online[online_used])
            online_used += 1
        elif local_used < local_target:
            selected.append(local[local_used])
            local_used += 1
    return selected
