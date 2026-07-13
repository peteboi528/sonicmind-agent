"""适配器共享工具：AgentAnswer / StreamEvent → BotResponse 转换。"""

from __future__ import annotations

from app.adapters.protocol import BotResponse, SongCard
from app.models import AgentAnswer, StreamEvent


def answer_to_bot_response(answer: AgentAnswer) -> BotResponse:
    """将同步 AgentAnswer 转为 BotResponse。

    从 answer.recommended_segments 提取歌曲信息；
    若 segments 为空，则尝试从 answer 文本中直接构造。
    """
    cards: list[SongCard] = []
    for seg in answer.recommended_segments:
        cards.append(
            SongCard(
                title=seg.scene_summary[:80] if seg.scene_summary else "",
                artist=" · ".join(seg.audio_tags[:3]) if seg.audio_tags else "",
                reason=seg.transcript[:60] if seg.transcript else "",
            )
        )
    return BotResponse(text=answer.answer, cards=cards)


def stream_events_to_bot_response(events: list[StreamEvent]) -> BotResponse:
    """将 StreamEvent 列表转为 BotResponse。

    - candidates 事件 → 歌曲卡片
    - song_card 事件 → 单张卡片
    - final 事件 → 最终文本
    """
    final_text = ""
    cards: list[SongCard] = []

    for event in events:
        if event.type == "final":
            final_text = event.content
            # 尝试从 final payload 中提取推荐曲目
            payload = event.payload or {}
            if "recommended_segments" in payload:
                for seg in payload["recommended_segments"]:
                    cards.append(
                        SongCard(
                            title=seg.get("scene_summary", "")[:80],
                            artist=" · ".join(seg.get("audio_tags", [])[:3]),
                            reason=seg.get("transcript", "")[:60],
                        )
                    )
        elif event.type == "candidates":
            for c in (event.payload or {}).get("cards", []):
                cards.append(
                    SongCard(
                        title=c.get("title", ""),
                        artist=c.get("artist", ""),
                        cover_url=c.get("cover_url", ""),
                        playback_url=c.get("playback_url", ""),
                        reason=c.get("reason", ""),
                        source=c.get("source", ""),
                        score=c.get("score"),
                    )
                )
        elif event.type == "song_card":
            p = event.payload or {}
            cards.append(
                SongCard(
                    title=p.get("title", ""),
                    artist=p.get("artist", ""),
                    cover_url=p.get("cover_url", ""),
                    playback_url=p.get("playback_url", ""),
                    reason=p.get("reason", ""),
                    source=p.get("source", ""),
                    score=p.get("score"),
                )
            )

    return BotResponse(text=final_text, cards=cards)
