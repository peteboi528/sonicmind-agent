"""Explicit response-language instructions and deterministic replies."""

from __future__ import annotations

import re
from typing import Literal

ReplyLanguage = Literal["zh", "en", "ja", "es"]
_LANGUAGE_NAMES = {"zh": "Simplified Chinese", "en": "English", "ja": "Japanese", "es": "Spanish"}
_LANGUAGE_REQUESTS: tuple[tuple[ReplyLanguage, re.Pattern[str]], ...] = (
    (
        "zh",
        re.compile(
            r"(?:请)?(?:用中文(?:回答|回复|收尾|总结|讲|说|写)|回到中文|换回中文|分别用一句中文讲|[，,]\s*中文(?=[。.!！?？\s]|$))"
        ),
    ),
    (
        "en",
        re.compile(
            r"\b(?:(?:please\s+)?(?:answer|reply|respond|explain|write|summarize)\s+in English|(?<=profile )in English(?=[.!?\s]*$))\b",
            re.I,
        ),
    ),
    ("ja", re.compile(r"日本語で(?:返事して|答えて|回答して|説明して|(?=[、。\s]|いま|今))")),
    ("es", re.compile(r"\b(?:(?:responde|contesta|explica|escribe)\s+en espa[ñn]ol)\b", re.I)),
)

_TEXT = {
    "no_tracks": (
        "这轮没有拿到可追溯的音乐候选；我不会用未核实歌名硬凑结果。",
        "I could not find any verified tracks that match this request.",
        "条件に合う確認済みの曲は見つかりませんでした。",
        "No encontré canciones verificadas que cumplan con esta solicitud.",
    ),
    "no_video": (
        "这轮没有搜到视频结果。",
        "I could not find a matching music video this time.",
        "条件に合う動画は見つかりませんでした。",
        "No encontré un video musical que coincida con la búsqueda.",
    ),
    "no_info": (
        "抱歉，暂时没找到相关信息。",
        "I could not verify reliable information for this request.",
        "この質問に答えられる確かな情報は見つかりませんでした。",
        "No encontré información fiable para responder a esta consulta.",
    ),
    "greeting": (
        "你好，我在。有什么音乐上的事可以帮你?",
        "Hello. What would you like to explore in music?",
        "こんにちは。どんな音楽をお探しですか？",
        "Hola. ¿Qué música te gustaría explorar?",
    ),
    "intro": (
        "我按在线优先策略整理了 {count} 个可追溯候选：",
        "I found {count} tracks from the available sources that you can review.",
        "確認できた候補を {count} 曲見つけました。下の一覧から選べます。",
        "Encontré {count} canciones en las fuentes disponibles. Puedes revisarlas en la lista.",
    ),
    "shortfall": (
        "你要求 {target} 首，目前找到了 {count} 首。",
        "You asked for {target} tracks. I could only find {count} that matched.",
        "ご希望は {target} 曲でしたが、確認できたのは {count} 曲です。",
        "Pediste {target} canciones. Encontré {count} que cumplen con la solicitud.",
    ),
    "playlist_definition": (
        "歌单是一组按某个主题、场景或顺序整理的歌曲，可以保存下来反复播放。",
        "A playlist is a collection of tracks grouped for a theme, activity, or listening order. You can save it and play it again later.",
        "プレイリストとは、テーマや場面、再生順に合わせてまとめた曲の一覧です。保存してあとから聴けます。",
        "Una lista de reproducción reúne canciones por tema, actividad u orden. Puedes guardarla para escucharla más tarde.",
    ),
    "no_profile": (
        "还没有足够品味数据。",
        "I do not have enough preference data to describe your listening taste yet.",
        "今のところ、好みをまとめるだけの情報がありません。",
        "Todavía no tengo suficientes datos para resumir tus gustos musicales.",
    ),
    "profile": (
        "已记录的音乐偏好有：{items}。",
        "Here are the music preferences I have recorded for you: {items}.",
        "これまでに記録した音楽の好みは次のとおりです。{items}。",
        "Estas son las preferencias musicales que tengo registradas: {items}.",
    ),
    "albums": (
        "找到 {count} 张专辑。",
        "I found {count} albums for this artist in the available sources.",
        "このアーティストのアルバムを {count} 枚確認できました。",
        "Encontré {count} álbumes de este artista en las fuentes disponibles.",
    ),
    "zero": (
        "已按你的要求跳过推荐，未添加任何曲目。",
        "I skipped the recommendations as requested and added no tracks.",
        "ご希望どおり、おすすめの曲は追加していません。",
        "Omití las recomendaciones como pediste y no añadí canciones.",
    ),
    "error": (
        "这轮处理出错了，请重试。",
        "I could not complete this request. Please try again.",
        "処理を完了できませんでした。もう一度お試しください。",
        "No pude completar la solicitud. Inténtalo de nuevo.",
    ),
}


def split_reply_request(query: str) -> tuple[str, ReplyLanguage | None]:
    """Return search text without explicit response-language instructions."""
    hits = [(match.start(), language) for language, pattern in _LANGUAGE_REQUESTS for match in pattern.finditer(query)]
    if not hits:
        return query, None
    for _, pattern in _LANGUAGE_REQUESTS:
        query = pattern.sub(" ", query)
    return query.strip(" 、，,。.!？?！"), max(hits)[1]


def reply_language_name(language: ReplyLanguage) -> str:
    return _LANGUAGE_NAMES[language]


def reply_text(key: str, language: ReplyLanguage, **values: str | int) -> str:
    index = ("zh", "en", "ja", "es").index(language)
    return _TEXT[key][index].format(**values)
