"""Planning and evidence checks for recommendations based on a music image."""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

from app.answer import infer_count
from app.models import MusicEntity, TrackRef
from app.reply_language import ReplyLanguage, split_reply_request

_RECOMMEND = re.compile(
    r"推荐|推几|来[一二两三四五几\d]+首|recommend|suggest|give me|おすすめ|お勧め|勧めて|recomi[eé]nd|sugiere", re.I
)
_REFERENCE = re.compile(
    r"这(?:个|位|张|首|名)|截图|图[中里上]|同一(?:个|位)|"
    r"\b(?:this|that|same) (?:artist|singer|song|image|screenshot)\b|"
    r"この(?:歌手|アーティスト|画像|曲)|同じ(?:歌手|アーティスト)|"
    r"\b(?:este|esta|ese) (?:artista|cantante|captura)\b",
    re.I,
)
_CORRECTION = re.compile(
    r"我[是指]*说的?是|我指的?是|换成|改成|\bi meant\b|\bi mean\b|"
    r"\bswitch to\b|の方|のほう|に変えて|のこと|me refer[ií]a a",
    re.I,
)
_COARTIST = re.compile(
    r"合作(?:歌手|艺人|方)|客串|合唱|collaborat|featured|co[- ]?artist|guest artist|コラボ|共演|客演|colabor", re.I
)
_SAVED = re.compile(r"收藏|已保存|\bsaved\b|\bfavou?rites?\b|保存済|お気に入り|guardad[oa]|favorit", re.I)
_EXCLUDE = re.compile(
    r"别|不要|去掉|去除|排除|跳过|过滤|不[要含包括推]|exclude|skip|leave out|without|no repeat|"
    r"除外|抜き|外して|なし|いらない|除いて|入れない|sin |exclu",
    re.I,
)
_INCLUDE_SAVED = re.compile(r"收藏(?:过的)?也可以|包括收藏|include (?:my )?saved|保存済みも含め|incluye.*guardad", re.I)
_IMAGE_TITLE = re.compile(r"^(?:截图曲名|图片里的歌(?:名)?(?:是)?|track|画像の曲名)\s*[:：]\s*(.+)$", re.I)
_IMAGE_TRACK = re.compile(r"截图|图[中里上]|screenshot|画像|captura", re.I)
_PICK = re.compile(r"^(?:选(?:艺人|歌手)|artist|アーティスト)\s*[:：]\s*(.+)$", re.I)
_TOPIC = re.compile(r"^(?:换个话题|新话题|new topic|話題を変え|cambiemos de tema)", re.I)
_VERSION_SUFFIX = re.compile(
    r"\s*(?:\([^()]*\b(?:remix|remaster(?:ed)?|radio edit|live version)\b[^()]*\)|"
    r"\s[-–—]\s[^()]*\b(?:remix|remaster(?:ed)?|radio edit|live version)\b[^()]*)$",
    re.I,
)
_UNSUPPORTED = re.compile(r"歌词|翻译|演唱会|巡演|lyrics|translate|concert|tour dates|歌詞|翻訳|ライブ日程", re.I)


class ImageRecommendationPlan(BaseModel):
    target: str = ""
    count: int = Field(default=3, ge=0, le=100)
    exclude_image: bool = True
    image_exclusion_required: bool = False
    exclude_saved: bool = False
    language: ReplyLanguage = "zh"


class ImageMusicSession(BaseModel):
    artists: list[MusicEntity] = Field(default_factory=list)
    primary_artist: str = ""
    screenshot_tracks: list[TrackRef] = Field(default_factory=list)
    observed_title: str = ""
    plan: ImageRecommendationPlan = Field(default_factory=ImageRecommendationPlan)
    candidates: list[TrackRef] = Field(default_factory=list)
    pending_artists: list[str] = Field(default_factory=list)


def normalized(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def artist_credits(value: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"[、，,;；]|\s+(?:/|／|&|feat\.?|ft\.?|featuring|with|×)\s+", value, flags=re.I)
        if part.strip()
    ]


def _base_title(title: str) -> str:
    return normalized(_VERSION_SUFFIX.sub("", unicodedata.normalize("NFKC", title)))


def same_track(left: TrackRef, right: TrackRef, *, include_versions: bool = False) -> bool:
    if left.source_id and left.source == right.source and left.source_id == right.source_id:
        return True
    left_artists = {normalized(name) for name in artist_credits(left.artist)}
    right_artists = {normalized(name) for name in artist_credits(right.artist)}
    title_key = _base_title if include_versions else normalized
    return bool(title_key(left.title) == title_key(right.title) and left_artists & right_artists)


def _mentions(text: str, name: str) -> bool:
    key = normalized(name)
    if not key:
        return False
    haystack = normalized(text)
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", key):
        return key in haystack
    return f" {key} " in f" {haystack} "


def verified_catalog_ref(track: TrackRef) -> bool:
    return bool(track.source_id and track.title and track.artist and track.source in {"local", "netease"})


def matching_candidates(
    tracks: list[TrackRef], target: str, excluded: list[TrackRef], already_selected: list[TrackRef]
) -> list[TrackRef]:
    selected: list[TrackRef] = []
    target_key = normalized(target)
    for track in tracks:
        if not verified_catalog_ref(track):
            continue
        if target_key not in {normalized(artist) for artist in artist_credits(track.artist)}:
            continue
        if any(same_track(track, prior, include_versions=True) for prior in excluded):
            continue
        if any(same_track(track, prior) for prior in [*already_selected, *selected]):
            continue
        selected.append(track)
    # Favor primary credits within each result batch. Across sources, retain the
    # cards already displayed to the user.
    return sorted(selected, key=lambda track: normalized(artist_credits(track.artist)[0]) != target_key)


def image_followup(message: str, session: ImageMusicSession) -> bool:
    text, _ = split_reply_request(message)
    if _TOPIC.search(text) or _UNSUPPORTED.search(text):
        return False
    if _PICK.fullmatch(text) or _IMAGE_TITLE.fullmatch(text):
        return True
    if any(normalized(text) == normalized(entity.name) for entity in session.artists):
        return True
    if _SAVED.search(text) and (_EXCLUDE.search(text) or _INCLUDE_SAVED.search(text)):
        # A new explicit recommendation target takes precedence over a filter-only edit.
        if not _RECOMMEND.search(text) or _REFERENCE.search(text) or _COARTIST.search(text):
            return True
    if _CORRECTION.search(text):
        return bool(_COARTIST.search(text) or any(_mentions(text, entity.name) for entity in session.artists))
    return bool(_RECOMMEND.search(text) and (_REFERENCE.search(text) or _COARTIST.search(text)))


def image_topic_changed(message: str) -> bool:
    """Drop image context for explicit topic changes or independent recommendations."""
    text, _ = split_reply_request(message)
    return bool(_TOPIC.search(text) or (_RECOMMEND.search(text) and not _REFERENCE.search(text)))


def supplied_image_title(message: str) -> str:
    text, _ = split_reply_request(message)
    match = _IMAGE_TITLE.fullmatch(text)
    return match[1].strip()[:160] if match else ""


def recommendation_requested(message: str) -> bool:
    return bool(_RECOMMEND.search(message) and not _UNSUPPORTED.search(message))


def _reply_language(text: str, previous: ReplyLanguage) -> ReplyLanguage:
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\u3400-\u9fff]", text):
        return "zh"
    if re.search(r"\b(?:recomi[eé]nd\w*|canciones|colaborador|guardadas|por favor)\b", text, re.I):
        return "es"
    if re.search(r"\b(?:recommend|suggest|songs?|please|meant|exclude|skip|saved)\b", text, re.I):
        return "en"
    return previous


def plan_image_turn(
    message: str, session: ImageMusicSession, *, followup: bool
) -> tuple[ImageRecommendationPlan, list[str]]:
    text, language = split_reply_request(message)
    previous = session.plan
    plan = previous.model_copy(update={"language": language or _reply_language(text, previous.language)})
    count = infer_count(text, allow_approximate=False)
    english = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:songs?|tracks?)\b", text, re.I)
    if count is None and english:
        count = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten").index(
            english[1].lower()
        ) + 1
    if count is not None:
        plan.count = count
    if _INCLUDE_SAVED.search(text):
        plan.exclude_saved = False
    elif _SAVED.search(text) and _EXCLUDE.search(text):
        plan.exclude_saved = True

    if _EXCLUDE.search(text) and _IMAGE_TRACK.search(text):
        plan.image_exclusion_required = True
    artists = [entity.name for entity in session.artists]
    pick = _PICK.fullmatch(text)
    explicit = [
        name for name in artists if not _IMAGE_TITLE.fullmatch(text) and _mentions(pick[1] if pick else text, name)
    ]
    # A complete name wins over a shorter substring of it.
    explicit = [name for name in explicit if not any(name != other and _mentions(other, name) for other in explicit)]
    if pick or explicit:
        choices = explicit or artists
    elif _COARTIST.search(text):
        choices = [name for name in artists if name != session.primary_artist]
    elif followup and previous.target in artists:
        choices = [previous.target]
    elif followup and session.pending_artists:
        choices = session.pending_artists
    elif session.primary_artist:
        choices = [session.primary_artist]
    else:
        choices = artists
    plan.target = choices[0] if len(choices) == 1 else ""
    return plan, choices or artists
