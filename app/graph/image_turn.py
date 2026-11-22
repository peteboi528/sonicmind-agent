"""LangGraph branch for screenshot evidence, corrections, and verified recommendations."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.answer import song_card, track_ref_from_card
from app.config import settings
from app.graph.execution import _tool_outcome
from app.graph.state import AgentState
from app.models import AgentAnswer, MusicEntity, StreamEvent, TrackRef
from app.services.image_music import (
    ImageMusicSession,
    artist_credits,
    matching_candidates,
    normalized,
    plan_image_turn,
    recommendation_requested,
    same_track,
    supplied_image_title,
    verified_catalog_ref,
)
from app.services.tools import tool_runtime
from app.tools.contracts import ToolCall, ToolContext

if TYPE_CHECKING:
    from app.agent import AudioVisualAgent


_TEXT = {
    "reading": ("正在读取音乐图片…", "Reading the music image…", "画像を読み取っています…", "Leyendo la imagen…"),
    "search": ("正在核实曲目", "Checking tracks", "楽曲を確認しています", "Verificando canciones"),
    "pick": (
        "请选择你想听的艺人。",
        "Choose the artist you want to hear.",
        "聴きたいアーティストを選んでください。",
        "Elige el artista que quieres escuchar.",
    ),
    "unknown": (
        "图片里没有核实到音乐实体。请补充歌名和艺人，或重新上传一张文字清晰的截图。",
        "I couldn't verify a music entity from the image. Enter the title and artist, or upload a clearer screenshot.",
        "画像から楽曲やアーティストを確認できませんでした。曲名とアーティスト名を入力するか、鮮明な画像を送ってください。",
        "No pude verificar la música de la imagen. Escribe el título y el artista o envía una captura más clara.",
    ),
    "task": (
        "可以基于这张图片推荐截图里艺人的歌曲。请告诉我想听谁、要几首。",
        "I can recommend songs by the artists in the image. Tell me which artist and how many.",
        "画像のアーティストの曲をおすすめできます。アーティスト名と曲数を教えてください。",
        "Puedo recomendar música de los artistas de la imagen. Indica quién y cuántas canciones.",
    ),
    "found": (
        "为你找到 {n} 首由 {artist} 参与的曲目。",
        "I found {n} verified tracks credited to {artist}.",
        "{artist} が参加した確認済みの曲が {n} 曲見つかりました。",
        "Encontré {n} canciones verificadas con crédito a {artist}.",
    ),
    "short": (
        "你要 {count} 首，目前核实到 {n} 首。",
        "You asked for {count}; {n} qualified.",
        "{count} 曲のご希望に対し、条件を満たすのは {n} 曲です。",
        "Pediste {count}; {n} cumplen las condiciones.",
    ),
    "zero": (
        "这轮需要 0 首，已跳过检索。",
        "No tracks requested.",
        "指定が 0 曲なので検索を省きました。",
        "No se solicitaron canciones.",
    ),
    "image": (
        "已排除截图中的歌曲。",
        "I excluded the song shown in the screenshot.",
        "画像に写っている曲は除外しました。",
        "Excluí la canción de la captura.",
    ),
    "image_unverified": (
        "截图曲名尚未核实，请检查文字或提供准确歌名。",
        "The screenshot title is still unverified. Check the text or provide the exact title.",
        "画像の曲名は未確認です。表示を確認するか、正しい曲名を入力してください。",
        "El título de la captura sigue sin verificar. Revisa el texto o indica el título exacto.",
    ),
    "needs_title": (
        "已辨认出艺人。要排除截图里的歌，请回复「截图曲名：歌名」或上传更清晰的截图。",
        "I identified the artist. To exclude the pictured song, reply with 'track: title' or send a clearer image.",
        "アーティストは確認できました。画像の曲を除外するため「画像の曲名：曲名」と入力してください。",
        "Identifiqué al artista. Para excluir la canción de la imagen, responde «track: título».",
    ),
    "saved_unavailable": (
        "读取收藏记录超时，本轮暂停推荐。",
        "Reading saved music timed out, so no recommendations were made.",
        "保存済み楽曲を読み込めなかったため、おすすめを中止しました。",
        "Se agotó el tiempo para leer la música guardada.",
    ),
    "saved": (
        "收藏过滤覆盖 SonicMind 已保存歌单和已收藏专辑。",
        "The saved filter covers SonicMind playlists and saved albums.",
        "保存済みの SonicMind プレイリストとアルバムを除外しました。",
        "El filtro incluye las listas y los álbumes guardados en SonicMind.",
    ),
    "degraded": (
        "在线目录暂时不可用或超时，已使用核实到的结果。",
        "The online catalog was unavailable or timed out; showing the verified results available.",
        "オンラインカタログに接続できなかったため、確認できた結果を表示します。",
        "El catálogo no respondió a tiempo o no está disponible. Muestro los resultados verificados.",
    ),
}


def _text(key: str, language: str, **values: Any) -> str:
    return _TEXT[key][("zh", "en", "ja", "es").index(language)].format(**values)


def _ref(card: dict[str, Any]) -> TrackRef:
    ref = track_ref_from_card(card)
    ref.cover_url = card.get("cover_url") or ""
    return ref


def _dedupe(tracks: list[TrackRef]) -> list[TrackRef]:
    result: list[TrackRef] = []
    for track in tracks:
        if verified_catalog_ref(track) and not any(same_track(track, previous) for previous in result):
            result.append(track)
    return result


def _resolved_image(observed: dict, tracks: list[TrackRef]) -> ImageMusicSession:
    title = str(observed.get("track") or "").strip()
    visible = str(observed.get("artist") or "")
    stated = [*artist_credits(visible), *(observed.get("collaborators") or [])]
    requested_keys = {normalized(artist) for artist in stated if artist}
    lines = str(observed.get("raw_text") or "").splitlines()
    raw_lines = {normalized(line) for line in lines if line.strip()}
    credit_lines = [[normalized(name) for name in artist_credits(line)] for line in lines]
    raw_credits = raw_lines | {name for credits in credit_lines for name in credits}
    matches: list[TrackRef] = []
    pictured_primary: set[str] = set()
    matched_artists: dict[str, MusicEntity] = {}
    for track in tracks:
        if not verified_catalog_ref(track):
            continue
        credits = artist_credits(track.artist)
        keys = {normalized(name) for name in credits}
        exact_song = bool(title and normalized(track.title) == normalized(title) and requested_keys <= keys)
        ocr_song = not title and normalized(track.title) in raw_lines and bool(keys & raw_credits)
        if exact_song or ocr_song:
            matches.append(track)
            if ocr_song:
                pictured_primary.update(
                    line[0] for line in credit_lines if len(set(line)) > 1 and set(line) <= keys
                )
        for name in credits:
            key = normalized(name)
            if (exact_song or ocr_song or key in requested_keys or key in raw_credits) and key not in matched_artists:
                matched_artists[key] = MusicEntity(
                    name=name,
                    source=track.source,
                    ambiguity="resolved",
                    confidence=1,
                    query_origin="music_image_catalog_credit",
                )
    # Catalog versions can reverse the same credits. Prefer the order shown
    # in the image when a verified recording corroborates the credit line.
    primary = normalized(stated[0]) if stated else ""
    if primary not in matched_artists:
        primary_keys = pictured_primary or {normalized(artist_credits(track.artist)[0]) for track in matches}
        primary = next(iter(primary_keys)) if len(primary_keys) == 1 else ""
    return ImageMusicSession(
        artists=list(matched_artists.values()),
        primary_artist=matched_artists[primary].name if primary in matched_artists else "",
        screenshot_tracks=_dedupe(matches),
        observed_title=title,
    )


async def run_image_turn(
    agent: AudioVisualAgent, state: AgentState, emit: Callable[[StreamEvent], None]
) -> AgentAnswer:
    user, thread = state["user_id"], state["dialogue_thread_id"]
    message = state["query"]
    new = state.get("image_observation") is not None
    session = ImageMusicSession.model_validate(state.get("image_session") or {})
    started = state["turn_started_at"]
    end = started + settings.turn_budget_seconds
    deadline = end - min(settings.turn_finalize_reserve_seconds, settings.turn_budget_seconds)
    context = ToolContext(
        run_id=state["run_id"],
        thread_id=thread,
        user_id=user,
        query=message,
        deadline_at=deadline,
        agent=agent,
    )
    outcomes: list[dict] = []
    planned: list[str] = []

    async def lookup(call: ToolCall) -> list[TrackRef]:
        planned.append(call.name)
        if time.monotonic() >= deadline:
            outcomes.append(
                {"tool": call.name, "status": "skipped", "metrics": {"deadline_skipped": True}, "card_count": 0}
            )
            return []
        emit(
            StreamEvent(type="tool_start", content=_text("search", session.plan.language), payload={"tool": call.name})
        )
        try:
            result = await tool_runtime.execute(call, context)
        except asyncio.CancelledError:
            outcomes.append({"tool": call.name, "status": "cancelled", "metrics": {}, "card_count": 0})
            raise
        outcome = _tool_outcome(call, result, 0)
        outcomes.append(outcome)
        emit(StreamEvent(type="tool_result", payload=outcome, content=result.status.value))
        return [_ref(card) for card in result.cards]

    def local(name: str) -> ToolCall:
        return ToolCall(name="image_library_lookup", arguments={"artist": name})

    def online(query: str, limit: int) -> ToolCall:
        return ToolCall(name="image_catalog_lookup", arguments={"query": query, "limit": limit})

    if new:
        observed = state["image_observation"] or {}
        # Run independent work only if local evidence cannot settle the image.
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                assets = await asyncio.to_thread(agent.list_assets)
        except TimeoutError:
            assets = []
        local_refs = [_ref(song_card(asset)) for asset in assets if asset.artist and asset.media_type == "audio"]
        session = _resolved_image(observed, local_refs)
        vision_title = observed.get("track") or ""
        vision_artist = observed.get("artist") or ""
        need_online = not session.artists or (vision_title and not session.screenshot_tracks)
        if need_online:
            ocr_lines = [
                line.strip()
                for line in (observed.get("raw_text") or "").splitlines()
                if 2 <= len(line.strip()) <= 90
                and not re.fullmatch(r"now ?playing|正在播放|再生中|(?:\d{1,2}:)?\d{1,2}:\d{2}", line.strip(), re.I)
            ]
            query = " ".join(filter(None, (vision_title, vision_artist or observed.get("album"))))
            if not query:
                query = " ".join(ocr_lines[:2])
            if query:
                checked = await lookup(online(query, 15))
                session = _resolved_image(observed, [*local_refs, *checked])
        if observed.get("method") == "ocr" and session.artists:
            emit(
                StreamEvent(
                    type="image_recognition",
                    payload={
                        "method": "catalog",
                        "track": session.screenshot_tracks[0].title if len(session.screenshot_tracks) == 1 else "",
                        "artist": session.primary_artist,
                    },
                )
            )
    else:
        title_correction = supplied_image_title(message)
        if title_correction:
            session.observed_title = title_correction
            session.screenshot_tracks = []
    plan, choices = plan_image_turn(message, session, followup=not new)
    same_target = normalized(plan.target) == normalized(session.plan.target)
    session.plan = plan
    session.pending_artists = choices if not plan.target else []
    language = plan.language
    selected: list[TrackRef] = []
    status = "ok"
    if not session.artists:
        status, answer_text = "unverified", _text("unknown", language)
    elif new and not recommendation_requested(message):
        status, answer_text = "needs_task", _text("task", language)
        choices = [entity.name for entity in session.artists]
    elif not plan.target:
        status, answer_text = "needs_artist", _text("pick", language)
        emit(StreamEvent(type="image_choices", content=answer_text, payload={"artists": choices}))
    elif plan.count == 0:
        answer_text = _text("zero", language)
    elif plan.image_exclusion_required and not (session.screenshot_tracks or session.observed_title):
        status, answer_text = "needs_title", _text("needs_title", language)
    else:
        # If we know the screenshot title but haven't verified its exact recording,
        # conservatively block that title for every potential screenshot credit.
        excluded = list(session.screenshot_tracks) if plan.exclude_image else []
        if plan.exclude_image and session.observed_title and not session.screenshot_tracks:
            excluded += [TrackRef(title=session.observed_title, artist=entity.name) for entity in session.artists]
        groups = None
        if plan.exclude_saved:
            try:
                async with asyncio.timeout(max(0, deadline - time.monotonic())):
                    groups = await asyncio.to_thread(
                        lambda: [*agent.list_playlists(user), *agent.list_saved_albums(user)]
                    )
            except TimeoutError:
                # Keep the hard filter if its source cannot be read before deadline.
                pass
            if groups is not None:
                excluded += [_ref(song_card(track)) for group in groups for track in group.tracks]

        def accept(candidates: list[TrackRef]) -> None:
            if plan.exclude_saved and groups is None:
                return
            batch = matching_candidates(candidates, plan.target, excluded, selected)[: plan.count - len(selected)]
            selected.extend(batch)
            if batch:
                emit(
                    StreamEvent(
                        type="candidates",
                        content=f"{len(batch)}",
                        payload={"cards": [song_card(track) for track in batch]},
                    )
                )

        cached = session.candidates if same_target or new else []
        accept(cached)
        fetched = list(cached)
        if len(selected) < plan.count and (not plan.exclude_saved or groups is not None):
            tasks = [
                asyncio.create_task(lookup(call))
                for call in (local(plan.target), online(plan.target, min(100, max(25, plan.count * 3))))
            ]
            try:
                for completed in asyncio.as_completed(tasks):
                    batch = await completed
                    fetched.extend(batch)
                    accept(batch)
                    if len(selected) >= plan.count:
                        break
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Drain tasks so their permits and trace spans are released.
                for result in await asyncio.gather(*tasks, return_exceptions=True):
                    if isinstance(result, list):
                        fetched.extend(result)
        session.candidates = _dedupe(fetched)[:100]
        answer_text = _text("found", language, n=len(selected), artist=plan.target)
        if len(selected) < plan.count:
            status = "partial" if selected else "empty"
            answer_text += " " + _text("short", language, n=len(selected), count=plan.count)
        if plan.exclude_image and session.screenshot_tracks:
            answer_text += " " + _text("image", language)
        elif plan.exclude_image and session.observed_title:
            answer_text += " " + _text("image_unverified", language)
        if plan.exclude_saved:
            answer_text += " " + _text("saved" if groups is not None else "saved_unavailable", language)

    failures = [outcome for outcome in outcomes if outcome["status"] in {"error", "skipped"}]
    if failures and len(selected) < plan.count:
        answer_text += " " + _text("degraded", language)
    elapsed = round(time.monotonic() - started, 3)
    agent.memory.save_image_dialogue(user, thread, session)
    return AgentAnswer(
        answer=answer_text,
        evidences=[],
        recommended_tracks=selected,
        fallback_reason="source_degraded" if failures else None,
        trace_summary={
            "intent": "image_recommend",
            "response_language": language,
            "strategy": "verified_image",
            "image_status": status,
            "image_vision_calls": state.get("image_vision_calls", 0),
            "image_reused": not new,
            "image_choices": choices if status in {"needs_artist", "needs_task"} else [],
            "tool_execution_state": "partial"
            if failures and selected
            else "error"
            if failures
            else "ok"
            if selected
            else "empty"
            if planned
            else "not_planned",
            "tools_planned": list(dict.fromkeys(planned)),
            "tools_executed": list(dict.fromkeys(o["tool"] for o in outcomes if o["status"] != "skipped")),
            "empty_results": [o["tool"] for o in outcomes if o["status"] == "empty"],
            "tool_errors": [o["tool"] for o in failures],
            "tool_outcomes": outcomes,
            "tool_candidate_count": sum(o["card_count"] for o in outcomes),
            "final_cards": len(selected),
            "sources": sorted({track.source for track in selected}),
            "latency_budget": {
                "elapsed_seconds": elapsed,
                "budget_seconds": settings.turn_budget_seconds,
                "timed_out_tools": [o["tool"] for o in outcomes if o["metrics"].get("timeout_phase")],
                "skipped_due_to_deadline": [o["tool"] for o in outcomes if o["metrics"].get("deadline_skipped")],
            },
        },
    )
