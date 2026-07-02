from __future__ import annotations

import re
from typing import Any

from app.knowledge import lookup_metadata, resolve_music_entities, search_reviews


def run_music_fact_check(
    *,
    agent: Any,
    query: str,
    claims_text: str | None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_text = (claims_text or query or "").strip()
    entities = resolve_music_entities(query or raw_text, "review_summary", plan or {})
    metadata_payload = lookup_metadata(agent, entities, None)
    review_payload = search_reviews(entities, None)
    citations = [
        *[dict(item) for item in metadata_payload.get("citations") or []],
        *[dict(item) for item in review_payload.get("citations") or []],
    ]
    evidence_text = "\n".join(
        [
            *[str(item.get("summary", "") or "") for item in metadata_payload.get("metadata") or []],
            *[str(item.get("excerpt", "") or "") for item in citations],
            *[str(item.get("title", "") or "") for item in citations],
        ]
    )
    claims = _extract_claims(raw_text, entities)
    verified_claims: list[dict[str, Any]] = []
    uncertain_claims: list[dict[str, Any]] = []
    for claim in claims:
        evaluated = _evaluate_claim(claim, evidence_text, entities)
        if evaluated["status"] == "verified":
            verified_claims.append(evaluated)
        else:
            uncertain_claims.append(evaluated)
    return {
        "type": "music_fact_check",
        "claims": claims,
        "verified_claims": verified_claims,
        "uncertain_claims": uncertain_claims,
        "citations": citations[:8],
    }


def _extract_claims(text: str, entities: list[Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    if not text and entities:
        for entity in entities[:1]:
            claims.append({"kind": "entity_exists", "text": f"{entity.name} 是一个真实的音乐实体"})
        return claims
    for year in re.findall(r"\b(?:19|20)\d{2}\b", text):
        claims.append({"kind": "year", "text": text, "value": year})
    if "专辑" in text:
        claims.append({"kind": "entity_type", "text": text, "value": "album"})
    if any(token in text for token in ("歌手", "艺人", "乐队")):
        claims.append({"kind": "entity_type", "text": text, "value": "artist"})
    if "的" in text and entities:
        for entity in entities:
            if entity.artist and entity.name in text:
                claims.append({"kind": "artist_relation", "text": f"{entity.name} 属于 {entity.artist}", "value": entity.artist})
    if not claims:
        for entity in entities[:2]:
            claims.append({"kind": "entity_exists", "text": f"{entity.name} 是一个真实的{entity.type}", "value": entity.type})
            if entity.artist:
                claims.append({"kind": "artist_relation", "text": f"{entity.name} 属于 {entity.artist}", "value": entity.artist})
    deduped: list[dict[str, Any]] = []
    seen = set()
    for claim in claims:
        key = (claim.get("kind"), claim.get("text"), claim.get("value"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped


def _evaluate_claim(claim: dict[str, Any], evidence_text: str, entities: list[Any]) -> dict[str, Any]:
    kind = claim.get("kind")
    if kind == "year":
        year = str(claim.get("value") or "")
        status = "verified" if year and year in evidence_text else "uncertain"
        rationale = "资料摘要里出现相同年份。" if status == "verified" else "当前资料里没有足够年份证据。"
    elif kind == "entity_type":
        expected = str(claim.get("value") or "")
        status = "verified" if any(getattr(entity, "type", "") == expected for entity in entities) else "uncertain"
        rationale = "实体解析结果与陈述类型一致。" if status == "verified" else "实体类型没有被稳定解析到。"
    elif kind == "artist_relation":
        artist = str(claim.get("value") or "")
        status = "verified" if artist and any(artist == getattr(entity, "artist", "") for entity in entities) else "uncertain"
        rationale = "实体解析出的 artist 与陈述一致。" if status == "verified" else "当前资料不足以确认归属艺人。"
    else:
        status = "verified" if entities else "uncertain"
        rationale = "已解析到对应音乐实体。" if status == "verified" else "没有稳定解析到对应实体。"
    return {**claim, "status": status, "rationale": rationale}
