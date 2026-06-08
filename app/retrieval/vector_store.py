from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.models import Modality, RagEvidence, Segment


TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


@dataclass(frozen=True)
class SearchDocument:
    segment: Segment
    modality: Modality
    content: str


class HybridRetriever:
    """Small hybrid retriever for interview demos.

    It combines sparse cosine similarity over token counts with keyword overlap.
    This mirrors the idea behind production hybrid search without requiring a
    vector database service.
    """

    def __init__(self, segments: list[Segment]) -> None:
        self.documents = self._build_documents(segments)
        self.doc_vectors = [vectorize(doc.content) for doc in self.documents]

    def search(self, query: str, top_k: int = 5) -> list[RagEvidence]:
        query_vector = vectorize(query)
        query_terms = set(query_vector)
        ranked: list[tuple[float, SearchDocument]] = []
        for doc, vector in zip(self.documents, self.doc_vectors, strict=True):
            dense_score = cosine(query_vector, vector)
            keyword_score = keyword_overlap(query_terms, set(vector))
            score = (0.72 * dense_score) + (0.28 * keyword_score)
            if score > 0:
                ranked.append((score, doc))

        if not ranked:
            ranked = [(0.01, doc) for doc in self.documents]

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            RagEvidence(
                segment_id=doc.segment.segment_id,
                timestamp=doc.segment.timestamp,
                content=doc.content,
                modality=doc.modality,
                keyframe_path=doc.segment.keyframe_path,
                similarity=round(score, 4),
                metadata={
                    "visual_tags": doc.segment.visual_tags,
                    "audio_tags": doc.segment.audio_tags,
                    "start_seconds": doc.segment.start_seconds,
                    "end_seconds": doc.segment.end_seconds,
                },
            )
            for score, doc in ranked[:top_k]
        ]

    @staticmethod
    def _build_documents(segments: list[Segment]) -> list[SearchDocument]:
        docs: list[SearchDocument] = []
        for segment in segments:
            docs.append(SearchDocument(segment, Modality.TEXT, segment.transcript))
            docs.append(SearchDocument(segment, Modality.VISION, " ".join(segment.visual_tags)))
            docs.append(SearchDocument(segment, Modality.AUDIO, " ".join(segment.audio_tags)))
            docs.append(SearchDocument(segment, Modality.SUMMARY, segment.scene_summary))
        return docs


def vectorize(text: str) -> Counter[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return Counter(tokens)


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(weight * right.get(token, 0) for token, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def keyword_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)

