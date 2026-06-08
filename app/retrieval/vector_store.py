from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.models import Modality, RagEvidence, Segment
from app.retrieval import embeddings


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

    Phase 3\uff1a\u5f53 sentence-transformers \u53ef\u7528\u65f6\uff08settings.enable_embeddings \u4e14\u4f9d\u8d56\u5df2\u88c5\uff09\uff0c
    dense \u5206\u91cf\u6539\u7528\u8bed\u4e49\u5411\u91cf\u7684\u4f59\u5f26\u76f8\u4f3c\u5ea6\uff0c\u8de8\u8bed\u8a00/\u8fd1\u4e49\u5339\u914d\u663e\u8457\u4f18\u4e8e TF cosine\uff1b
    \u5426\u5219\u81ea\u52a8\u56de\u9000\u5230\u539f token-count cosine\u3002keyword overlap \u5206\u91cf\u59cb\u7ec8\u4fdd\u7559\u3002
    """

    def __init__(self, segments: list[Segment]) -> None:
        self.documents = self._build_documents(segments)
        self.doc_vectors = [vectorize(doc.content) for doc in self.documents]
        # \u5c1d\u8bd5\u9884\u7f16\u7801\u8bed\u4e49\u5411\u91cf\uff1b\u4e0d\u53ef\u7528\u65f6 self.doc_embeddings \u4e3a None\uff0csearch \u8d70 TF cosine\u3002
        self.doc_embeddings: list[list[float]] | None = None
        if embeddings.embeddings_available() and self.documents:
            self.doc_embeddings = embeddings.encode([doc.content for doc in self.documents])

    def search(self, query: str, top_k: int = 5) -> list[RagEvidence]:
        query_vector = vectorize(query)
        query_terms = set(query_vector)

        # \u8bed\u4e49\u5411\u91cf\u53ef\u7528\u65f6\u7f16\u7801 query\uff1b\u4efb\u4e00\u6b65\u5931\u8d25\u5219\u8be5\u8f6e\u56de\u9000 TF cosine\u3002
        query_embedding: list[float] | None = None
        if self.doc_embeddings is not None:
            encoded = embeddings.encode([query])
            query_embedding = encoded[0] if encoded else None

        ranked: list[tuple[float, SearchDocument]] = []
        for idx, (doc, vector) in enumerate(zip(self.documents, self.doc_vectors, strict=True)):
            if query_embedding is not None and self.doc_embeddings is not None:
                dense_score = embeddings.cosine_normalized(query_embedding, self.doc_embeddings[idx])
            else:
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

