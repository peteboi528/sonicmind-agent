from __future__ import annotations

from app.models import Asset, Segment, SimilarAssetResult, SimilarSegmentResult
from app.retrieval.vector_store import HybridRetriever
from app.storage import JsonStore


class AssetSimilarity:
    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def find_similar_assets(self, asset_id: str, top_k: int = 5) -> list[SimilarAssetResult]:
        target = self.store.read_model("assets", asset_id, Asset)
        if target is None:
            raise ValueError(f"Unknown asset_id: {asset_id}")
        target_tags = set(target.tags_fingerprint or [])
        if not target_tags:
            return []

        all_keys = self.store.list_keys("assets")
        scored: list[SimilarAssetResult] = []
        for key in all_keys:
            if key == asset_id:
                continue
            other = self.store.read_model("assets", key, Asset)
            if other is None:
                continue
            other_tags = set(other.tags_fingerprint or [])
            if not other_tags:
                continue
            shared = target_tags & other_tags
            union = target_tags | other_tags
            jaccard = len(shared) / len(union)
            if jaccard > 0:
                scored.append(SimilarAssetResult(
                    asset_id=other.asset_id,
                    title=other.title,
                    score=round(jaccard, 4),
                    shared_tags=sorted(shared),
                ))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def find_similar_segments(
        self, asset_id: str, segment_id: str, top_k: int = 5
    ) -> list[SimilarSegmentResult]:
        segments = self.store.read_models("segments", asset_id, Segment)
        if not segments:
            raise ValueError(f"No segments for asset_id: {asset_id}")

        target = None
        others: list[Segment] = []
        for seg in segments:
            if seg.segment_id == segment_id:
                target = seg
            else:
                others.append(seg)

        if target is None:
            raise ValueError(f"Unknown segment_id: {segment_id}")
        if not others:
            return []

        retriever = HybridRetriever(others)
        evidences = retriever.search(target.searchable_text(), top_k=top_k)

        seg_map = {s.segment_id: s for s in others}
        seen: set[str] = set()
        results: list[SimilarSegmentResult] = []
        for ev in evidences:
            if ev.segment_id in seen:
                continue
            seen.add(ev.segment_id)
            seg = seg_map.get(ev.segment_id)
            if seg:
                results.append(SimilarSegmentResult(
                    segment=seg,
                    score=ev.similarity,
                    matching_modalities=[ev.modality.value],
                ))
        return results[:top_k]
