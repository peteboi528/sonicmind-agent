"""RagService —— 基于素材片段的检索 / 推荐 / 报告。

从 `AudioVisualAgent` 抽离的 RAG 能力：单素材证据检索（retrieve_evidence）、全库
证据检索（retrieve_library_evidence）、记忆驱动的片段推荐（recommend_with_memory）、
素材报告（generate_report），以及共享的 _require_segments（缺片段时触发 analyze）。

依赖通过构造注入（store/media/memory + analyze_media/list_assets 回调）；agent 侧
保留同名薄委托，外部 `agent.retrieve_evidence` / discover 注入的
`retrieve_library_evidence` / handlers 调用 / 测试均不受影响。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from app.media.pipeline import MediaPipeline
from app.memory import MemoryManager
from app.models import AgentAnswer, Asset, RagEvidence, Segment
from app.retrieval.vector_store import HybridRetriever
from app.storage import JsonStore

logger = logging.getLogger(__name__)


class RagService:
    def __init__(
        self,
        store: JsonStore,
        media: MediaPipeline,
        memory: MemoryManager,
        *,
        analyze_media: Callable[[str], tuple[Asset, list[Segment]]],
        list_assets: Callable[[], list[Asset]],
    ) -> None:
        self.store = store
        self.media = media
        self.memory = memory
        # 回调：_require_segments 缺片段时要走 agent.analyze_media（含缓存失效 + library
        # 同步），不能直接调 media.analyze_media 绕过；retrieve_library_evidence 需当前资产视图。
        self._analyze_media = analyze_media
        self._list_assets = list_assets

    def _require_segments(self, asset_id: str) -> list[Segment]:
        segments = self.media.get_segments(asset_id)
        if not segments:
            _, segments = self._analyze_media(asset_id)
        return segments

    def retrieve_evidence(self, asset_id: str, query: str, top_k: int = 5) -> list[RagEvidence]:
        segments = self._require_segments(asset_id)
        evidences = HybridRetriever(segments).search(query=query, top_k=top_k)
        asset = self.store.read_model("assets", asset_id, Asset)
        title = asset.title if asset else asset_id
        for evidence in evidences:
            evidence.metadata["asset_id"] = asset_id
            evidence.metadata["asset_title"] = title
        return evidences

    def retrieve_library_evidence(self, query: str, top_k: int = 5) -> list[RagEvidence]:
        ranked: list[RagEvidence] = []
        for asset in self._list_assets():
            if asset.status != "analyzed":
                continue
            # 全库搜索必须是只读操作。过去这里调用 retrieve_evidence，后者会在
            # segments 缺失时自动 analyze_media，导致一次普通歌曲/歌手搜索悄悄
            # 改写曲库指纹和 updated_at。未分析片段只是不参与 RAG，不应在查询时补写。
            segments = self.media.get_segments(asset.asset_id)
            if not segments:
                continue
            evidences = HybridRetriever(segments).search(query=query, top_k=min(3, top_k))
            for evidence in evidences:
                evidence.metadata["asset_id"] = asset.asset_id
                evidence.metadata["asset_title"] = asset.title
            ranked.extend(evidences)
        ranked.sort(key=lambda evidence: evidence.similarity, reverse=True)
        return ranked[:top_k]

    def recommend_with_memory(self, asset_id: str, user_id: str, goal: str, top_k: int = 3) -> AgentAnswer:
        memory = self.memory.get_memory(user_id)
        memory_query = self.memory.weighted_query(memory, include_artists=False)
        evidences = self.retrieve_evidence(asset_id, f"{goal} {memory_query}".strip(), top_k=max(top_k * 2, top_k))
        segment_map = {segment.segment_id: segment for segment in self._require_segments(asset_id)}
        segments: list[Segment] = []
        seen: set[str] = set()
        for evidence in evidences:
            if evidence.segment_id in seen:
                continue
            segment = segment_map.get(evidence.segment_id)
            if segment is not None:
                seen.add(evidence.segment_id)
                segments.append(segment)
        lines = [
            f"{index}. {segment.timestamp} - {segment.scene_summary}"
            for index, segment in enumerate(segments[:top_k], start=1)
        ]
        answer = "基于你的记忆和当前素材，我优先推荐这些片段：\n" + "\n".join(lines) if lines else "当前素材里没有足够明显的高匹配片段。"
        return AgentAnswer(
            answer=answer,
            evidences=evidences[:top_k],
            recommended_segments=segments[:top_k],
            agent_trace=[
                f"goal={goal}",
                f"memory_query={memory_query or 'none'}",
                f"evidence_chunks={len(evidences)}",
            ],
        )

    def generate_report(self, asset_id: str) -> dict[str, Any]:
        asset = self.store.read_model("assets", asset_id, Asset)
        if asset is None:
            raise ValueError(f"Unknown asset_id: {asset_id}")
        segments = self._require_segments(asset_id)
        evidences = self.retrieve_evidence(asset_id, "high energy climax mood genre summary", top_k=4)
        return {
            "asset": asset.model_dump(mode="json"),
            "summary": f"{asset.title} 已拆分为 {len(segments)} 个片段，可用于风格检索、推荐解释和相似内容分析。",
            "top_evidences": [evidence.model_dump(mode="json") for evidence in evidences],
            "fingerprint": {
                "genre": asset.genre,
                "mood": asset.mood,
                "tempo_bpm": asset.tempo_bpm,
                "energy_level": asset.energy_level,
            },
        }
