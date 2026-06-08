"""Phase 3 测试：embedding 后端的可用/降级两条路径。

不依赖真实安装 sentence-transformers——用 monkeypatch 注入假模型验证
"可用走 dense 向量"路径，并验证未安装时检索优雅回退 TF cosine。
"""

import app.retrieval.embeddings as emb
from app.models import Segment
from app.retrieval.vector_store import HybridRetriever


def _seg(sid: str, transcript: str, summary: str) -> Segment:
    return Segment(
        segment_id=sid,
        asset_id="a1",
        start_seconds=0,
        end_seconds=30,
        transcript=transcript,
        scene_summary=summary,
    )


def test_unavailable_falls_back(monkeypatch):
    """embedding 不可用时，检索仍正常返回（走 TF cosine）。"""
    monkeypatch.setattr(emb, "_model", None)
    monkeypatch.setattr(emb, "_load_attempted", True)  # 模拟已尝试加载失败
    assert emb.embeddings_available() is False

    segments = [_seg("s1", "轻柔的钢琴旋律", "安静的夜晚场景")]
    results = HybridRetriever(segments).search("钢琴", top_k=3)
    assert results
    assert results[0].segment_id == "s1"


def test_cosine_normalized():
    assert emb.cosine_normalized([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert emb.cosine_normalized([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert emb.cosine_normalized([], [1.0]) == 0.0  # 长度不一致安全返回 0


def test_available_path_uses_dense(monkeypatch):
    """注入假模型，验证 embedding 可用路径被走到且检索正常。"""

    class FakeModel:
        def encode(self, texts, **kwargs):
            # 给"钢琴"相关文本高相似向量，其余正交
            out = []
            for t in texts:
                if "钢琴" in t or "piano" in t.lower():
                    out.append([1.0, 0.0])
                else:
                    out.append([0.0, 1.0])
            return out

    monkeypatch.setattr(emb, "_model", FakeModel())
    monkeypatch.setattr(emb, "_load_attempted", True)
    assert emb.embeddings_available() is True

    segments = [
        _seg("piano", "piano melody", "piano melody"),
        _seg("drum", "drum beat", "drum beat"),
    ]
    retriever = HybridRetriever(segments)
    assert retriever.doc_embeddings is not None  # 已预编码语义向量
    results = retriever.search("钢琴", top_k=2)
    # 钢琴 query 向量 [1,0] 应与 piano 文档最相似
    assert results[0].segment_id == "piano"


def test_encode_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(emb, "_model", None)
    monkeypatch.setattr(emb, "_load_attempted", True)
    assert emb.encode(["任意文本"]) is None
