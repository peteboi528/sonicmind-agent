"""Phase 3：sentence-transformers embedding 后端。

可选依赖 + 优雅降级：未安装 sentence-transformers 或加载失败时，
`embeddings_available()` 返回 False，调用方回退到 TF cosine。
模型懒加载并模块级单例缓存——HybridRetriever 每次检索都 new 实例，
模型加载昂贵，绝不可每次重载。
"""

from __future__ import annotations

import logging
import threading

from app.config import settings

logger = logging.getLogger(__name__)

_model = None
_load_attempted = False
_lock = threading.Lock()


def _load_model():
    """懒加载并缓存模型。失败时返回 None 并记住，不重复尝试。"""
    global _model, _load_attempted
    if _load_attempted:
        return _model
    with _lock:
        if _load_attempted:
            return _model
        _load_attempted = True
        if not settings.enable_embeddings:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(settings.embedding_model)
        except Exception:
            logger.debug("Embedding model load failed; falling back to sparse retrieval", exc_info=True)
            _model = None
        return _model


def embeddings_available() -> bool:
    """检索层据此决定走 dense 向量还是回退 TF cosine。"""
    return _load_model() is not None


def encode(texts: list[str]) -> list[list[float]] | None:
    """批量编码为归一化向量；不可用时返回 None。"""
    model = _load_model()
    if model is None:
        return None
    try:
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vectors]
    except Exception:
        logger.debug("Embedding encode failed; falling back to sparse retrieval", exc_info=True)
        return None


def cosine_normalized(a: list[float], b: list[float]) -> float:
    """归一化向量的余弦相似度即点积。长度不一致时安全返回 0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=False))


def semantic_scores(query: str, candidate_texts: list[str]) -> list[float] | None:
    """三锚精排的语义锚：query 与每个候选文本的语义相似度，归一化到 [0,1]。

    embedding 不可用时返回 None，调用方回退到 TF cosine。
    模型输出已是归一化向量，cosine ∈ [-1,1]，这里映射到 [0,1]：(x+1)/2。
    """
    if not candidate_texts:
        return []
    vectors = encode([query, *candidate_texts])
    if vectors is None:
        return None
    query_vec = vectors[0]
    return [(cosine_normalized(query_vec, vec) + 1.0) / 2.0 for vec in vectors[1:]]


def _reset_for_test() -> None:
    """仅供测试：清空单例缓存以便重新评估开关。"""
    global _model, _load_attempted
    _model = None
    _load_attempted = False
