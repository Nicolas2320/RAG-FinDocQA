from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from rag_langchain.config import CROSS_ENCODER_MODEL, RERANKER_MAX_LENGTH, get_device

_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = HuggingFaceCrossEncoder(
            model_name=CROSS_ENCODER_MODEL,
            model_kwargs={"device": get_device(), "max_length": RERANKER_MAX_LENGTH},
        )
    return _cross_encoder


def rerank(question, candidates, top_k=10):
    # Segunda pasada con cross-encoder, igual que src/retrieval/rerank.py.
    if not candidates:
        return []
    cross_encoder = get_cross_encoder()
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = cross_encoder.score(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]  # [(Document, score), ...]