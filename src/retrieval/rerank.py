# src/retrieval/rerank.py
# Reordena los candidatos del dense search con un CrossEncoder.
# Reutiliza el bi-encoder y el cliente de Qdrant de filtered_search.py: no
# volvemos a instanciar nada para no cargar bge-small dos veces en memoria.
import pandas as pd
from sentence_transformers import CrossEncoder

from src.config import CROSS_ENCODER_MODEL, RERANKER_MAX_LENGTH, get_device
from src.retrieval.filtered_search import filtered_dense_search, known_companies

# max_length explícito: los chunks densos (~1800 chars) superan los 512 tokens
# de bge-reranker-base; sin esto el tokenizer decide el corte y puede tirar la
# línea relevante del estado financiero.
reranker = CrossEncoder(CROSS_ENCODER_MODEL, max_length=RERANKER_MAX_LENGTH, device=get_device())


def rerank(question, candidates, top_k=8):
    if not candidates:
        return []
    pairs = [(question, c.payload["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    qa = pd.read_json("data/raw/financebench/financebench_open_source.jsonl", lines=True)

    question = qa.iloc[0]["question"]
    candidatos = filtered_dense_search(question, known_companies, top_k=20)
    top = rerank(question, candidatos, top_k=15)

    for c, score in top:
        print(f"rerank_score={score:.3f} | {c.payload['doc_name']} | página {c.payload['page_num']}")
