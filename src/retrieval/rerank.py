# Cross-encoder (bge-reranker-base, usado aquí): mete la pregunta y el chunk juntos como un solo input ((question, text)) en una sola pasada por el transformer, permitiendo que cada palabra de la pregunta atienda directamente a cada palabra del chunk. Esto da un score de relevancia mucho más preciso, pero es caro: no se puede precalcular (tienes que correr el modelo en el momento de cada pregunta) ni escala a millones de documentos.

from sentence_transformers import CrossEncoder

from src.config import CROSS_ENCODER_MODEL, RERANKER_MAX_LENGTH, get_device

# from src.retrieval.filtered_search import filtered_dense_search, known_companies

reranker = CrossEncoder(CROSS_ENCODER_MODEL, max_length=RERANKER_MAX_LENGTH, device=get_device())


def rerank(question, candidates, top_k=10):
    # Segunda pasada sobre los candidatos del dense search. El CrossEncoder lee
    # la pregunta y el chunk juntos (no vectores por separado), asi que puntua
    # la relevancia con mucha mas precision que la similitud coseno inicial.
    if not candidates:
        return []
    # Un par (pregunta, texto_chunk) por candidato -> el modelo da un score cada uno.
    pairs = [(question, c.payload["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    # Reordena de mayor a menor score y devuelve los top_k como (candidato, score).
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# if __name__ == "__main__":
#     qa = pd.read_json("data/raw/financebench/financebench_open_source.jsonl", lines=True)

#     question = qa.iloc[0]["question"]
#     candidatos = filtered_dense_search(question, known_companies, top_k=20)
#     top = rerank(question, candidatos, top_k=15)

#     for c, score in top:
#         print(f"rerank_score={score:.3f} | {c.payload['doc_name']} | página {c.payload['page_num']}")
