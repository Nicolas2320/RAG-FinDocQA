from rag_langchain.generation.generate_answer import generate_answer
from rag_langchain.retrieval.filtered_search import filtered_dense_search, known_companies
from rag_langchain.retrieval.rerank import rerank

DENSE_TOP_K_DEFAULT = 50
FINAL_TOP_K_DEFAULT = 10


def answer_question(question, dense_top_k=DENSE_TOP_K_DEFAULT, final_top_k=FINAL_TOP_K_DEFAULT):
    candidatos = filtered_dense_search(question, known_companies, top_k=dense_top_k)
    top = rerank(question, candidatos, top_k=final_top_k)
    docs = [doc for doc, _score in top]
    answer = generate_answer(question, docs)
    sources = [
        {"doc_name": doc.metadata["doc_name"], "page_num": doc.metadata["page_num"], "rerank_score": float(score)}
        for doc, score in top
    ]
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import sys
    pregunta = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Escribe tu pregunta: ")
    resultado = answer_question(pregunta)
    print("\n--- Respuesta ---")
    print(resultado["answer"])
    print("\n--- Fuentes ---")
    for s in resultado["sources"]:
        print(f"  {s['doc_name']} | página {s['page_num']} | score={s['rerank_score']:.3f}")