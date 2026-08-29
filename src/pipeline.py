from src.retrieval.filtered_search import filtered_dense_search, known_companies
from src.retrieval.rerank import rerank
from src.generation.generate_answer import generate_answer

def answer_question(question, dense_top_k=50, final_top_k=10):
    candidatos = filtered_dense_search(question, known_companies, top_k=dense_top_k)
    top = rerank(question, candidatos, top_k=final_top_k)
    chunks_para_llm = [
        {"doc_name": c.payload["doc_name"], "page_num": c.payload["page_num"], "text": c.payload["text"]}
        for c, _score in top
    ]
    answer = generate_answer(question, chunks_para_llm)
    sources = [
        {
            "doc_name": c.payload["doc_name"],
            "page_num": c.payload["page_num"],
            "rerank_score": float(score),
        }
        for c, score in top
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