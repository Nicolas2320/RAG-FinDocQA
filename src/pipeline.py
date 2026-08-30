# Orquestador del RAG de punta a punta: encadena las tres etapas
# (retrieval denso -> rerank -> generación) que viven en sus propios módulos.
from src.retrieval.filtered_search import filtered_dense_search, known_companies
from src.retrieval.rerank import rerank
from src.generation.generate_answer import generate_answer

# Defaults del pipeline. Se exponen como constantes para que otros módulos
# (p. ej. la API en src/api/main.py) los reutilicen en vez de duplicar los valores.
DENSE_TOP_K_DEFAULT = 50   # candidatos que trae el retrieval denso
FINAL_TOP_K_DEFAULT = 10   # cuántos deja el reranker para el LLM

def answer_question(question, dense_top_k=DENSE_TOP_K_DEFAULT, final_top_k=FINAL_TOP_K_DEFAULT):
    # 1) Retrieval denso con filtro de metadata: trae dense_top_k candidatos amplios.
    candidatos = filtered_dense_search(question, known_companies, top_k=dense_top_k)
    # 2) Rerank con cross-encoder: se queda con los final_top_k mas relevantes.
    top = rerank(question, candidatos, top_k=final_top_k)
    # 3) Aplana el payload de Qdrant a los campos que necesita el prompt del LLM.
    chunks_para_llm = [
        {"doc_name": c.payload["doc_name"], "page_num": c.payload["page_num"], "text": c.payload["text"]}
        for c, _score in top
    ]
    # 4) Generación anclada en esos chunks.
    answer = generate_answer(question, chunks_para_llm)
    # 5) Fuentes: mismos chunks con su score de rerank, para mostrar/depurar de dónde salió la respuesta.
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