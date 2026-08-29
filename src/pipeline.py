from src.retrieval.filtered_search import filtered_dense_search, known_companies
from src.retrieval.rerank import rerank
from src.generation.generate_answer import generate_answer

def answer_question(question, dense_top_k=20, final_top_k=15):
    candidatos = filtered_dense_search(question, known_companies, top_k=dense_top_k)
    top = rerank(question, candidatos, top_k=final_top_k)
    chunks_para_llm = [
        {"doc_name": c.payload["doc_name"], "page_num": c.payload["page_num"], "text": c.payload["text"]}
        for c, _score in top
    ]
    return generate_answer(question, chunks_para_llm)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pregunta = " ".join(sys.argv[1:])
    else:
        pregunta = input("Escribe tu pregunta: ")

    respuesta = answer_question(pregunta)
    print("\n--- Respuesta ---")
    print(respuesta)