# API HTTP que expone el pipeline RAG (src/pipeline.py) como servicio.
# Se levanta con: uvicorn src.api.main:app
from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import DENSE_TOP_K_DEFAULT, FINAL_TOP_K_DEFAULT, answer_question

app = FastAPI(title="FinDocQA API")


# Esquema del body de la petición a /query. Los top_k toman como default los
# mismos valores que src/pipeline.py, así que el cliente solo necesita mandar {"question": "..."}.
class QueryRequest(BaseModel):
    question: str
    dense_top_k: int = DENSE_TOP_K_DEFAULT 
    final_top_k: int = FINAL_TOP_K_DEFAULT


# Un elemento de la lista de fuentes que acompaña a la respuesta.
class SourceItem(BaseModel):
    doc_name: str
    page_num: int
    rerank_score: float


# Esquema de la respuesta. Al declararlo como response_model, FastAPI filtra y
# valida la salida del pipeline contra esta forma y lo documenta en /docs.
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


# Endpoint de liveness: sirve para healthchecks (Docker, load balancer, etc.).
@app.get("/health")
def health():
    return {"status": "ok"}


# Endpoint principal: recibe la pregunta y delega en el pipeline RAG completo
# (retrieval -> rerank -> generación). Devuelve {answer, sources}.
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    return answer_question(request.question, request.dense_top_k, request.final_top_k)