from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import answer_question

app = FastAPI(title="FinDocQA API")


class QueryRequest(BaseModel):
    question: str
    dense_top_k: int = 20
    final_top_k: int = 15


class SourceItem(BaseModel):
    doc_name: str
    page_num: int
    rerank_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    return answer_question(request.question, request.dense_top_k, request.final_top_k)