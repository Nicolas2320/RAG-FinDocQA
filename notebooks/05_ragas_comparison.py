import asyncio
import os
import sys
import time
import types
from pathlib import Path

# Igual que 04_ragas_evaluation.py: el script vive en RAG/notebooks pero usa
# rutas e imports relativos a la raíz.
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Mismo stub que 04: ragas importa un módulo de langchain-community que ya no existe.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertex_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertex_stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _vertex_stub

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import BaseRagasEmbeddings
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, get_device

# --- Pipeline manual (src/) ---
from src.generation.generate_answer import generate_answer as generate_answer_manual
from src.retrieval.filtered_search import filtered_dense_search as filtered_dense_search_manual
from src.retrieval.filtered_search import known_companies as known_companies_manual
from src.retrieval.rerank import rerank as rerank_manual

# --- Pipeline LangChain (rag_langchain/) ---
from rag_langchain.generation.generate_answer import generate_answer as generate_answer_langchain
from rag_langchain.retrieval.filtered_search import filtered_dense_search as filtered_dense_search_langchain
from rag_langchain.retrieval.filtered_search import known_companies as known_companies_langchain
from rag_langchain.retrieval.rerank import rerank as rerank_langchain

load_dotenv()

# Mismo juez (LLM + embeddings) para las dos corridas, si el juez cambiara
# entre pipelines, cualquier diferencia en las métricas dejaría de ser
# atribuible al framework y pasaría a ser ruido del juez.
evaluator_llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
)


class LocalBGEEmbeddings(BaseRagasEmbeddings):
    def __init__(self):
        self._encoder = SentenceTransformer(EMBEDDING_MODEL, device=get_device())

    def embed_query(self, text):
        return self._encoder.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts):
        return self._encoder.encode(list(texts), normalize_embeddings=True).tolist()

    async def aembed_query(self, text):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, list(texts))


evaluator_embeddings = LocalBGEEmbeddings()

metrics = [Faithfulness(), AnswerRelevancy(strictness=1), ContextPrecision(), ContextRecall()]

# Misma muestra para ambos pipelines: mismo archivo + mismo random_state ->
# qa.sample(...) selecciona exactamente las mismas 10 preguntas las dos veces.
SAMPLE_SIZE = 10
qa = pd.read_json("data/raw/financebench/financebench_open_source.jsonl", lines=True)
sample = qa if SAMPLE_SIZE is None else qa.sample(n=SAMPLE_SIZE, random_state=42)


def run_manual(sample):
    rows, elapsed = [], []
    for _, row in sample.iterrows():
        t0 = time.perf_counter()
        candidatos = filtered_dense_search_manual(row["question"], known_companies_manual, top_k=20)
        top = rerank_manual(row["question"], candidatos, top_k=15)
        contexts = [c.payload["text"] for c, _score in top]
        chunks_para_llm = [
            {"doc_name": c.payload["doc_name"], "page_num": c.payload["page_num"], "text": c.payload["text"]}
            for c, _score in top
        ]
        answer = generate_answer_manual(row["question"], chunks_para_llm)
        elapsed.append(time.perf_counter() - t0)
        rows.append({
            "user_input": row["question"],
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": row["answer"],
        })
        print(f"[manual] [{row['financebench_id']}] listo")
    return rows, elapsed


def run_langchain(sample):
    rows, elapsed = [], []
    for _, row in sample.iterrows():
        t0 = time.perf_counter()
        candidatos = filtered_dense_search_langchain(row["question"], known_companies_langchain, top_k=20)
        top = rerank_langchain(row["question"], candidatos, top_k=15)
        docs = [doc for doc, _score in top]
        contexts = [doc.page_content for doc in docs]
        answer = generate_answer_langchain(row["question"], docs)
        elapsed.append(time.perf_counter() - t0)
        rows.append({
            "user_input": row["question"],
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": row["answer"],
        })
        print(f"[langchain] [{row['financebench_id']}] listo")
    return rows, elapsed


rows_manual, elapsed_manual = run_manual(sample)
rows_langchain, elapsed_langchain = run_langchain(sample)

results_manual = evaluate(
    dataset=EvaluationDataset.from_list(rows_manual),
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)
results_langchain = evaluate(
    dataset=EvaluationDataset.from_list(rows_langchain),
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

df_manual = results_manual.to_pandas()
df_manual["pipeline"] = "manual"
df_manual["elapsed_s"] = elapsed_manual

df_langchain = results_langchain.to_pandas()
df_langchain["pipeline"] = "langchain"
df_langchain["elapsed_s"] = elapsed_langchain

df_comparison = pd.concat([df_manual, df_langchain], ignore_index=True)
df_comparison.to_csv("data/processed/ragas_comparison.csv", index=False)

print("\n=== Promedios: manual vs. LangChain ===")
cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "elapsed_s"]
print(df_comparison.groupby("pipeline")[cols].mean())