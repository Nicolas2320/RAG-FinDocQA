import asyncio
import os
import sys
import types
from pathlib import Path

# El script vive en RAG/notebooks pero usa rutas e imports relativos a la raíz.
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# ragas importa `langchain_community.chat_models.vertexai`, módulo eliminado en
# langchain-community >= 0.4. Creamos un stub (no lo usamos: el juez es OpenAI).
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertex_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertex_stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _vertex_stub

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import BaseRagasEmbeddings
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from src.config import EMBEDDING_MODEL, get_device
from src.generation.generate_answer import generate_answer
from src.retrieval.filtered_search import filtered_dense_search, known_companies
from src.retrieval.rerank import rerank

load_dotenv()

# LLM juez: mismo modelo de generación. Pasamos el ChatOpenAI de LangChain a
# evaluate(), que lo envuelve en LangchainLLMWrapper (BaseRagasLLM). No usamos
# llm_factory(client=...): en ragas 0.3.9 devuelve un InstructorLLM que las
# métricas clásicas de ragas.metrics no saben invocar (agenerate_prompt).
evaluator_llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
)


# Embeddings del juez: nuestro BGE local, con la interfaz clásica
# BaseRagasEmbeddings (embed_query / embed_documents) que espera AnswerRelevancy.
class LocalBGEEmbeddings(BaseRagasEmbeddings):
    # OJO: no llamar al atributo `model`. ragas envuelve esto en
    # LangchainEmbeddingsWrapper y su telemetría hace getattr(self, "model"),
    # que debe ser str|None; un SentenceTransformer ahí revienta con ValidationError.
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

metrics = [
    Faithfulness(),
    AnswerRelevancy(strictness=1),
    ContextPrecision(),
    ContextRecall(),
]

# Corremos retrieval + rerank + generación pregunta por pregunta, guardando
# el TEXTO crudo de los chunks (la API no lo expone, solo doc_name/page_num)
SAMPLE_SIZE = 10

qa = pd.read_json("data/raw/financebench/financebench_open_source.jsonl", lines=True)
sample = qa if SAMPLE_SIZE is None else qa.sample(n=SAMPLE_SIZE, random_state=42)

rows = []
for _, row in sample.iterrows():
    candidatos = filtered_dense_search(row["question"], known_companies, top_k=20)
    top = rerank(row["question"], candidatos, top_k=15)
    contexts = [c.payload["text"] for c, _score in top]
    chunks_para_llm = [
        {"doc_name": c.payload["doc_name"], "page_num": c.payload["page_num"], "text": c.payload["text"]}
        for c, _score in top
    ]
    answer = generate_answer(row["question"], chunks_para_llm)

    rows.append({
        "user_input": row["question"],
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": row["answer"],
    })
    print(f"[{row['financebench_id']}] listo")
    
eval_dataset = EvaluationDataset.from_list(rows)

results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)
df_results = results.to_pandas()
df_results.to_csv("data/processed/ragas_results.csv", index=False)

print("\n=== Promedios ===")
print(df_results[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())