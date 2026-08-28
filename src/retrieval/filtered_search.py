# src/retrieval/filtered_search.py
import os
import re

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_URL, QUERY_PREFIX, CROSS_ENCODER_MODEL

load_dotenv()

client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")


def extract_year(question):
    m = re.search(r"FY\s?(\d{4})", question, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"FY\s?(\d{2})\b", question, re.IGNORECASE)
    if m:
        return 2000 + int(m.group(1))
    m = re.search(r"\b(19|20)\d{2}\b", question)
    return int(m.group(0)) if m else None


def extract_company(question, known_companies):
    q_lower = question.lower()
    for company in sorted(known_companies, key=len, reverse=True):
        if company.lower() in q_lower:
            return company
    return None


def build_metadata_filter(question, known_companies):
    company = extract_company(question, known_companies)
    year = extract_year(question)
    conditions = []
    if company:
        conditions.append(FieldCondition(key="company", match=MatchValue(value=company)))
    if year:
        conditions.append(FieldCondition(key="doc_period", match=MatchValue(value=year)))
    return (Filter(must=conditions) if conditions else None), company, year


def filtered_dense_search(question, known_companies, top_k=20):
    query_filter, company, year = build_metadata_filter(question, known_companies)
    query_vector = model.encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    ).points
    print(f"[empresa={company!r}, año={year!r}, filtro_aplicado={query_filter is not None}]")
    return results

def rerank(question, candidates, top_k=5):
    pairs = [(question, c.payload["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

if __name__ == "__main__":
    meta = pd.read_json("data/raw/financebench/financebench_document_information.jsonl", lines=True)
    qa = pd.read_json("data/raw/financebench/financebench_open_source.jsonl", lines=True)
    known_companies = meta["company"].unique().tolist()

    for pregunta in [qa.iloc[0]["question"]]:
        print(pregunta)
        for r in filtered_dense_search(pregunta, known_companies, top_k=20):
            print(f"  score={r.score:.3f} | {r.payload['doc_name']} | página {r.payload['page_num']}")

    reranker = CrossEncoder(CROSS_ENCODER_MODEL, device="cuda")

    question = qa.iloc[0]["question"]
    candidatos = filtered_dense_search(question, known_companies, top_k=20)
    top5 = rerank(question, candidatos, top_k=20)

    for c, score in top5:
        print(f"rerank_score={score:.3f} | {c.payload['doc_name']} | página {c.payload['page_num']}")

