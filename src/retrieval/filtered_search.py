# src/retrieval/filtered_search.py
import re

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_URL, QUERY_PREFIX, get_device

load_dotenv()

DOC_INFO_PATH = "data/raw/financebench/financebench_document_information.jsonl"

client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(EMBEDDING_MODEL, device=get_device())

# Lista de empresas conocidas: se usa para detectar la empresa mencionada en la
# pregunta y construir el filtro de metadata. A nivel de módulo para que otros
# módulos (pipeline.py) puedan importarla.
known_companies = (
    pd.read_json(DOC_INFO_PATH, lines=True)["company"].unique().tolist()
)


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

if __name__ == "__main__":
    qa = pd.read_json("data/raw/financebench/financebench_open_source.jsonl", lines=True)

    for pregunta in [qa.iloc[0]["question"]]:
        print(pregunta)
        for r in filtered_dense_search(pregunta, known_companies, top_k=10):
            print(f"  score={r.score:.3f} | {r.payload['doc_name']} | página {r.payload['page_num']}")