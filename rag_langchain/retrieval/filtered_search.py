import re

import pandas as pd
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag_langchain.config import COLLECTION_NAME, QDRANT_URL
from rag_langchain.embedding.embedding_qdrant import BGEEmbeddings

load_dotenv()

DOC_INFO_PATH = "data/raw/financebench/financebench_document_information.jsonl"
known_companies = pd.read_json(DOC_INFO_PATH, lines=True)["company"].unique().tolist()

_vector_store = None


def get_vector_store():
    # Inicialización perezosa: a diferencia del manual,
    # nada de esto corre solo por hacer `import` -- el módulo se puede importar
    # en tests/CI sin Qdrant ni el modelo de embeddings cargados.
    global _vector_store
    if _vector_store is None:
        client = QdrantClient(url=QDRANT_URL)
        _vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=BGEEmbeddings())
    return _vector_store


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
    # Los campos van bajo "metadata." porque QdrantVectorStore anida ahí
    # la metadata del Document (a diferencia del payload plano del proyecto manual).
    company = extract_company(question, known_companies)
    year = extract_year(question)
    conditions = []
    if company:
        conditions.append(FieldCondition(key="metadata.company", match=MatchValue(value=company)))
    if year:
        conditions.append(FieldCondition(key="metadata.doc_period", match=MatchValue(value=year)))
    return (Filter(must=conditions) if conditions else None), company, year


def filtered_dense_search(question, known_companies, top_k=50):
    vector_store = get_vector_store()
    query_filter, company, year = build_metadata_filter(question, known_companies)

    results = vector_store.similarity_search(question, k=top_k, filter=query_filter)

    if not results and company and year:
        fallback_filter = Filter(must=[FieldCondition(key="metadata.company", match=MatchValue(value=company))])
        results = vector_store.similarity_search(question, k=top_k, filter=fallback_filter)
        print(f"[fallback aplicado: se descartó el filtro de año, solo empresa={company!r}]")

    print(f"[empresa={company!r}, año={year!r}, filtro_aplicado={query_filter is not None}]")
    return results  # lista de Document