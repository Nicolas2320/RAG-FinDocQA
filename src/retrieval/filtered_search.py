# Bi-encoder (bge-small-en-v1.5, usado en filtered_search.py): codifica la pregunta y cada chunk por separado, cada uno en su propio vector, y luego los compara con coseno. La ventaja enorme es que los vectores de los chunks se calculan una sola vez (en Day 3) y quedan indexados — comparar contra millones de chunks es solo aritmética de vectores, muy rápido. La desventaja: la pregunta y el chunk nunca "se ven" el uno al otro durante la codificación, así que la señal de relevancia es más burda.

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
    # Extrae el año fiscal a partir del texto de la pregunta, probando patrones
    # de mas a menos especifico y devolviendo el primero que encaje:
    #   1) "FY2018" -> 2018
    #   2) "FY18"   -> 2018
    #   3) cualquier año suelto de 4 digitos (1900-2099)
    m = re.search(r"FY\s?(\d{4})", question, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"FY\s?(\d{2})\b", question, re.IGNORECASE)
    if m:
        return 2000 + int(m.group(1))
    m = re.search(r"\b(19|20)\d{2}\b", question)
    return int(m.group(0)) if m else None


def extract_company(question, known_companies):
    # Busca cual de las empresas conocidas se menciona en la pregunta.
    # Recorre la lista de mas largo a mas corto para que un nombre compuesto
    # gane frente a uno mas corto contenido en el (evita falsos positivos).
    # Coincidencia por substring, sin distinguir mayusculas. None si ninguna.
    q_lower = question.lower()
    for company in sorted(known_companies, key=len, reverse=True):
        if company.lower() in q_lower:
            return company
    return None


def build_metadata_filter(question, known_companies):
    # Construye el filtro de metadata de Qdrant a partir de la empresa y el año
    # detectados en la pregunta. Solo añade condiciones para lo que se detecto;
    # con `must` todas las condiciones presentes deben cumplirse (AND).
    # Devuelve (filtro | None, company, year) — company/year sirven para loguear.
    company = extract_company(question, known_companies)
    year = extract_year(question)
    conditions = []
    if company:
        conditions.append(FieldCondition(key="company", match=MatchValue(value=company)))
    if year:
        conditions.append(FieldCondition(key="doc_period", match=MatchValue(value=year)))
    return (Filter(must=conditions) if conditions else None), company, year


def filtered_dense_search(question, known_companies, top_k=50):
    # Búsqueda densa (semantica) restringida por metadata:
    #   1) deriva el filtro empresa/año de la pregunta,
    #   2) vectoriza la pregunta con el mismo modelo de la ingesta (QUERY_PREFIX
    #      es el prefijo de "query" que pide bge; normalize -> coherente con coseno),
    #   3) pide a Qdrant los top_k chunks mas cercanos que ademas pasen el filtro.
    # Devuelve la lista de puntos (con .score y .payload) para pasar al reranker.
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

# if __name__ == "__main__":
#     qa = pd.read_json("data/raw/financebench/financebench_open_source.jsonl", lines=True)

#     for pregunta in [qa.iloc[0]["question"]]:
#         print(pregunta)
#         for r in filtered_dense_search(pregunta, known_companies, top_k=10):
#             print(f"  score={r.score:.3f} | {r.payload['doc_name']} | página {r.payload['page_num']}")