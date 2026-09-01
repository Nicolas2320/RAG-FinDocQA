import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag_langchain.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    QDRANT_URL,
    QUERY_PREFIX,
    get_device,
)

load_dotenv()

CHUNKS_PATH = Path("data/processed/chunks/financebench_chunks_langchain.jsonl")


class BGEEmbeddings(Embeddings):
    # Eel prefijo va SOLO en la query, nunca en los pasajes indexados. 
    def __init__(self):
        self._model = SentenceTransformer(EMBEDDING_MODEL, device=get_device())

    def embed_documents(self, texts):
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self._model.encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()


def build_index():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    print(f"Total de chunks: {len(chunks)}")

    documents = [
        Document(page_content=c["text"], metadata={k: v for k, v in c.items() if k != "text"})
        for c in chunks
    ]

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    QdrantVectorStore.from_documents(
        documents,
        embedding=BGEEmbeddings(),
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    print(f"Colección '{COLLECTION_NAME}' creada con {len(documents)} puntos.")


if __name__ == "__main__":
    build_index()