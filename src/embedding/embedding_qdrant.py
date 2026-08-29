import json
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_URL, get_device

load_dotenv()

CHUNKS_PATH = Path("data/processed/chunks/financebench_chunks.jsonl")

client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(EMBEDDING_MODEL, device=get_device())
vector_size = model.get_embedding_dimension()

if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

chunks = [json.loads(line) for line in open(CHUNKS_PATH)]
print(f"Total de chunks: {len(chunks)}")

BATCH_SIZE = 128
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    vectors = model.encode([c["text"] for c in batch], normalize_embeddings=True)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=i + j, vector=v.tolist(), payload=chunk)
            for j, (chunk, v) in enumerate(zip(batch, vectors))
        ],
    )
    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  {i}/{len(chunks)}")

print("Listo.")