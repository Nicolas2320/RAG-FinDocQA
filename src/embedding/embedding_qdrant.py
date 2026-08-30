import json
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_URL, get_device

load_dotenv() 

# Salida del paso de chunking: un JSON por linea con {chunk_id, doc_name, ..., text}.
CHUNKS_PATH = Path("data/processed/chunks/financebench_chunks.jsonl")

# Cliente de la base vectorial y modelo de embeddings (Se uso Docker)
client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(EMBEDDING_MODEL, device=get_device())
vector_size = model.get_embedding_dimension()  # dimension del vector -> config de la coleccion

# Reconstruye la coleccion desde cero, si ya existe, la borra para no mezclar
# los vectores viejos con los de esta corrida (ingesta idempotente).
if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)

# Crea la coleccion con la dimension del modelo y distancia coseno (similitud semantica).
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

chunks = [json.loads(line) for line in open(CHUNKS_PATH)]
print(f"Total de chunks: {len(chunks)}")

# Procesa los chunks en lotes: cada iteracion embebe BATCH_SIZE textos de una vez
# (mas eficiente que uno a uno) y los sube a Qdrant.
BATCH_SIZE = 128
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    # Vectoriza los textos del lote; normalize=True -> vectores unitarios, coherente
    # con la distancia coseno de la coleccion.
    vectors = model.encode([c["text"] for c in batch], normalize_embeddings=True)

    # Inserta/actualiza un punto por chunk: id correlativo global (i + j),
    # el vector, y el chunk completo como payload (metadata para filtrar y citar).
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=i + j, vector=v.tolist(), payload=chunk)
            for j, (chunk, v) in enumerate(zip(batch, vectors))
        ],
    )
    # Log de progreso cada 10 lotes.
    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  {i}/{len(chunks)}")