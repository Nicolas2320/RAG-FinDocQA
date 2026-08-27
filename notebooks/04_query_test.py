from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Mismo destino y modelo que se usaron en src/embedding/embedding_qdrant.py
COLLECTION_NAME = "financebench_chunks"

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

info = client.count(collection_name=COLLECTION_NAME, exact=True)
print(f"Puntos en Qdrant: {info.count}")

query = "What is Adobe's operating cash flow ratio for fiscal year 2015?"
query_vector = model.encode(query, normalize_embeddings=True).tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5, # Top 5
)

for r in results.points:
    print(f"score={r.score:.3f} | {r.payload['doc_name']} | página {r.payload['page_num']}")
    print(r.payload["text"][:150])
    print("---")


    
