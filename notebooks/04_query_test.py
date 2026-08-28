from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd

COLLECTION_NAME = "financebench_chunks"

# bge-small-en-v1.5 espera este prefijo de instruccion SOLO en la query.
# Los pasajes se indexan sin prefijo (ver src/embedding/embedding_qdrant.py).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

info = client.count(collection_name=COLLECTION_NAME, exact=True)
print(f"Puntos en Qdrant: {info.count}")

qa = pd.read_json("data/raw/financebench/financebench_open_source.jsonl", lines=True)
query = qa.iloc[103]["question"]
print(f"Pregunta: {query}")
query_vector = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5, # Top 5
)

# Solo usando el embeddings + carga a Qdrant
for r in results.points:
    print(f"score={r.score:.3f} | {r.payload['doc_name']} | página {r.payload['page_num']}")
    print(r.payload["text"][:150])
    print("---")

# Resultado no tan esperado, rankea chunks no esperados, los pdfs contienen estructura
# similar de diferentes empresas, usaremos busqueda hibrida con BM25 en /retrieval


    
