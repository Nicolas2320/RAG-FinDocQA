# Configuración compartida del sistema RAG.
# Fuente única de verdad para los valores que usan los scripts de embedding,
# retrieval y la API. Si cambias la colección o el modelo, cámbialo solo aquí.

COLLECTION_NAME = "financebench_chunks"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"

# bge-small-en-v1.5 espera este prefijo de instrucción SOLO en la query.
# Los pasajes se indexan sin prefijo (ver src/embedding/embedding_qdrant.py).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
