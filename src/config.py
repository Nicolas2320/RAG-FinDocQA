# Configuración compartida del sistema RAG.
# Fuente para los valores que usan los scripts de embedding,
# retrieval y la API.

COLLECTION_NAME = "financebench_chunks"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"

# bge-small-en-v1.5 espera este prefijo de instrucción SOLO en la query.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Los pasajes se indexan sin prefijo (ver src/embedding/embedding_qdrant.py).

# bge-reranker-base admite como máximo 512 tokens. Lo fijamos explícito para que
# el CrossEncoder trunque de forma determinista y no dependa del tokenizer.
RERANKER_MAX_LENGTH = 512


def get_device():
    # Usa GPU si hay CUDA disponible, si no, cae a CPU.
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
