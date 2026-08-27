import os
import time
import torch
from sentence_transformers import SentenceTransformer

print("Núcleos de CPU disponibles:", os.cpu_count())
print("Threads que usa PyTorch internamente:", torch.get_num_threads())
print("¿Tienes GPU disponible (CUDA)?:", torch.cuda.is_available())
