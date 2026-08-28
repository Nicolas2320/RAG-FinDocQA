import os
import torch

print("Núcleos de CPU disponibles:", os.cpu_count())
print("Threads que usa PyTorch internamente:", torch.get_num_threads())
print("¿Tienes GPU disponible (CUDA)?:", torch.cuda.is_available())
