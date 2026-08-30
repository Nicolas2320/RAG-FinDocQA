# Imagen base
FROM python:3.11-slim

# Copiamos el binario de uv desde su imagen oficial
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

WORKDIR /app

# Evita bytecode residual raro y usa copy en vez de symlinks (mejor para capas Docker)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# 1) Copiamos SOLO los manifiestos de dependencias primero.
#    Mientras no cambiemos pyproject.toml/uv.lock, Docker reusa esta capa
#    en cache y no vuelve a bajar/instalar todo cada vez que cambias un .py.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) Copiamos el código fuente.
COPY src/ src/

# 3) Instalamos el propio proyecto
RUN uv sync --frozen --no-dev

# El entorno virtual que crea uv queda en /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Documentan qué puertos usa la imagen
EXPOSE 8000 8501

# Levanta la API. La UI lo sobreescribe en docker-compose.yml.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]