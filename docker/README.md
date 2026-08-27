# Docker — servicios locales

Infraestructura local para no depender de servicios cloud (ni de API keys).

## Qdrant (vector DB)

Se usa Qdrant en local en lugar de Qdrant Cloud. No hace falta API key: el
cliente apunta a `http://localhost:6333`.

### Opción A — docker compose (recomendado)

Desde esta carpeta:

```bash
docker compose up -d      # arrancar
docker compose down       # parar (los datos se conservan en el volumen)
docker compose down -v    # parar y BORRAR los datos
```

### Opción B — docker run (equivalente)

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### Puertos

| Puerto | Uso                                                    |
|--------|--------------------------------------------------------|
| 6333   | REST API + dashboard web (http://localhost:6333/dashboard) |
| 6334   | gRPC API                                               |

### Persistencia

Los datos viven en el volumen Docker `qdrant_storage`, así que sobreviven a
reinicios del contenedor. Para empezar de cero: `docker compose down -v`.
