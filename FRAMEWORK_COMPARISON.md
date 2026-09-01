# FinDocQA: tu pipeline RAG vs. LangChain y LlamaIndex

Guía de referencia construida a partir de un recorrido archivo por archivo del proyecto `FinDocQA`, comparando cada pieza escrita a mano contra su equivalente conceptual en LangChain y LlamaIndex — para entender qué haría cada framework por ti si lo usaras, sin haberlo usado.

**Cómo leer este documento:** sigue el orden de construcción del proyecto (Day 1 a Day 5, según `PROJECT_PLAN.md`). Cada sección cubre un archivo de `src/`, explica qué hace, y lo mapea contra la abstracción equivalente en cada framework. Al final hay una tabla resumen y dos temas transversales (selección de modelo LLM / conteo de tokens, y por qué preguntar en español da resultados peores).

---

## Day 1 — Datos y `src/config.py`

**Los datos.** `data/raw/financebench/` trae tres piezas: el jsonl de preguntas/respuestas, el jsonl de metadata por documento (empresa, tipo, año fiscal), y los PDFs. El proyecto cruza ambos jsonl para quedarse solo con los 84 documentos que las preguntas realmente usan. Esta lógica de selección es específica del dataset — ni LangChain ni LlamaIndex tienen opinión sobre qué subconjunto de tus documentos te interesa; siempre es una decisión tuya.

**`config.py` como fuente única de verdad.** Constantes compartidas (nombre de colección, URL de Qdrant, modelo de embedding, modelo de reranker, prefijo de query, `max_length` del reranker) más una función `get_device()`.

- **LlamaIndex** tiene un paralelo casi literal: el objeto global `Settings` (`Settings.embed_model`, `Settings.llm`, `Settings.chunk_size`, ...), que se inyecta implícitamente en todo lo que construyas después.
- **LangChain** no tiene equivalente de framework — instancias cada objeto explícito donde lo necesitas, o armas tu propio `settings.py` (típicamente con `pydantic.BaseSettings`). Tu `config.py` ya es ese patrón, solo sin pydantic.
- El `QUERY_PREFIX` (se aplica solo a la query, nunca a los pasajes indexados) es una particularidad de `bge-small-en-v1.5`, no un concepto de RAG en general — ningún framework lo sabe automáticamente; hay que leerlo de la model card y replicarlo tú.
- `get_device()` es conveniencia de `sentence-transformers`/`torch`, equivalente a pasar `model_kwargs={"device": ...}` al wrapper de embeddings en cualquiera de los dos frameworks.

---

## Day 2 — `src/chunking/chunking_metadata.py`

**Parsing.** PyMuPDF extrae texto página por página — cada chunk queda contenido dentro de una sola página, nunca cruza el límite entre dos.

- **LangChain**: `PyMuPDFLoader(pdf_path).load()` hace exactamente esto, devolviendo un `Document` por página con metadata `source`/`page` ya poblada.
- **LlamaIndex**: `SimpleDirectoryReader` / un `PDFReader` específico cumplen el mismo rol.

**Chunking.** La función `chunk_text(max_chars=1800, overlap=400)` es un corte **ciego por caracteres** — avanza una ventana sin mirar el contenido, pudiendo partir una palabra o cifra justo en el borde.

- **LangChain**: `CharacterTextSplitter` es el equivalente directo (mismo comportamiento ciego). Lo que se usa normalmente es `RecursiveCharacterTextSplitter`, que intenta cortar primero en `\n\n`, luego `\n`, luego espacio, y solo al final por carácter — respeta párrafos/oraciones cuando puede.
- **LlamaIndex**: el `NodeParser` típico es `SentenceSplitter`. Diferencia importante: **cuenta en tokens (tiktoken), no en caracteres** — un `chunk_size=1800` ahí no es comparable 1:1 con tu `max_chars=1800`.

> **Conexión con `LIMITATIONS.md`:** el "future work" que documentaste ahí (`pdfplumber`, `camelot`, `unstructured`, o extracción multimodal para tablas) es exactamente lo que reemplazaría este parser en un pipeline con LangChain/LlamaIndex — ambos tienen loaders que envuelven `unstructured` (`UnstructuredPDFLoader` en LangChain) o parsers table-aware (LlamaParse en LlamaIndex), que preservan la estructura de fila/columna en vez de aplanarla a texto lineal como hace `get_text()`. El bug de 3M que documentaste no es una limitación de RAG en general — es una limitación específica de este parser.

**Metadata por chunk.** Cada chunk lleva su propia copia de `company`, `doc_type`, `doc_period`, `page_num` — no solo el documento padre.

*¿Por qué se denormaliza así?* Una vez que los chunks están en Qdrant, cada uno se recupera como unidad independiente y autocontenible — no hay un segundo lookup a una tabla de documentos. Esto habilita dos cosas directamente: (1) el filtro de metadata en la búsqueda (`company == "3M"` se evalúa sobre el payload del chunk mismo), y (2) la citación en la respuesta final (`doc_name`/`page_num` se leen directo del chunk recuperado). Es el clásico trade-off normalizado (una tabla, con `JOIN`) vs. denormalizado (todo repetido en cada fila) — las bases vectoriales denormalizan a propósito para que cada resultado de búsqueda sea autosuficiente.

- **LangChain**: `Document.metadata` es un dict libre; el splitter propaga automáticamente la metadata del `Document` padre a cada chunk hijo.
- **LlamaIndex**: mismo concepto — patrón idiomático `SimpleDirectoryReader(file_metadata=fn)`; los `Node` heredan la metadata del `Document` padre.
- El `chunk_id` determinístico (`f"{doc_name}_p{page}_c{i}"`) no tiene equivalente automático en ningún framework (generan UUIDs por defecto); hay que pasar IDs explícitos al vector store igual que aquí.

---

## Day 3 — `src/embedding/embedding_qdrant.py`

**Ciclo de vida de la colección.** `collection_exists → delete_collection → create_collection` es reindexado manual completo.

- **LangChain**: `Qdrant.from_documents(docs, embedding, url=..., collection_name=..., force_recreate=True)` hace delete + create + embed + upsert en una sola llamada.
- **LlamaIndex**: `QdrantVectorStore(collection_name=..., client=...)` + `StorageContext` maneja la creación/recreación.

**Embedding en batches.** `model.encode([c["text"] for c in batch], normalize_embeddings=True)` — el embedding se hace **solo sobre el texto**, nunca sobre la metadata; la metadata jamás entra a la red neuronal, solo viaja adjunta como payload. Por eso el filtro de metadata en Qdrant es un mecanismo aparte (structured match) del todo independiente de la búsqueda semántica: primero se filtra por payload, y solo entre los sobrevivientes se calcula similitud coseno.

Aquí tampoco se aplica el `QUERY_PREFIX` (solo se usa en `filtered_search.py`, con la query) — esa asimetría tiene un paralelo de diseño directo: la interfaz `Embeddings` de ambos frameworks separa `embed_documents(texts)` de `embed_query(text)` precisamente para poder aplicar tratamientos distintos a query vs. pasaje sin depender de que el desarrollador se acuerde manualmente en cada script.

**Upsert.** `PointStruct(id=i+j, vector=v.tolist(), payload=chunk)` — el chunk completo es el payload; confirma la denormalización del Day 2. `add_documents()` en LangChain vuelca `page_content` + `metadata` al payload automáticamente; en LlamaIndex, `Node.text` + `Node.metadata` se mapean igual vía `QdrantVectorStore`.

**Un chunk = un vector.** `model.encode()` sobre un batch de 128 textos devuelve 128 vectores — el batching es pura optimización de rendimiento, no cambia la relación 1:1. `bge-small-en-v1.5` produce vectores de **384 dimensiones** (por eso `vector_size` se calcula dinámico con `get_embedding_dimension()` en vez de hardcodearse).

*Comparación con tokens en un LLM:* un transformer sí genera un vector **por token** internamente (contextual, dinámico, se actualiza capa a capa por atención) — eso es lo que usa un LLM para generar texto. Un modelo de sentence-embeddings genera esos mismos vectores por token internamente, pero al final aplica **pooling** (promedio o token `[CLS]`) para colapsarlos en un único vector fijo por chunk — eso es lo que devuelve `model.encode()` y lo que se guarda en Qdrant. Token embedding = uno por palabra/subpalabra, contextual; chunk embedding = uno por texto completo, estático una vez calculado.

**"Payload"** es el término de Qdrant (Pinecone lo llama "metadata", Weaviate "properties") para el JSON arbitrario adjunto a un vector: `point = {id, vector, payload}`. El vector es lo único que se usa para similitud; el payload es lo que se devuelve y sobre lo que se filtra.

**Mitigar el costo de actualizar metadata denormalizada.** Si mañana cambia el nombre de una empresa, no hace falta re-embeber nada — el vector no cambia, solo el payload:

1. Casi todas las vector DBs (Qdrant `set_payload`/`overwrite_payload`, Pinecone, Weaviate, Milvus) permiten actualizar el payload de puntos existentes vía filtro, sin tocar el vector.
2. Patrón "vector DB como índice, no como fuente de verdad": guardar solo un `doc_id` en el payload y mantener la metadata real normalizada en Postgres, con `JOIN` al responder — por esto muchos equipos usan `pgvector` en vez de una vector DB pura.
3. Pipelines idempotentes (como el JSONL intermedio de este proyecto) permiten parchear metadata y re-upsertear (mismo `id` = overwrite) sin re-parsear ni re-embeber.

Ni LangChain ni LlamaIndex resuelven esto arquitectónicamente por sí solos — igual denormalizan hacia el vector store que elijas.

**Corte grande vs. "usar el framework de verdad":** el quickstart de LlamaIndex hace `VectorStoreIndex.from_documents(...)` — chunking + embedding + upsert en una sola llamada, usando `Settings.node_parser` y `Settings.embed_model`. Este proyecto separó Day 2 y Day 3 en dos scripts con un JSONL intermedio inspeccionable — mejor práctica de ingeniería de datos (auditable, re-embebible sin re-parsear los PDFs), aunque el "camino feliz" del framework lo oculte por defecto.

---

## Day 4a — `src/retrieval/filtered_search.py`

`extract_year`/`extract_company` son NER hecho a mano con regex y substring-matching (con el detalle fino de ordenar `known_companies` de más largo a más corto para evitar falsos positivos por substring). `build_metadata_filter` + `filtered_dense_search` arman un filtro de metadata que Qdrant aplica como pre-condición antes del ranking por coseno.

**Esto tiene nombre propio en ambos frameworks: Self-Query Retrieval.**

- **LangChain**: `SelfQueryRetriever` — le das un esquema de metadata (campos, tipo, descripción en lenguaje natural) y un **LLM** traduce la pregunta a un filtro estructurado automáticamente, en vez de regex. La versión sin LLM (más parecida a lo que hizo este proyecto) es pasar un filtro fijo vía `vectorstore.as_retriever(search_kwargs={"filter": ...})`.
- **LlamaIndex**: `VectorIndexRetriever(filters=MetadataFilters(...))` es el equivalente directo de un filtro explícito; `VectorIndexAutoRetriever` usa un LLM + un `VectorStoreInfo` para inferir el filtro desde la pregunta — el equivalente exacto del `SelfQueryRetriever`.

**Trade-off entre el regex propio y el enfoque LLM de los frameworks:**

| | Regex (este proyecto) | SelfQuery / AutoRetriever (LLM) |
|---|---|---|
| Costo | Gratis, cero latencia extra | Una llamada a LLM extra por pregunta |
| Robustez | Frágil — solo funciona si el nombre aparece literal (p. ej. "Minnesota Mining" no matchea "3M") | Entiende sinónimos y paráfrasis |
| Riesgo | Ninguno (determinístico) | El LLM puede alucinar un filtro incorrecto |

No hay una respuesta "correcta" — es un trade-off consciente costo/latencia vs. robustez, y documentarlo como decisión de diseño (no como limitación no examinada) suma en un portfolio.

El `print` de debug (`[empresa=..., año=..., filtro_aplicado=...]`) es logging manual; en un pipeline LangChain tendrías callbacks/tracing (LangSmith) capturando automáticamente qué filtro se generó y qué se recuperó, sin instrumentar cada función a mano.

---

## Day 4b — `src/retrieval/rerank.py`

**Bi-encoder vs. cross-encoder — el concepto central de este archivo.** El bi-encoder (`bge-small`, usado en Day 3/4a) codifica query y chunk **por separado**, cada uno en su vector, precalculable — rápido pero burdo, nunca "se ven" entre sí. El cross-encoder (`bge-reranker-base`, usado aquí) codifica el **par** `(question, text)` junto en una sola pasada, permitiendo atención cruzada token-a-token — mucho más preciso, pero caro: no se puede precalcular y no escala a millones de documentos.

Por eso el pipeline hace **retrieval en dos etapas**: el bi-encoder trae un conjunto amplio y barato optimizado para *recall* (`dense_top_k=50`), el cross-encoder lo reordena con precisión y lo reduce a los mejores (`final_top_k=10`) antes de generar. Patrón estándar de sistemas de búsqueda, no exclusivo de RAG.

- **LangChain**: `ContextualCompressionRetriever` + `CrossEncoderReranker` (sobre `HuggingFaceCrossEncoder`); alternativa gestionada: `CohereRerank`.
- **LlamaIndex**: node postprocessor `SentenceTransformerRerank`, conectado como `node_postprocessors=[...]` al `query_engine`.

Dato clave: ambos wrappers usan por dentro la **misma clase `CrossEncoder` de `sentence-transformers`** que este proyecto importa directo — no es una versión simplificada, es el mismo mecanismo sin la capa de interfaz del framework.

### Por qué `max_length=512` importa: ejemplo de truncamiento

`bge-reranker-base` soporta como máximo 512 tokens. Un token no es un carácter — en prosa inglesa normal la proporción es ~4 caracteres/token, pero texto financiero denso en cifras tokeniza peor (`"$1,577"` se parte en varios tokens: `"$"`, `"1"`, `","`, `"577"`), bajando esa proporción a ~2 caracteres/token o menos. Los tokenizers, por defecto, truncan desde **el final** (se quedan con el principio, descartan la cola).

Ejemplo ilustrativo con la forma típica de una página de 10-K:

```
[~1400 caracteres de prosa narrativa: "Durante el año fiscal 2018, la compañía
continuó invirtiendo en sus segmentos operativos..." → ~350 tokens, eficiente]

Capital Spending by year and region (in millions):
United States: $891 (2018), $823 (2017), $756 (2016)
EMEA:          $412 (2018), $389 (2017), $401 (2016)
Total Company: $1,577 (2018), $1,373 (2017), $1,420 (2016)   ← ~150-250 tokens
```

Total: ~500-600 tokens para un chunk de solo 1800 caracteres — se pasa del límite de 512 antes de terminar de leer la tabla. La cifra que responde la pregunta (`Total Company: $1,577`) suele quedar justo al final, siendo lo primero que se pierde en el truncamiento. El cross-encoder no da error: simplemente puntúa la relevancia con lo que sí alcanzó a leer (la prosa, sin cifras) y probablemente subestima un chunk que sí tenía la respuesta correcta. Es el mismo síntoma documentado en `LIMITATIONS.md` para el caso de 3M (una cifra correcta invisible para el modelo), aunque ahí la causa raíz fue distinta (sesgo hacia texto narrativo en el scoring, no truncamiento) — el patrón general de fondo es el mismo. Fijar `max_length=512` explícito no evita este riesgo — solo garantiza que el corte sea determinista en vez de depender del default del tokenizer.

---

## Day 4c — `src/generation/generate_answer.py` + `src/pipeline.py`

`format_context` (numerar chunks + cabecera de fuente + concatenar todo en un bloque) es la estrategia **"stuff"** — una de las cuatro clásicas de `RetrievalQA` en LangChain (`stuff`/`map_reduce`/`refine`/`map_rerank`), y equivalente al modo `compact` del response synthesizer de LlamaIndex. Funciona bien aquí porque solo 10 chunks caben cómodos en la ventana de contexto del modelo de generación.

El `SYSTEM_PROMPT` de citación (responder solo con los fragmentos dados, admitir cuando falta información, citar `[Fuente: documento, página X]`) es ingeniería de prompt pura — ningún framework te la resuelve mágicamente. LlamaIndex sí tiene un motor con nombre propio para justo este patrón, `CitationQueryEngine`, que numera los nodos fuente y agrega las instrucciones de citación por ti.

`pipeline.py::answer_question` es tu propia versión, escrita a mano, de lo que en cada framework es un objeto de primera clase:

- **LangChain**: composición **LCEL** con `|` — `retriever | rerank_step | format_docs | prompt | llm | parser`.
- **LlamaIndex**: `RetrieverQueryEngine` (retriever + `node_postprocessors=[reranker]` + `response_synthesizer`), o `index.as_query_engine(...)`. `query_engine.query(question)` devuelve `.response` + `.source_nodes` ≈ tu `{"answer": ..., "sources": [...]}`.

`chunks_para_llm` (dicts planos `doc_name`/`page_num`/`text`) cumple el mismo rol que las clases `Document`/`Node` de los frameworks: una capa de datos neutral que desacopla la estructura interna de Qdrant de lo que necesita la función de generación — ya aplicado correctamente, solo con un dict en vez de una clase con nombre.

---

## Day 5 — `src/api/main.py` + `src/ui/streamlit_app.py`

FastAPI con `/health` y `/query`, esquemas Pydantic para validar entrada/salida. **Nota de consistencia a corregir:** los defaults de `QueryRequest` (`dense_top_k=50`, `final_top_k=10`) no coinciden con los defaults de `answer_question()` en `pipeline.py` (`dense_top_k=50`, `final_top_k=10`) — hoy el comportamiento cambia según si invocas por CLI o por API sin especificar esos campos. Vale la pena unificarlos (idealmente que la API herede los defaults de `pipeline.py` en vez de redefinir los suyos).

- **LangChain**: la herramienta con nombre propio para esta capa es **LangServe** — toma una chain LCEL y genera rutas FastAPI automáticamente (`/invoke`, `/batch`, `/stream`, playground) a partir del esquema de entrada/salida de la chain, sin escribir `QueryRequest`/`QueryResponse` a mano.
- **LlamaIndex**: no tiene una herramienta de auto-serve tan prominente — el patrón típico es exactamente lo que hace este proyecto: envolver `query_engine.query()` a mano en un FastAPI propio.

En `streamlit_app.py`, la respuesta se muestra dos veces (`st.markdown(...)` con el `$` escapado, y luego `st.write(...)` sin escapar) — probablemente residuo de debugging; conviene quedarse solo con la línea de `st.markdown`.

---

## Selección de modelo LLM y conteo de tokens

Tokens de entrada = system prompt + contexto (chunks "stuffed") + pregunta. Tokens de salida = la respuesta generada.

**Dónde se cuentan:**

1. **Después de la llamada**: la respuesta de la API de OpenAI incluye `usage: {prompt_tokens, completion_tokens, total_tokens}` — ground truth real. `generate_answer.py` no lo captura ni loguea hoy — mejora barata pendiente, y coincide con un stretch goal ya escrito en `PROJECT_PLAN.md` ("Add query/answer logging to Postgres... token usage").
2. **Antes de la llamada** (estimación): `tiktoken` (tokenizer oficial de OpenAI) para contar tokens localmente antes de enviar el request.

LangChain tiene `get_openai_callback()` (suma automáticamente tokens/costo a través de una chain completa); LlamaIndex tiene `TokenCountingHandler` en `Settings.callback_manager` — ambos son wrappers convenientes sobre el mismo `usage` que la API ya devuelve.

**Por qué `max_completion_tokens=500` tiene sentido aunque el modelo soporte mucho más de salida:** es un techo de seguridad, no un objetivo — el modelo se detiene naturalmente al terminar su respuesta. Un QA factual con cita necesita típicamente 50-150 tokens; 500 da margen cómodo sin exponerse al peor caso (una respuesta que empiece a divagar), lo cual importa porque el token de **salida** casi siempre cuesta varias veces más que el de entrada (verificado en agosto 2026: GPT-5.4 mini cuesta $4.50/M tokens de salida contra $0.75/M de entrada, 6x más caro).

**Cómo decidir qué modelo usar, en general:**

1. **Complejidad de la tarea**: "sintetiza una respuesta a partir de contexto dado y cita la fuente" es una tarea acotada — no requiere el modelo más grande de la familia; un tier "mini" suele bastar.
2. **Ventana de contexto necesaria**: el input típico de este proyecto (system prompt + hasta 10 chunks de ~1800 caracteres + pregunta) ronda unos pocos miles de tokens en el peor caso — muy por debajo de cualquier ventana moderna (GPT-5.4 mini ofrece 400,000 tokens de contexto), así que el tamaño de ventana casi nunca es el cuello de botella para RAG salvo que se metan cientos de chunks sin resumir.
3. **Costo**: el input suele dominar el conteo total de tokens (los chunks pesan más que la respuesta corta), aunque el precio de salida importa igual por ser más caro por token.
4. **Latencia**: modelos "mini" responden más rápido — relevante para una UI síncrona.
5. **La forma correcta de decidir es empírica, no teórica**: correr el set de evaluación (RAGAS, pendiente en el Day 6 del plan) contra 2-3 modelos candidatos, comparar faithfulness/answer relevancy contra costo/latencia, y quedarse con el más barato que pase la barra de calidad mínima.

> Nota: al verificar el pricing vigente (agosto 2026), la familia `gpt-4o` que usa hoy este proyecto ya no aparece en la página de pricing actual de OpenAI — la línea vigente es `gpt-5.x`. Vale la pena confirmar la disponibilidad de `gpt-4o-mini` antes de apoyar el proyecto en ella a largo plazo (decisión ya tomada durante esta sesión: migrar `OPENAI_MODEL` a `gpt-5.4-nano`).

---

## Por qué preguntar en español da resultados distintos (y peores)

`bge-small-en-v1.5` es un modelo **monolingüe en inglés** — fue entrenado casi exclusivamente con pares de texto en inglés, así que nunca aprendió una alineación cruzada entre español e inglés (a diferencia de modelos explícitamente multilingües como `BAAI/bge-m3`, `intfloat/multilingual-e5-base`, o `paraphrase-multilingual-mpnet-base-v2`, entrenados con corpus paralelos para que frases equivalentes en distintos idiomas queden cerca en el espacio vectorial). No es que el español quede "un poco más lejos" en algún sentido geométrico genérico — es que la cercanía entre una pregunta en español y un pasaje en inglés que significan lo mismo es esencialmente ruido no aprendido por el modelo.

Dos factores lo agravan en este proyecto:

1. El `QUERY_PREFIX` mismo está en inglés — preguntar en español produce un input mixto (instrucción en inglés + pregunta en español) aún más distinto de lo que el modelo vio en su entrenamiento.
2. El reranker (`bge-reranker-base`) es igual de monolingüe — degrada la señal por segunda vez en la etapa de rerank.

**Matiz importante para diagnosticar:** la etapa de **generación** (GPT) sí es multilingüe y competente — el cuello de botella está específicamente en retrieval + rerank, no en la generación.

**Cómo arreglarlo:**

- **Modelos multilingües** (`bge-m3` + reranker multilingüe correspondiente) resuelven el problema de raíz, pero implican re-embeber todo el corpus desde cero.
- **Traducir la pregunta al inglés antes de buscar** (usando el propio LLM de generación) — no toca el índice existente, más barato de implementar. Tiene nombre en ambos frameworks: **query transformation**, un paso insertado antes del retriever en la chain (LCEL en LangChain) o el módulo de query transform en LlamaIndex.

Cambiar de modelo de embedding en cualquiera de los dos frameworks es tan simple como cambiar el nombre del modelo en el constructor de `Embeddings` — la interfaz es la misma sin importar qué modelo hay detrás, así que ese swap específico no obligaría a reescribir el resto del pipeline.

---

## Tabla resumen: tu archivo → concepto → LangChain → LlamaIndex

| Tu archivo | Qué hace | Equivalente LangChain | Equivalente LlamaIndex |
|---|---|---|---|
| `config.py` | Configuración global compartida | Sin equivalente directo; se instancia todo explícito o se arma un `settings.py` propio | `Settings` (objeto global) |
| `chunking_metadata.py` (parsing) | PDF → texto por página | `PyMuPDFLoader` | `SimpleDirectoryReader` / `PDFReader` |
| `chunking_metadata.py` (chunking) | Ventana ciega por caracteres | `CharacterTextSplitter` / `RecursiveCharacterTextSplitter` | `NodeParser` (`SentenceSplitter`, basado en tokens) |
| `embedding_qdrant.py` | Embeber + upsert a Qdrant | `Qdrant.from_documents(..., force_recreate=True)` | `VectorStoreIndex.from_documents(...)` |
| `filtered_search.py` | Filtro de metadata (regex) + dense search | `SelfQueryRetriever` (con LLM) / filtro fijo explícito (sin LLM) | `VectorIndexAutoRetriever` (con LLM) / `VectorIndexRetriever(filters=...)` (sin LLM) |
| `rerank.py` | Re-ranking con cross-encoder | `ContextualCompressionRetriever` + `CrossEncoderReranker` | `SentenceTransformerRerank` (node postprocessor) |
| `generate_answer.py` | Prompt de citación + llamada al LLM | `ChatOpenAI` + `ChatPromptTemplate` (o `CitationQueryEngine` para citación automática) | `LLM` wrapper + `response_synthesizer` / `CitationQueryEngine` |
| `pipeline.py` | Orquestación de todo el flujo | Composición LCEL (`retriever \| rerank \| prompt \| llm`) | `RetrieverQueryEngine` / `index.as_query_engine(...)` |
| `api/main.py` | Servir como API | `LangServe` (`add_routes`) | Sin herramienta de auto-serve equivalente; FastAPI manual (como este proyecto) |

---

*Documento generado a partir de una sesión de revisión conversacional del proyecto `FinDocQA`. No cubre Day 6 (evaluación con RAGAS) ni Day 7 (Docker/CI) porque aún no están implementados en el proyecto al momento de esta revisión.*
