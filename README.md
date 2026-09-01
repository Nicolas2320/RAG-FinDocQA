# FinDocQA

RAG system for question-answering over SEC financial filings (10-K / 10-Q),
built on the [FinanceBench](https://github.com/patronus-ai/financebench) corpus.
Answers are grounded strictly in retrieved filing text and cite the source
document + page.

- [LIMITATIONS.md](./LIMITATIONS.md): known failure modes.

## Two implementations

| Path             | Stack                                                     | Qdrant collection                |
|------------------|----------------------------------------------------------|----------------------------------|
| `src/`           | hand-written (`qdrant-client`, `sentence-transformers`, `openai`) | `financebench_chunks`            |
| `rag_langchain/` | LangChain (`langchain-qdrant`, `langchain-openai`, `langchain-text-splitters`, …) | `financebench_chunks_langchain`  |

* `src/` has the FastAPI + Streamlit serving layer. `rag_langchain/` is a
retrieval-to-generation pipeline used for manual vs framework comparison. Notebook
`notebooks/05_ragas_comparison.py` scores the two side by side with RAGAS (Not yet compared with all questions).

## Quickstart

### 1. Install

```bash
uv sync                    # runtime deps (src/ pipeline + API + UI)
uv sync --extra dev        # + pytest / ruff
uv sync --extra langchain  # + the rag_langchain/ pipeline deps
```

### 2. Environment

```bash
cp .env.example .env
# then edit .env and set:
#   OPENAI_API_KEY=sk-...
#   OPENAI_MODEL=<chat model id> 
```

### 3. Data

Download the FinanceBench corpus
(https://github.com/patronus-ai/financebench) into `data/raw/financebench/`

```
data/raw/financebench/
├── financebench_open_source.jsonl
├── financebench_document_information.jsonl
└── pdfs/
    └── <doc_name>.pdf
```

### 4. Vector DB

```bash
docker compose up -d qdrant   # Qdrant on http://localhost:6333 (dashboard at /dashboard)
```

### 5. Build the index

Hand-written pipeline:

```bash
python -m src.chunking.chunking_metadata     # PDFs -> data/processed/chunks/financebench_chunks.jsonl
python -m src.embedding.embedding_qdrant     # chunks -> Qdrant collection `financebench_chunks`
```

LangChain pipeline (independent chunk file + collection):

```bash
python -m rag_langchain.chunking.chunking_metadata   # -> data/processed/chunks/financebench_chunks_langchain.jsonl
python -m rag_langchain.embedding.embedding_qdrant   # -> Qdrant collection `financebench_chunks_langchain`
```

### 6. Run

```bash
# one check query from the CLI
python -m src.pipeline "What is the FY2018 capital expenditure amount for 3M?"
python -m rag_langchain.pipeline "What is the FY2018 capital expenditure amount for 3M?"

# UI (needs the API running)
streamlit run src/ui/streamlit_app.py

# or the whole stack (qdrant + api + ui) in containers
docker compose up -d --build
```

## Evaluation

RAGAS scoring against the FinanceBench gold Q&A set. The LLM model from
`.env` is the judge LLM. `BAAI/bge-small-en-v1.5` is the judge's embedding
model. Both scripts default to a 10-question sample (`SAMPLE_SIZE`). Total
evaluation would be **150 questions**.

```bash
python notebooks/04_ragas_evaluation.py   # src/ pipeline  -> data/processed/ragas_results.csv from manual RAG
python notebooks/05_ragas_comparison.py   # src/ vs rag_langchain/ -> data/processed/ragas_comparison.csv Manual vs Langchain
```

Latest 10-question run (faithfulness / answer relevancy / context precision /
context recall):

| Pipeline    | Faithfulness | Answer rel. | Ctx precision | Ctx recall | Latency/q |
|-------------|--------------|-------------|---------------|------------|-----------|
| Manual      | 0.68         | 0.45        | 0.34          | 0.20       | ~3.3 s    |
| LangChain   | 0.69         | 0.43        | 0.29          | 0.23       | ~4.2 s    |

Sample is small and metrics are noisy because we evaluate just 10 questions, treat these as a smoke test, not a
benchmark. Context recall/precision are low. See [LIMITATIONS.md](./LIMITATIONS.md).

## Other notebooks

| Script | Purpose |
|--------|---------|
| `notebooks/01_explore_financebench.py` | Inspect the QA + document-metadata JSONL files. |
| `notebooks/02_distribution_chunking_evidence.py` | Chunk-count distribution per doc. Check gold evidence lands in the right chunk (~78% exact, rest formatting-only misses). |
| `notebooks/03_pilot_test.ipynb` | Re-run pipeline on questions whose evidence page was missed in an earlier pilot. |

## Status

- Data setup
- Parsing & chunking
- Embedding & indexing
- Retrieval & generation
- API + UI
- Evaluation (RAGAS)
- Docker Compose + GitHub Actions CI (lint + tests)
- Parallel LangChain implementation + head-to-head RAGAS comparison

## Future work

- **Full eval run** — set `SAMPLE_SIZE = None` in the eval scripts to score all
  150 questions and publish real numbers in `notebooks/05_ragas_comparison.py`.
- **Table-aware extraction**: replace `page.get_text()` with `pdfplumber` /
  `camelot` / `unstructured` (or multimodal page-image reading) to fix the
  multi-column numeric-table misreads based on [LIMITATIONS.md](./LIMITATIONS.md).
- **Token/cost logging**: capture the OpenAI `usage` block per query.
