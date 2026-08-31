# FinDocQA

RAG system for question-answering over SEC financial filings. See
[PROJECT_PLAN.md](./PROJECT_PLAN.md) for the full architecture, dataset,
stack, and 1-week build plan. Known failure modes are documented in
[LIMITATIONS.md](./LIMITATIONS.md).

## Quickstart

All commands are run **from the repo root** — the scripts import as
`src.<module>`, so they must be invoked with `python -m`, not by path.

### 1. Install

```bash
uv sync                # runtime deps
uv sync --extra dev    # + pytest / ruff
```

### 2. Environment

```bash
cp .env.example .env
# then edit .env and set:
#   OPENAI_API_KEY=sk-...
#   OPENAI_MODEL=gpt-version
```

### 3. Data

Download the FinanceBench corpus
(https://github.com/patronus-ai/financebench) into `data/raw/financebench/`:

```
data/raw/financebench/
├── financebench_open_source.jsonl
├── financebench_document_information.jsonl
└── pdfs/
    └── <doc_name>.pdf
```

### 4. Vector DB

```bash
cd docker && docker compose up -d && cd ..   # Qdrant on http://localhost:6333
```

### 5. Build the index

```bash
python -m src.chunking.chunking_metadata     # PDFs -> data/processed/chunks/financebench_chunks.jsonl
python -m src.embedding.embedding_qdrant     # chunks -> Qdrant collection
```

### 6. Run

```bash
# one-off query from the CLI
python -m src.pipeline "What is the FY2018 capital expenditure amount for 3M?"

# API (http://localhost:8000/docs)
uvicorn src.api.main:app --reload

# UI (needs the API running)
streamlit run src/ui/streamlit_app.py
```

## Status

- [x] Day 1: data setup
- [x] Day 2: parsing & chunking
- [x] Day 3: embedding & indexing
- [x] Day 4: retrieval & generation
- [x] Day 5: API + UI
- [x] Day 6: evaluation (RAGAS)
- [ ] Day 7: polish, Docker, CI, demo
