# FinDocQA — RAG over SEC Financial Filings

**One-liner for the resume:** A production-style Retrieval-Augmented Generation
system that answers questions over real SEC financial filings (10-K/10-Q),
with hybrid retrieval, re-ranking, automated RAG evaluation, and a served API —
built to demonstrate Data Engineering skills relevant to banking/financial-services roles.

## Why this project

- Directly relevant to Data Engineer roles in banking (e.g. Sii Group) — the
  domain is financial filings, not a generic PDF-chat demo.
- Complements `realtime-ecom-rt-platform` (Kafka/lakehouse project) by adding
  an unstructured-data / LLM-retrieval pipeline to the portfolio, so the two
  projects together read as "structured + unstructured data engineering."
- Includes a real evaluation harness (RAGAS), which is what separates a
  junior "LangChain toy" from a project that shows engineering judgment.

## Dataset

**FinanceBench** — https://github.com/patronus-ai/financebench
- Public SEC filings (10-K, 10-Q, earnings docs) for ~30 public companies,
  including financial-sector names.
- Ships with ~150 expert-written question/answer pairs + the exact evidence
  page/quote for each answer — this becomes the gold evaluation set, so you
  don't have to hand-write eval questions.
- Fallback / stretch goal: swap in a live connector to the SEC EDGAR
  full-text search API (data.sec.gov) to pull filings for specific banks —
  turns the ingestion layer into a real API-integration exercise for v2.

## Architecture

```
 [SEC filings: PDFs]
        |
        v
 (1) INGESTION  --------------------->  data/raw/  (bronze: raw PDFs + metadata)
        |
        v
 (2) PARSING + CHUNKING  ------------->  data/processed/chunks/  (silver: JSONL
        |                                  chunks tagged with company, filing
        |                                  type, fiscal year, page number)
        v
 (3) EMBEDDING  ---------------------->  Qdrant vector DB (gold: vectors +
        |                                  metadata, filterable)
        v
 (4) RETRIEVAL (hybrid: dense + BM25)
        + RE-RANKING (cross-encoder)
        |
        v
 (5) GENERATION (LLM + citation prompt) --> answer + cited source pages
        |
        v
 (6) SERVING: FastAPI  /query  endpoint  +  Streamlit demo UI
        |
        v
 (7) EVALUATION: RAGAS metrics (faithfulness, answer relevancy,
     context precision/recall) computed against FinanceBench's gold Q&A set
```

## Tech stack

| Layer          | Choice                                              |
|----------------|------------------------------------------------------|
| Env/deps       | `uv` (consistent with your other project)            |
| Parsing        | PyMuPDF / `unstructured`                             |
| Orchestration  | LlamaIndex or LangChain pipelines (Prefect optional)  |
| Vector DB      | Qdrant (Docker locally, or free managed cloud tier)   |
| Embeddings     | OpenAI `text-embedding-3-small` (or local `bge-small-en` for zero cost) |
| Reranker       | cross-encoder `bge-reranker-base` (local, free) or Cohere Rerank |
| Generation LLM | GPT-4o-mini or Claude Haiku, citation-constrained prompt |
| API            | FastAPI                                              |
| UI             | Streamlit                                            |
| Evaluation     | RAGAS                                                |
| Testing/CI     | pytest + GitHub Actions (uv-based, like your other repo) |
| Packaging      | Docker Compose (qdrant + api + ui)                    |

## 1-week MVP plan

- **Day 1 — Setup & data.** `uv init`, repo skeleton, clone/download
  FinanceBench docs + QA jsonl into `data/raw/`. Explore a few filings and
  the QA pairs manually.
- **Day 2 — Parsing & chunking.** Extract text per page from PDFs, chunk
  (recursive/semantic splitter), attach metadata (company, filing type,
  fiscal year, page). Write chunks to `data/processed/chunks/*.jsonl`.
- **Day 3 — Embedding & indexing.** Batch-embed chunks, upsert into Qdrant
  with metadata payload for filtering (e.g. filter by company or filing type).
- **Day 4 — Retrieval & generation.** Build hybrid retriever (dense + BM25),
  add cross-encoder re-ranking, wire up the LLM generation step with a
  prompt that forces citations back to page numbers. Sanity-check on ~10
  FinanceBench questions by hand.
- **Day 5 — Serving.** FastAPI `/query` endpoint (question in, answer +
  sources out). Minimal Streamlit UI on top for a live demo / recording.
- **Day 6 — Evaluation.** Run the full FinanceBench QA set through the
  pipeline, score with RAGAS (faithfulness, answer relevancy, context
  precision/recall), save a metrics report (`eval_report.md` or a small
  table/chart).
- **Day 7 — Polish.** README with architecture diagram + real metrics,
  Dockerfile + docker-compose.yml, GitHub Actions CI (lint + tests), record
  a short demo GIF, write the resume bullets (draft below) and push to GitHub.

## Resume bullets (draft — fill in your real numbers after Day 6)

- "Built a Retrieval-Augmented Generation system over SEC financial filings
  (FinanceBench corpus), implementing hybrid dense+sparse retrieval with
  cross-encoder re-ranking on Qdrant; evaluated with RAGAS, achieving
  __% faithfulness and __% context precision on a 150-question benchmark."
- "Designed an end-to-end unstructured-data pipeline (ingestion → chunking
  with metadata lineage → embedding → indexing) served via a FastAPI
  endpoint and containerized with Docker Compose, with CI via GitHub Actions."

## Stretch goals (if time remains / v2)

- Swap FinanceBench's static corpus for a live SEC EDGAR ingestion connector
  (real "Data Engineering" ETL story: scheduled pulls, incremental updates).
- Add a lightweight Prefect/Airflow DAG around ingestion → chunk → embed.
- Add query/answer logging to Postgres for basic observability (latency,
  retrieved chunk IDs, token usage).
- Multi-tenant filtering by company/fiscal year in the UI.

## Key links

- FinanceBench: https://github.com/patronus-ai/financebench
- RAGAS: https://github.com/explodinggradients/ragas
- Qdrant: https://qdrant.tech/documentation/
- SEC EDGAR full-text search API (stretch goal): https://www.sec.gov/edgar/search/