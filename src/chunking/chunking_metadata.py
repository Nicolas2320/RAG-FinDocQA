import json
from pathlib import Path
import pymupdf
import pandas as pd

RAW = Path("data/raw/financebench")
OUT = Path("data/processed/chunks")
OUT.mkdir(parents=True, exist_ok=True)

qa = pd.read_json(RAW / "financebench_open_source.jsonl", lines=True)
meta = pd.read_json(RAW / "financebench_document_information.jsonl", lines=True)
needed_docs = qa["doc_name"].unique().tolist()  # los 84 que sí usan las preguntas

PAGE_OFFSET = 0 

def chunk_text(text, max_chars=1800, overlap=400):
    # 1800/400 (antes 1200/200): un estado financiero denso como el
    # "Consolidated Statement of Cash Flows" de 3M (~2000 chars) cabe casi
    # entero en un chunk en vez de partirse a la mitad.
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + max_chars])
        start += max_chars - overlap
    return chunks

records = []
missing = []
for doc_name in needed_docs:
    pdf_path = RAW / "pdfs" / f"{doc_name}.pdf"
    if not pdf_path.exists():
        missing.append(doc_name)
        continue
    row = meta[meta["doc_name"] == doc_name].iloc[0]
    pdf = pymupdf.open(pdf_path)
    for page_idx in range(len(pdf)):
        text = pdf[page_idx].get_text()
        if not text.strip():
            continue
        human_page = page_idx + PAGE_OFFSET
        for i, chunk in enumerate(chunk_text(text)):
            records.append({
                "chunk_id": f"{doc_name}_p{human_page}_c{i}",
                "doc_name": doc_name,
                "company": row["company"],
                "doc_type": row["doc_type"],
                "doc_period": int(row["doc_period"]),
                "page_num": human_page,
                "text": chunk,
            })
            

with open(OUT / "financebench_chunks.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"{len(records)} chunks de {len(needed_docs) - len(missing)} documentos")
if missing:
    print("Faltan PDFs para:", missing)