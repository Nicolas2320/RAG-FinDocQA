import json
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW = Path("data/raw/financebench")
OUT = Path("data/processed/chunks")
OUT.mkdir(parents=True, exist_ok=True)

qa = pd.read_json(RAW / "financebench_open_source.jsonl", lines=True)
meta = pd.read_json(RAW / "financebench_document_information.jsonl", lines=True)
docs = qa["doc_name"].unique().tolist()

# Mismo chunk_size/overlap que el proyecto manual, para que el tamaño de
# ventana sea comparable, la diferencia real está en CÓMO corta cada uno.
splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=400)

records = []
missing = []
for doc_name in docs:
    pdf_path = RAW / "pdfs" / f"{doc_name}.pdf"
    if not pdf_path.exists():
        missing.append(doc_name)
        continue
    row = meta[meta["doc_name"] == doc_name].iloc[0]

    # mode="page" -> un Document por página, con metadata {"source":..., "page": idx}
    pages = PyMuPDFLoader(str(pdf_path), mode="page").load()

    for page_doc in pages:
        page_num = page_doc.metadata["page"]  # verifica que sea 0-indexado como en pymupdf.open()
        if not page_doc.page_content.strip():
            continue
        for i, chunk_text in enumerate(splitter.split_text(page_doc.page_content)):
            records.append({
                "chunk_id": f"{doc_name}_p{page_num}_c{i}",
                "doc_name": doc_name,
                "company": row["company"],
                "doc_type": row["doc_type"],
                "doc_period": int(row["doc_period"]),
                "page_num": page_num,
                "text": chunk_text,
            })

with open(OUT / "financebench_chunks_langchain.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in records)

print(f"{len(records)} chunks de {len(docs) - len(missing)} documentos (LangChain)")
if missing:
    print("Faltan PDFs para:", missing)