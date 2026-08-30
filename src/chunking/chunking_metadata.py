import json
from pathlib import Path

import pandas as pd
import pymupdf

RAW = Path("data/raw/financebench")
OUT = Path("data/processed/chunks")
OUT.mkdir(parents=True, exist_ok=True)

# Preguntas y respuestas
qa = pd.read_json(RAW / "financebench_open_source.jsonl", lines=True)
# Metadata por documento
meta = pd.read_json(RAW / "financebench_document_information.jsonl", lines=True)

docs = qa["doc_name"].unique().tolist()  # Hay 84 documentos unicos que sí usan las preguntas

PAGE_OFFSET = 0 

def chunk_text(text, max_chars=1800, overlap=400):
    # * Parte el texto de una pagina en trozos solapados por ventana deslizante.
    # * Cada trozo tiene como maximo `max_chars` caracteres.
    # * La ventana avanza `max_chars - overlap`, asi que los ultimos
    # `overlap` caracteres de un trozo se repiten al inicio del siguiente.
    # Ese solape evita cortar una frase o una cifra justo en el limite y perder el contexto necesario para responder.
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + max_chars])
        start += max_chars - overlap
    return chunks

# Recorre los 84 documentos usados por las preguntas y construye un registro
# (fila del .jsonl de salida) por cada trozo de texto, adjuntandole su metadata.
records = []
missing = []  # documentos sin PDF en disco; se reportan al final
for doc_name in docs:
    # Localiza el PDF, si no esta, lo anota y pasa al siguiente documento.
    pdf_path = RAW / "pdfs" / f"{doc_name}.pdf"
    if not pdf_path.exists():
        missing.append(doc_name)
        continue
    # Metadata de este documento (company, doc_type, doc_period) desde el catalogo.
    row = meta[meta["doc_name"] == doc_name].iloc[0]
    pdf = pymupdf.open(pdf_path)
    # Recorre el PDF pagina a pagina para conservar el numero de pagina en cada chunk.
    for page_idx in range(len(pdf)):
        text = pdf[page_idx].get_text()
        if not text.strip():
            continue  # salta paginas vacias o solo imagen (sin texto extraible)
        real_page = page_idx + PAGE_OFFSET
        # Chunkea el texto de la pagina y crea un registro por cada trozo.
        for i, chunk in enumerate(chunk_text(text)):
            records.append({
                # chunk_id unico: documento + pagina + indice del trozo dentro de la pagina
                "chunk_id": f"{doc_name}_p{real_page}_c{i}",
                "doc_name": doc_name,
                "company": row["company"],
                "doc_type": row["doc_type"],
                "doc_period": int(row["doc_period"]),
                "page_num": real_page,
                "text": chunk,
            })
            
# Guardar los chunks
with open(OUT / "financebench_chunks.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in records)

print(f"{len(records)} chunks de {len(docs) - len(missing)} documentos")
if missing:
    print("Faltan PDFs para:", missing)