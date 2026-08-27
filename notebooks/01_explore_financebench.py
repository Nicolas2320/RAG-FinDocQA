import pandas as pd

qa = pd.read_json("data/raw/financebench/financebench_open_source.jsonl", lines=True)
meta = pd.read_json("data/raw/financebench/financebench_document_information.jsonl", lines=True)
full = pd.merge(qa, meta, on="doc_name")

print(f"Preguntas: {len(qa)}")
print(f"Documentos únicos: {qa['doc_name'].nunique()}")
print(full[["company_x", "company_y", "doc_type", "doc_period"]].drop_duplicates())
print(qa.iloc[0].to_dict())