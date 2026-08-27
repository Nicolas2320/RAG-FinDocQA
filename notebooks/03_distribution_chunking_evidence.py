import pandas as pd

chunks = pd.read_json("data/processed/chunks/financebench_chunks.jsonl", lines=True)
por_doc = chunks.groupby("doc_name").size()
print(por_doc.describe())
print("Mínimo:", por_doc.idxmin(), por_doc.min())
print("Máximo:", por_doc.idxmax(), por_doc.max())

# Results:
# min         9.000000
# 25%       365.000000
# 50%       519.500000
# 75%       712.750000
# max      1738.000000
# dtype: float64
# Mínimo: FOOTLOCKER_2022_8K_dated-2022-05-20 9
# Máximo: JPMORGAN_2022_10K 1738

# si algún documento salió con muy pocos chunks (por ejemplo 5 o 10, cuando el resto tiene cientos), es señal de que ese PDF específico podría ser una imagen escaneada sin texto extraíble. PyMuPDF no lanza error en ese caso, simplemente devuelve texto vacío por página

qa = pd.read_json("data/raw/financebench/financebench_open_source.jsonl", lines=True)

hits = 0
total_evidencias = 0
for _, row in qa.iterrows():
    for ev in row["evidence"]:
        total_evidencias += 1
        candidatos = chunks[
            (chunks.doc_name == ev["doc_name"]) &
            (chunks.page_num == ev["evidence_page_num"])
        ]
        if candidatos["text"].str.contains(ev["evidence_text"][:60], case=False, regex=False).any():
            hits += 1

print(f"Evidencia encontrada en el chunk correcto: {hits}/{total_evidencias} ({hits/total_evidencias:.0%})")