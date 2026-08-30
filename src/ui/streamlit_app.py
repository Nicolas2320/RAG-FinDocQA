# Front-end mínimo en Streamlit. Es un cliente HTTP del API
# (src/api/main.py), que debe estar corriendo en localhost:8000.
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/query")

st.set_page_config(page_title="FinDocQA", page_icon="📊")
st.title("FinDocQA: Financial Document Q&A")
st.caption("RAG sobre 10-K/10-Q de FinanceBench. Respuestas basadas únicamente en los documentos indexados.")

question = st.text_input("Pregunta:", placeholder="What is the FY2018 capital expenditure amount for 3M?")

# Solo actúa si se pulsó el botón y hay texto. El spinner se muestra mientras
# se espera la respuesta del API.
if st.button("Preguntar") and question:
    with st.spinner("Buscando y generando respuesta..."):
        try:
            # POST al endpoint /query; timeout amplio porque el rerank + LLM pueden tardar.
            resp = requests.post(API_URL, json={"question": question}, timeout=60)
            resp.raise_for_status()  # convierte 4xx/5xx en excepción
            data = resp.json()
        except requests.RequestException as e:
            # Error de red o respuesta HTTP de error: se muestra y no se pinta nada más.
            st.error(f"Error llamando a la API: {e}")
        else:
            # Respuesta + fuentes.
            st.subheader("Respuesta")
            # Escapa los '$' para que Streamlit no los interprete como delimitadores LaTeX.
            st.markdown(data["answer"].replace("$", "\\$"))
            st.subheader("Fuentes recuperadas")
            for s in data["sources"]:
                st.markdown(f"- **{s['doc_name']}**, página {s['page_num']} (score: {s['rerank_score']:.3f})")