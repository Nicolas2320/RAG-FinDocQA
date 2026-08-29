import requests
import streamlit as st

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="FinDocQA", page_icon="📊")
st.title("FinDocQA: Financial Document Q&A")
st.caption("RAG sobre 10-K/10-Q de FinanceBench. Respuestas basadas únicamente en los documentos indexados.")

question = st.text_input("Pregunta:", placeholder="What is the FY2018 capital expenditure amount for 3M?")

if st.button("Preguntar") and question:
    with st.spinner("Buscando y generando respuesta..."):
        try:
            resp = requests.post(API_URL, json={"question": question}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            st.error(f"Error llamando a la API: {e}")
        else:
            st.subheader("Respuesta")
            st.markdown(data["answer"].replace("$", "\\$"))
            st.write(data["answer"])
            st.subheader("Fuentes recuperadas")
            for s in data["sources"]:
                st.markdown(f"- **{s['doc_name']}**, página {s['page_num']} (score: {s['rerank_score']:.3f})")