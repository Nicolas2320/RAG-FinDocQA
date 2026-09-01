import os

from dotenv import load_dotenv
from langchain_core.exceptions import LangChainException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

GENERATION_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not GENERATION_MODEL or not OPENAI_API_KEY:
    raise RuntimeError("Faltan variables de entorno: define OPENAI_MODEL y OPENAI_API_KEY en tu .env")

SYSTEM_PROMPT = """Eres un analista financiero que responde preguntas basándote ÚNICAMENTE en los \
fragmentos de documentos que se te proporcionan a continuación.

Reglas estrictas:
1. No uses conocimiento externo ni asumas cifras que no estén literalmente en los fragmentos.
2. Si la información no alcanza para responder con certeza, dilo explícitamente en vez de inventar un número.
3. Al final de tu respuesta, cita la fuente en el formato: [Fuente: NOMBRE_DOCUMENTO, página X], \
usando el nombre de documento y la página que aparecen en la cabecera del fragmento que utilizaste."""

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=GENERATION_MODEL, api_key=OPENAI_API_KEY, model_kwargs={"max_completion_tokens": 500})
    return _llm


def format_context(docs):
    bloques = [
        f"[{i}] (documento: {d.metadata['doc_name']}, página: {d.metadata['page_num']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    ]
    return "\n\n".join(bloques)

def generate_answer(question, docs):
    if not docs:
        return "No se recuperó ningún fragmento para responder la pregunta."
        
    context = format_context(docs)
    user_prompt = f"Fragmentos:\n\n{context}\n\nPregunta: {question}"
    
    try:
        response = get_llm().invoke([
            SystemMessage(content=SYSTEM_PROMPT), 
            HumanMessage(content=user_prompt)
        ])
    except LangChainException as e:
        return f"Error al llamar al modelo de generación: {e}"
        
    return response.content