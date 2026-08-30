import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

GENERATION_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not GENERATION_MODEL or not OPENAI_API_KEY:
    raise RuntimeError(
        "Faltan variables de entorno: define OPENAI_MODEL y OPENAI_API_KEY en tu .env"
    )

client_llm = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """Eres un analista financiero que responde preguntas basándote ÚNICAMENTE en los \
fragmentos de documentos que se te proporcionan a continuación.

Reglas estrictas:
1. No uses conocimiento externo ni asumas cifras que no estén literalmente en los fragmentos.
2. Si la información no alcanza para responder con certeza, dilo explícitamente en vez de inventar un número.
3. Al final de tu respuesta, cita la fuente en el formato: [Fuente: NOMBRE_DOCUMENTO, página X], \
usando el nombre de documento y la página que aparecen en la cabecera del fragmento que utilizaste."""


def format_context(chunks):
    # Serializa los chunks recuperados en un solo bloque de texto para el prompt.
    # Cada fragmento lleva una cabecera numerada con doc_name y page_num: es la
    # info que el system prompt le pide al modelo para construir la cita
    # [Fuente: NOMBRE_DOCUMENTO, página X].
    bloques = [
        f"[{i}] (documento: {c['doc_name']}, página: {c['page_num']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(bloques)


def generate_answer(question, chunks):
    # Paso final del RAG: pide al LLM una respuesta anclada solo en los chunks
    # ya recuperados y reordenados.
    if not chunks:
        return "No se recuperó ningún fragmento para responder la pregunta."

    # Arma el mensaje del usuario: contexto (fragmentos) + la pregunta.
    context = format_context(chunks)
    user_prompt = f"Fragmentos:\n\n{context}\n\nPregunta: {question}"

    # Llamada al Chat Completions API. SYSTEM_PROMPT fija las reglas.
    # El tope de tokens acota el coste y la longitud de la respuesta.
    try:
        response = client_llm.chat.completions.create(
            model=GENERATION_MODEL,
            max_completion_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except OpenAIError as e:
        # Fallo de red/API/credenciales
        return f"Error al llamar al modelo de generación: {e}"

    return response.choices[0].message.content
