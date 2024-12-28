import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
import os
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
import openai
from urllib.parse import unquote



# Configura las claves de API desde los secretos
openai.api_key = st.secrets["general"]["openai_api_key"]
pinecone_api_key = st.secrets["general"]["pinecone_api_key"]
genai_api_key = st.secrets["general"]["genai_api_key"]

# Configuración de la página
st.set_page_config(page_title="Chatbot Psoriasis", layout="wide")

# Configurar Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("embedding-psoriasis-large")



# Generar embeddings con OpenAI
def generar_embeddings(query):
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-3-large"  # Ajustar si procede
    )
    embedding_vector = response['data'][0]['embedding']
    return [embedding_vector]


# Configurr modelo genai
genai.configure(api_key=genai_api_key)
model = genai.GenerativeModel("gemini-1.5-flash-latest")


st.title("¿Tiene alguna consulta sobre el tratamiento? ¡Cuéntanos!")

params = st.query_params

formulario_list = params.get("formulario", [])
tratamiento_list = params.get("tratamiento", [])

# Si viene vacío, asignamos None
formulario_codificado = formulario_list[0] if formulario_list else None
tratamiento_codificado = tratamiento_list[0] if tratamiento_list else None

# Decodificar los valores si existen, o asignar valores por defecto
if formulario_codificado:
    texto_formulario = unquote(formulario_codificado)
else:
    texto_formulario = "No se ha proporcionado formulario."

if tratamiento_codificado:
    texto_tratamiento = unquote(tratamiento_codificado)
else:
    texto_tratamiento = "No se ha proporcionado tratamiento."

# Debugging en consola
print("El texto del formulario es: ", texto_formulario)
print("El texto del tratamiento es: ", texto_tratamiento)

# Iniciar el estado de la sesión
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Mostrar el historial de mensajes (en la parte superior)
st.markdown("### Historial de Conversación:")
for i, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")
    else:
        message(msg["content"], is_user=False, key=f"assistant_{i}")

# Inicializar el estado para el campo de entrada si no está definido
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# Campo de entrada de texto (en la parte inferior)
st.markdown("---")  # Separador visual
query = st.text_input(
    "",
    value=st.session_state["user_input"],
    key="user_input",
    placeholder="Escriba aquí su mensaje..."
)

# Botón para enviar
if st.button("Enviar"):
    if query.strip():  # Verificar que el campo no esté vacío
        # Agregar la consulta del usuario al historial
        st.session_state["messages"].append({"role": "user", "content": query})

        with st.spinner("Su respuesta se está generando..."):
            # Generar embeddings y buscar en Pinecone
            vector = generar_embeddings(query)
            filter_conditions = {
                "$and": [
                    {"texto": {"$exists": True}},
                    {"fuente": {"$exists": False}}
                ]
            }

            resultados = index.query(
                vector=vector,
                top_k=5,
                include_values=True,
                include_metadata=True,
                filter=filter_conditions
            )

            textos_similares = [res["metadata"]["texto"] for res in resultados["matches"]]
            fragmentos_recuperados = " ".join(textos_similares)

            # Crear el prompt para el modelo generativo
            prompt = f"""
            Eres un chatbot inteligente especializado en Dermatología, concretamente en Psoriasis.
            Tu misión es ayudar a los dermatólogos profesionales con las dudas o consultas que tengan acerca de tratamientos de Psoriasis.
            Recibirás la siguiente información:
            - Formulario con los datos del paciente
            - Tratamiento generado para el paciente
            - Fuente de datos que usarás para responder a la consulta del dermatólogo.
            - Consulta del dermatólogo. 

            El formulario con los datos del paciente es el siguiente: {texto_formulario}.

            El tratamiento generado para el paciente es el siguiente: {texto_tratamiento}.

            La fuente de datos que usarás para responder a la consulta del paciente es: {fragmentos_recuperados}.

            La consulta del dermatólogo, la cual debes responder es la siguiente: {query}.

            Por favor, responde a la consulta del dermatólogo. Debes ser amable, claro y preciso, no te inventes ninguna información y se fiel a la información que recibes.
            Si consideras que la información que recibes no es suficiente para responder la consulta, comunícaselo al dermatólogo.
            Si consideras que la consulta del dermatólogo no es clara o no sabes su intención, no dudes en preguntarle de nuevo.

            No te presentes ni te despidas en el mensaje, se directo, recuerda que eres un chatbot.

            Si lo haces bien serás recompensado.
            """

            response = model.generate_content(prompt)
            respuesta = response.candidates[0].content.parts[0].text

        # Una vez finalizado el with st.spinner(), se quita el mensaje y se muestra la respuesta
        st.session_state["messages"].append({"role": "assistant", "content": respuesta})
        st.rerun()

