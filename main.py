import streamlit as st
import time
from chatbot.rag import ChatBot

# ------------------------------------------------------------------------------
# Inicialización del chatbot (una sola vez por sesión)
# ------------------------------------------------------------------------------
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatBot()

chat = st.session_state.chatbot

# ------------------------------------------------------------------------------
# Función generadora que simula la escritura palabra por palabra
# ------------------------------------------------------------------------------

def response_generator(user_input: str, bot_context: str):
    """
    Genera la respuesta del chatbot palabra por palabra con una breve pausa.

    Args:
        user_input (str): Pregunta del usuario.
        bot_context (str): Contexto adicional para el modelo (actualmente no usado).

    Yields:
        str: Palabras de la respuesta separadas por pausas.
    """
    response = chat.llama_response(user_input)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# ------------------------------------------------------------------------------
# Interfaz de usuario con Streamlit
# ------------------------------------------------------------------------------

# Título principal de la app
st.title("Chat DeNexus")

# Inicializar historial de mensajes si aún no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar el historial de mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura de entrada del usuario
if prompt := st.chat_input("¿Qué quieres preguntar?", key=2):
    # Guardar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Mostrar el mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar y mostrar la respuesta del asistente
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt, ""))
    st.session_state.messages.append({"role": "assistant", "content": response})
