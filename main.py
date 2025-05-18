import time
import streamlit as st
from chatbot.rag import ChatBot
from log_config import get_logstash_logger
logger = get_logstash_logger("streamlit-chatbot")

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
    response = chat.llama_response(user_input)

    # Loggear la interacción
    logger.info(f"User input: {user_input}")
    logger.info(f"Bot response: {response}")

    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# ------------------------------------------------------------------------------
# Interfaz de usuario con Streamlit
# ------------------------------------------------------------------------------

st.title("Chat DeNexus")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué quieres preguntar?", key=2):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt, ""))
    st.session_state.messages.append({"role": "assistant", "content": response})
