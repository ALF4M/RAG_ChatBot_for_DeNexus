import streamlit as st
import time
from RAG import ChatBot

# Crear la instancia del chatbot solo una vez por sesión
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatBot()

chat = st.session_state.chatbot

# Función generadora de respuesta
def response_generator(userInput, botContext):
    response = chat.llamaResponse(userInput)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Título de la app
st.title("Chat DeNexus")

# Inicializar historial de mensajes si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de usuario
if prompt := st.chat_input("What do you want to ask?", key=2):
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar y mostrar respuesta del chatbot
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt, ""))
    st.session_state.messages.append({"role": "assistant", "content": response})
