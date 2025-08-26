import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ---------------------------
# Configuración inicial
# ---------------------------
st.set_page_config(page_title="Agente de Biología 🧬", page_icon="🧬", layout="centered")

st.title("🧬 Agente Virtual de Biología")
st.write("Este agente responde **únicamente preguntas relacionadas con biología**.")

# ---------------------------
# API Key de Groq
# ---------------------------
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

def set_api_key():
    st.session_state.groq_api_key = st.session_state.key_input

if not st.session_state.groq_api_key:
    st.text_input("🔑 Ingresa tu API Key de Groq:", type="password", key="key_input")
    st.button("Guardar API Key", on_click=set_api_key)
    st.stop()

# ---------------------------
# Inicializar memoria
# ---------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="input", return_messages=True
    )

# ---------------------------
# Prompt restringido a biología
# ---------------------------
prompt_template = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=(
        "Eres un experto en biología. Responde únicamente preguntas relacionadas "
        "con biología: conceptos básicos, identificación de especies, procesos biológicos. "
        "Si la pregunta no está relacionada con biología, responde: "
        "'Lo siento, solo puedo responder preguntas de biología.'\n\n"
        "Historial de conversación:\n{chat_history}\n\n"
        "Pregunta: {input}\nRespuesta:"
    ),
)

# ---------------------------
# Modelo Groq (llama3-8b-8192)
# ---------------------------
llm = ChatGroq(
    groq_api_key=st.session_state.groq_api_key,
    model_name="llama3-8b-8192",
    temperature=0.3,
    max_tokens=512,
)

chain = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    prompt=prompt_template,
    verbose=False,
)

# ---------------------------
# Historial de conversación
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Definir input box con key fijo
# ---------------------------
if "input_widget" not in st.session_state:
    st.session_state.input_widget = ""

def limpiar_input():
    st.session_state.input_widget = ""

def borrar_historial():
    st.session_state.messages = []
    st.session_state.memory.clear()

user_input = st.text_input(
    "💬 Escribe tu pregunta de biología:",
    key="input_widget"
)

col1, col2 = st.columns(2)
with col1:
    st.button("🧹 Limpiar caja", on_click=limpiar_input)
with col2:
    st.button("🗑️ Borrar historial", on_click=borrar_historial)

# ---------------------------
# Procesar respuesta
# ---------------------------
if user_input:
    if len(st.session_state.messages) == 0 or user_input != st.session_state.messages[-1]["user"]:
        response = chain.run(user_input)
        st.session_state.messages.append({"user": user_input, "bot": response})

# ---------------------------
# Mostrar historial
# ---------------------------
if st.session_state.messages:
    st.subheader("📜 Historial de conversación")
    for msg in st.session_state.messages:
        st.markdown(f"**👤 Tú:** {msg['user']}")
        st.markdown(f"**🤖 Agente:** {msg['bot']}")
