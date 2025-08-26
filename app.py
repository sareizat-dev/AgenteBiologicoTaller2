import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ---------------------------
# Configuración inicial
# ---------------------------
st.set_page_config(page_title="Agente de Biología 🧬", page_icon="🌱", layout="wide")

st.markdown(
    """
    <style>
    body {background-color: #f0fff4;}
    .stTextInput > div > div > input {
        background-color: #e6ffe6;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌱 Agente Virtual de Biología")
st.caption("Responde únicamente preguntas relacionadas con biología: especies, procesos biológicos y conceptos básicos.")

# ---------------------------
# API Key
# ---------------------------
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

def set_api_key():
    st.session_state.groq_api_key = st.session_state.key_input

if not st.session_state.groq_api_key:
    with st.container():
        st.subheader("🔑 Configuración")
        st.text_input("Ingresa tu API Key de Groq:", type="password", key="key_input")
        st.button("Guardar API Key", on_click=set_api_key)
    st.stop()

# ---------------------------
# Modelo seleccionado
# ---------------------------
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3-8b-8192"  # por defecto

def load_chain():
    """Carga el LLM y la cadena conversacional según el modelo elegido"""
    llm = ChatGroq(
        groq_api_key=st.session_state.groq_api_key,
        model_name=st.session_state.selected_model,
        temperature=0.3,
        max_tokens=512,
    )
    return ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        prompt=prompt_template,
        verbose=False,
    )

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
# Inicializar cadena conversacional
# ---------------------------
if "chain" not in st.session_state:
    st.session_state.chain = load_chain()

# ---------------------------
# Inicializar historial
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Layout con tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["💬 Conversación", "⚙️ Configuración", "🧠 Modelos"])

with tab1:
    st.subheader("Chat de Biología 🧬")

    # Input del usuario
    if "input_widget" not in st.session_state:
        st.session_state.input_widget = ""

    def limpiar_input():
        st.session_state.input_widget = ""

    def borrar_historial():
        st.session_state.messages = []
        st.session_state.memory.clear()

    user_input = st.text_input(
        "Pregunta de biología:",
        key="input_widget",
        placeholder="Ejemplo: ¿Cómo funciona la fotosíntesis?"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.button("🧹 Limpiar caja", on_click=limpiar_input)
    with col2:
        st.button("🗑️ Borrar historial", on_click=borrar_historial)

    # Procesar respuesta
    if user_input:
        if len(st.session_state.messages) == 0 or user_input != st.session_state.messages[-1]["user"]:
            response = st.session_state.chain.run(user_input)
            st.session_state.messages.append({"user": user_input, "bot": response})

    # Mostrar historial estilo chat
    if st.session_state.messages:
        for msg in st.session_state.messages:
            with st.chat_message("user", avatar="👩‍🔬"):
                st.markdown(msg["user"])
            with st.chat_message("assistant", avatar="🧬"):
                st.markdown(msg["bot"])

with tab2:
    st.subheader("⚙️ Configuración")
    st.write("Aquí puedes volver a ingresar tu API Key si lo deseas.")
    st.text_input("Reingresar API Key:", type="password", key="re_key_input")
    if st.button("Actualizar API Key"):
        st.session_state.groq_api_key = st.session_state.re_key_input
        st.success("✅ API Key actualizada correctamente")

with tab3:
    st.subheader("🧠 Selección de Modelo LLM")
    modelo = st.radio(
        "Elige el modelo de Groq:",
        ["llama3-8b-8192", "llama3-70b-8192"],
        index=0 if st.session_state.selected_model == "llama3-8b-8192" else 1
    )

    if modelo != st.session_state.selected_model:
        st.session_state.selected_model = modelo
        st.session_state.chain = load_chain()
        st.success(f"✅ Modelo actualizado: {modelo}")
