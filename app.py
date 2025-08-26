import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# ==============================
# 🎨 Estilo WhatsApp con CSS
# ==============================
st.markdown(
    """
    <style>
    /* Fondo general */
    .main {
        background-color: #ece5dd;
    }

    /* Mensajes */
    .user-msg {
        background-color: #dcf8c6;
        color: black;
        border-radius: 10px;
        padding: 8px;
        margin: 5px;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .bot-msg {
        background-color: white;
        color: black;
        border-radius: 10px;
        padding: 8px;
        margin: 5px;
        max-width: 70%;
        float: left;
        clear: both;
        border: 1px solid #ddd;
    }

    /* Texto negro en inputs */
    input, textarea {
        color: black !important;
    }

    /* Caja de mensajes */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        background-color: #fafafa;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# 🎯 Configuración general
# ==============================
st.set_page_config(page_title="Agente de Biología 🧬", page_icon="🧬", layout="wide")
st.title("🧬 Agente Experto en Biología")
st.write("Haz preguntas sobre **biología** y recibe respuestas claras. 🌱🔬")

# ==============================
# 🔑 API Key
# ==============================
api_key = st.text_input("Introduce tu API Key de Groq:", type="password")

# ==============================
# ⚙️ Selección de modelo
# ==============================
model_choice = st.selectbox(
    "Elige el modelo de LLM:",
    ["llama3-8b-8192", "llama3-70b-8192"],
    index=0
)

# ==============================
# 🧠 Configuración de memoria
# ==============================
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "conversation" not in st.session_state and api_key:
    llm = ChatGroq(
        groq_api_key=api_key,
        model=model_choice,
        temperature=0.3,
        max_tokens=512,
    )
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False
    )

# ==============================
# 🧹 Botones de limpieza
# ==============================
col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Limpiar historial"):
        st.session_state.memory.clear()
        if "conversation" in st.session_state:
            st.session_state.conversation.memory.clear()
        st.session_state.history = []
        st.success("Historial limpiado.")

with col2:
    if st.button("🗑️ Limpiar pregunta"):
        st.session_state.input_box = ""  # 🔥 Solo borra la caja de entrada
        st.experimental_rerun()

# ==============================
# 📜 Caja de chat estilo WhatsApp
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("### 💬 Chat")
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# 📝 Entrada de usuario
# ==============================
user_input = st.text_input("Escribe tu pregunta sobre biología:", key="input_box")

if user_input and api_key:
    if "conversation" in st.session_state:
        response = st.session_state.conversation.predict(input=user_input)

        # Guardar mensajes estilo chat
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": response})

        st.session_state.input_box = ""  # 🔥 limpia automáticamente después de enviar
        st.experimental_rerun()
