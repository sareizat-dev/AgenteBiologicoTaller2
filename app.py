import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Configuración de la página
st.set_page_config(page_title="🧬 Agente de Biología", page_icon="🧬", layout="wide")

# Estilos CSS personalizados para darle temática biológica
st.markdown(
    """
    <style>
    body {
        background-color: #f0fff4;
    }
    .stTextInput > div > div > input {
        border: 2px solid #2f855a;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #38a169;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #2f855a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título
st.title("🧬 Agente Virtual de Biología")
st.write("Este asistente responde **únicamente preguntas de biología**. "
         "Puede ayudarte a entender conceptos, identificar especies por descripción "
         "y explicar procesos biológicos complejos.")

# Entrada de API Key
api_key = st.text_input("🔑 Ingresa tu API Key de Groq:", type="password")

if api_key:
    # Prompt especializado en biología
    prompt_template = """
    Eres un experto en biología. Responde de forma clara, educativa y detallada
    únicamente a preguntas de biología. 

    Si la pregunta no está relacionada con biología, responde estrictamente:
    "Lo siento, solo puedo responder preguntas sobre biología."

    Pregunta: {input}
    Respuesta:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["input"])

    # Inicializar el modelo Llama3 Groq
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        temperature=0.3,
        max_tokens=512,
    )

    # Manejo de memoria y conversación
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationChain(
            llm=llm,
            prompt=PROMPT,
            memory=st.session_state.memory,
            verbose=False
        )
    if "history" not in st.session_state:
        st.session_state.history = []

    # Layout para entrada de usuario
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        user_input = st.text_input("💬 Haz una pregunta sobre biología:", key="input_box")
    with col2:
        if st.button("🧹 Limpiar caja"):
            st.session_state.input_box = ""  # Limpia la caja
    with col3:
        if st.button("🗑️ Borrar historial"):
            st.session_state.history = []  # Limpia historial

    # Procesar pregunta
    if st.button("Enviar", type="primary"):
        if user_input.strip():
            with st.spinner("🧬 Analizando tu pregunta..."):
                response = st.session_state.conversation.predict(input=user_input)
            # Guardar solo pregunta y respuesta
            st.session_state.history.append(
                {"pregunta": user_input, "respuesta": response}
            )
            st.session_state.input_box = ""  # Limpia después de enviar
        else:
            st.warning("⚠️ Escribe una pregunta antes de enviar.")

    # Mostrar historial
    st.subheader("📜 Historial de conversación")
    for chat in st.session_state.history:
        st.markdown(f"**👤 Usuario:** {chat['pregunta']}")
        st.markdown(f"**🧬 Agente:** {chat['respuesta']}")
else:
    st.info("🔑 Ingresa tu API Key de Groq para comenzar.")

# Footer
st.caption("🌱 Powered by LangChain + Groq + Llama3-8B-8192")
