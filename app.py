import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Configuración de la página
st.set_page_config(page_title="🧬 Agente de Biología", page_icon="🧬", layout="wide")

st.title("🧬 Agente Virtual de Biología")
st.write("Este asistente responde **únicamente preguntas de biología**.")

# Campo para ingresar la API key
api_key = st.text_input("🔑 Ingresa tu API Key de Groq:", type="password")

if api_key:
    # Definir prompt restringido a biología
    prompt_template = """
    Eres un experto en biología. Responde de forma clara, educativa y detallada
    solo a preguntas de biología. Si la pregunta no está relacionada con biología,
    responde: "Lo siento, solo puedo responder preguntas sobre biología."

    Pregunta: {input}
    Respuesta:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["input"])

    # Configurar modelo Llama3 en Groq
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        temperature=0.3,
    )

    # Memoria de conversación (pero no mostramos el prompt completo en historial)
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationChain(
            llm=llm,
            prompt=PROMPT,
            memory=st.session_state.memory,
            verbose=False
        )
    if "history" not in st.session_state:
        st.session_state.history = []

    # Input del usuario
    col1, col2 = st.columns([4,1])
    with col1:
        user_input = st.text_input("💬 Haz una pregunta sobre biología:", key="input_box")
    with col2:
        if st.button("🧹 Limpiar"):
            st.session_state.input_box = ""  # Limpia la caja de texto

    if st.button("Enviar"):
        if user_input:
            response = st.session_state.conversation.predict(input=user_input)
            # Guardar en historial solo la pregunta y respuesta
            st.session_state.history.append({"pregunta": user_input, "respuesta": response})
            st.session_state.input_box = ""  # Limpia la caja después de enviar

    # Mostrar historial de conversación limpio
    st.subheader("📜 Historial de conversación")
    for chat in st.session_state.history:
        st.markdown(f"**👤 Usuario:** {chat['pregunta']}")
        st.markdown(f"**🧬 Agente:** {chat['respuesta']}")

    """,
    unsafe_allow_html=True
)
