import streamlit as st
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Configuración de la página
st.set_page_config(page_title="Agente de Biología", page_icon="🧬", layout="centered")

# Encabezado
st.markdown(
    """
    <h1 style="text-align: center; color: green;">
        🧬 Asistente Virtual de Biología
    </h1>
    <p style="text-align: center;">
        Aprende sobre conceptos biológicos, identificación de especies y procesos de la vida.
    </p>
    """,
    unsafe_allow_html=True
)

# Entrada de API Key
groq_api_key = st.text_input(
    "Introduce tu GROQ API Key:",
    type="password",
    help="Puedes obtener tu API Key en https://console.groq.com/"
)

# Inicializar memoria para historial de conversación
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)

# Inicializar caché de respuestas
if "cache" not in st.session_state:
    st.session_state["cache"] = {}

# Inicializar modelo si hay API Key
if groq_api_key:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama3-8b-8192",
        temperature=0.3,       # más estable y rápido
        max_tokens=512,
        streaming=True         # activa respuesta en streaming
    )

    # Prompt general restringido a biología
    system_prompt = (
        "Eres un experto en biología. "
        "Solo debes responder preguntas relacionadas con biología, "
        "como conceptos básicos, identificación de especies a partir de descripciones, "
        "procesos celulares, fisiológicos, genéticos, evolutivos o ecológicos. "
        "Si el usuario pregunta algo fuera de biología, responde: "
        "'Lo siento, solo puedo responder preguntas relacionadas con biología'."
    )

    chain = ConversationChain(
        llm=llm,
        memory=st.session_state["memory"],
        verbose=False
    )

    # Preguntas sugeridas
    ejemplos = [
        "¿Cuál es la diferencia entre mitosis y meiosis?",
        "Explica el proceso de fotosíntesis.",
        "¿Qué características permiten identificar a un mamífero?",
        "Describe cómo funciona la cadena trófica en un ecosistema.",
        "¿Cómo se diferencian las bacterias de los virus?",
        "Dibuja la estructura básica de una célula eucariota."
    ]

    st.subheader("🔍 Haz tu pregunta de biología")

    ejemplo_seleccionado = st.selectbox(
        "Elige un ejemplo de pregunta o escribe la tuya:",
        ["---"] + ejemplos
    )

    pregunta_usuario = st.text_area("Escribe aquí tu consulta:", height=120)

    if ejemplo_seleccionado != "---":
        pregunta_usuario = ejemplo_seleccionado

    if st.button("Enviar", type="primary"):
        if pregunta_usuario.strip():

            # Revisión en caché para optimizar rendimiento
            if pregunta_usuario in st.session_state["cache"]:
                respuesta = st.session_state["cache"][pregunta_usuario]
            else:
                with st.spinner("Analizando tu pregunta... 🌱"):
                    respuesta = chain.run(f"{system_prompt}\n\nPregunta: {pregunta_usuario}")
                    # Guardar en caché
                    st.session_state["cache"][pregunta_usuario] = respuesta

            st.markdown("### 📖 Respuesta:")
            st.write(respuesta)

            # Mostrar historial de conversación
            with st.expander("📜 Historial de conversación"):
                for msg in st.session_state["memory"].chat_memory.messages:
                    role = "👤 Usuario" if msg.type == "human" else "🤖 Agente"
                    st.markdown(f"**{role}:** {msg.content}")

            # Visualizaciones según la temática
            if "fotosíntesis" in pregunta_usuario.lower():
                st.subheader("🌞 Visualización del proceso de fotosíntesis")
                fig, ax = plt.subplots(figsize=(6,4))
                ax.text(0.1, 0.8, "🌞 Luz solar", fontsize=12)
                ax.text(0.1, 0.6, "🌿 Cloroplasto", fontsize=12)
                ax.text(0.1, 0.4, "CO₂ + H₂O", fontsize=12)
                ax.text(0.1, 0.2, "➡ Glucosa + O₂", fontsize=12)
                ax.set_axis_off()
                st.pyplot(fig)

            elif "cadena trófica" in pregunta_usuario.lower():
                st.subheader("🌍 Cadena trófica simplificada")
                niveles = ["☘️ Productores", "🐇 Consumidores primarios", "🦅 Consumidores secundarios", "🍄 Descomponedores"]
                fig, ax = plt.subplots(figsize=(6,4))
                for i, nivel in enumerate(niveles):
                    ax.text(0.5, 1 - i*0.25, nivel, fontsize=12, ha="center")
                    if i < len(niveles)-1:
                        ax.arrow(0.5, 1 - i*0.25 - 0.05, 0, -0.12,
                                 head_width=0.05, head_length=0.05, fc="black", ec="black")
                ax.set_axis_off()
                st.pyplot(fig)

            elif "célula" in pregunta_usuario.lower():
                st.subheader("🔬 Esquema básico de una célula eucariota")
                fig, ax = plt.subplots(figsize=(5,5))
                circle = plt.Circle((0.5,0.5), 0.4, fill=False, linewidth=2)
                ax.add_patch(circle)
                nucleo = plt.Circle((0.5,0.5), 0.15, fill=False, color="blue", linewidth=2)
                ax.add_patch(nucleo)
                ax.text(0.5, 0.5, "Núcleo", fontsize=10, ha="center", va="center", color="blue")
                ax.text(0.8, 0.5, "Membrana\ncelular", fontsize=9, ha="center", va="center")
                ax.text(0.5, 0.2, "Citoplasma", fontsize=9, ha="center", va="center")
                ax.set_xlim(0,1)
                ax.set_ylim(0,1)
                ax.set_aspect("equal")
                ax.axis("off")
                st.pyplot(fig)

        else:
            st.warning("Por favor ingresa una pregunta antes de enviar.")
else:
    st.info("🔑 Por favor, introduce tu GROQ API Key para comenzar.")

# Pie de página
st.markdown(
    """
    <hr>
    <p style="text-align: center; font-size: 14px; color: gray;">
        🌿 Desarrollado con LangChain + Groq + Llama3-8B-8192 <br>
        Optimizado para aprendizaje en biología.
    </p>
    """,
    unsafe_allow_html=True
)
