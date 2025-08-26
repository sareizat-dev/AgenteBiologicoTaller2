# 🧬 Agente Virtual de Biología

Este proyecto implementa un **agente conversacional especializado en biología**, desarrollado con **LangChain**, **Groq** y **Streamlit**, utilizando el modelo **Llama3-8B-8192** como LLM.

El asistente está diseñado para **responder únicamente preguntas relacionadas con biología**, tales como:  
- Conceptos básicos (ej. mitosis, fotosíntesis, genética).  
- Identificación de especies a partir de descripciones textuales.  
- Explicación de procesos biológicos complejos (ej. cadenas tróficas, funcionamiento de células).  

Si el usuario hace preguntas fuera del ámbito de la biología, el agente responde con un aviso indicando que solo puede atender consultas de biología.

---

## 🚀 Características principales

- **Interfaz en Streamlit** con temática de biología.  
- **Restricción temática:** solo responde sobre biología.  
- **Respuestas en streaming** → aparecen en tiempo real.  
- **Memoria de conversación** → conserva el historial del chat.  
- **Caché de respuestas** → reutiliza resultados para preguntas repetidas, mejorando la velocidad.  
- **Visualizaciones dinámicas** con gráficos sencillos en `matplotlib` para temas como:  
  - Proceso de fotosíntesis.  
  - Cadenas tróficas.  
  - Esquema de célula eucariota.  

---

## 📂 Archivos del proyecto

- `app.py` → Aplicación principal en Streamlit.  
- `requirements.txt` → Dependencias necesarias.  
- `runtime.txt` → Versión de Python utilizada (3.12).  

---

## 🔑 Requisitos previos

1. Tener **Python 3.12** instalado.  
2. Crear un entorno virtual (opcional, pero recomendado).  
3. Instalar las dependencias:  

```bash
pip install -r requirements.txt
