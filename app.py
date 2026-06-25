import os
import tempfile
import streamlit as st
import fitz  # PyMuPDF

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# ─────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Asistente Universitario",
    page_icon="🎓",
    layout="wide"
)

COLLECTION_NAME = "reglamento_universitario"

# ─────────────────────────────────────────
# ESTADO DE SESIÓN
# ─────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "historial" not in st.session_state:
    st.session_state.historial = []
if "docs_cargados" not in st.session_state:
    st.session_state.docs_cargados = []

# ─────────────────────────────────────────
# EMBEDDER
# ─────────────────────────────────────────
@st.cache_resource
def cargar_embedder():
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

# ─────────────────────────────────────────
# PDF
# ─────────────────────────────────────────
def extraer_texto_pdf(ruta):
    doc = fitz.open(ruta)
    texto = ""
    for pagina in doc:
        t = pagina.get_text("text").strip()
        if len(t) > 50:
            texto += t + "\n\n"
    doc.close()
    return texto.strip()

# ─────────────────────────────────────────
# CHROMA (FIX DEFINITIVO)
# ─────────────────────────────────────────
def procesar_documentos(archivos):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    embedder = cargar_embedder()

    textos = []
    metadatos = []

    for archivo in archivos:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(archivo.read())
            tmp_path = tmp.name

        texto = extraer_texto_pdf(tmp_path)
        chunks = splitter.split_text(texto)

        for i, chunk in enumerate(chunks):
            textos.append(chunk)
            metadatos.append({
                "fuente": archivo.name,
                "chunk_id": str(i),
                "tipo": "reglamento_universitario"
            })

        os.unlink(tmp_path)

    # 🔴 FIX CLAVE: sin persist_directory (evita 1032)
    vectorstore = Chroma.from_texts(
        texts=textos,
        embedding=embedder,
        metadatas=metadatos,
        collection_name=COLLECTION_NAME
    )

    return vectorstore

# ─────────────────────────────────────────
# AGENTE
# ─────────────────────────────────────────
def consultar_agente(pregunta, vectorstore, api_key):

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key
    )

    fuentes = vectorstore.similarity_search(pregunta, k=8)

    contexto = "\n\n".join([
        f"[Fragmento {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(fuentes)
    ])

    mensaje = HumanMessage(content=f"""
Eres un asistente universitario.
Responde en español, claro, sin listas.

Usa SOLO el contexto:

{contexto}

Pregunta: {pregunta}
""")

    respuesta = llm.invoke([mensaje])

    return respuesta.content, fuentes

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Asistente Universitario")
    st.markdown("---")

    api_key = st.text_input(
        "GROQ API Key",
        value="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",
        type="password"
    )

    st.markdown("---")

    archivos = st.file_uploader(
        "Sube PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if archivos and st.button("Procesar documentos"):
        with st.spinner("Procesando..."):
            try:
                st.session_state.vectorstore = procesar_documentos(archivos)
                st.session_state.docs_cargados = [a.name for a in archivos]
                st.session_state.historial = []
                st.success("Documentos procesados correctamente")
            except Exception as e:
                st.error(f"Error: {e}")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
st.title("🎓 Asistente de Reglamentos")

if not api_key:
    st.warning("Ingresa API Key")
    st.stop()

if st.session_state.vectorstore is None:
    st.info("Sube PDFs para comenzar")
    st.stop()

for msg in st.session_state.historial:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])

if pregunta := st.chat_input("Pregunta algo..."):
    st.chat_message("user").markdown(pregunta)

    respuesta, fuentes = consultar_agente(
        pregunta,
        st.session_state.vectorstore,
        api_key
    )

    with st.chat_message("assistant"):
        st.markdown(respuesta)

    st.session_state.historial.append({"rol": "user", "contenido": pregunta})
    st.session_state.historial.append({"rol": "assistant", "contenido": respuesta})