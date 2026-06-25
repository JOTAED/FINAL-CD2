import os
import shutil
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

CHROMA_PATH     = "./chroma_app"
COLLECTION_NAME = "reglamento_universitario"

# ─────────────────────────────────────────
# ESTADO DE SESIÓN
# ─────────────────────────────────────────
if "vectorstore"   not in st.session_state: st.session_state.vectorstore   = None
if "historial"     not in st.session_state: st.session_state.historial     = []
if "docs_cargados" not in st.session_state: st.session_state.docs_cargados = []

# ─────────────────────────────────────────
# EMBEDDER — mismo modelo que notebook 02
# ─────────────────────────────────────────
@st.cache_resource
def cargar_embedder():
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

# ─────────────────────────────────────────
# EXTRACCIÓN PDF — mismo que notebook 01
# ─────────────────────────────────────────
def extraer_texto_pdf(ruta):
    doc   = fitz.open(ruta)
    texto = ""
    for pagina in doc:
        t = pagina.get_text("text").strip()
        if len(t) > 50:
            texto += t + "\n\n"
    doc.close()
    return texto.strip()

# ─────────────────────────────────────────
# CHUNKING + CHROMADB — mismo que notebook 03/04
# ─────────────────────────────────────────
def procesar_documentos(archivos):
    # Mismo splitter que notebook 01
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    embedder = cargar_embedder()

    # Limpiar colección anterior
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    textos    = []
    metadatos = []

    for archivo in archivos:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(archivo.read())
            tmp_path = tmp.name

        texto  = extraer_texto_pdf(tmp_path)
        chunks = splitter.split_text(texto)

        for i, chunk in enumerate(chunks):
            textos.append(chunk)
            metadatos.append({
                "fuente":   archivo.name,
                "chunk_id": str(i),
                "tipo":     "reglamento_universitario"
            })

        os.unlink(tmp_path)

    # Mismo Chroma.from_texts que notebook 04
    vectorstore = Chroma.from_texts(
        texts=textos,
        embedding=embedder,
        metadatas=metadatos,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vectorstore

# ─────────────────────────────────────────
# HERRAMIENTAS — mismas que notebook 05
# ─────────────────────────────────────────
def crear_herramientas(vectorstore, llm):

    @tool
    def buscar_reglamento(query: str) -> str:
        """
        Busca información relevante en el reglamento universitario.
        Usar cuando el estudiante haga preguntas sobre normas,
        artículos o políticas de la universidad.
        """
        docs = vectorstore.similarity_search(query, k=8)
        if not docs:
            return "No se encontró información relevante en el reglamento."
        return "\n\n".join([
            f"[Fragmento {i+1} — {doc.metadata.get('fuente','?')}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])

    @tool
    def generar_resumen(texto: str) -> str:
        """
        Resume un texto del reglamento en párrafos naturales en español.
        Usar cuando el estudiante pida explicación simple de un tema.
        """
        respuesta = llm.invoke(
            f"""Explica el siguiente fragmento del reglamento universitario
en un párrafo fluido y natural, como si se lo explicaras a un compañero estudiante.
Sin bullet points, solo texto corrido y claro en español:

{texto}"""
        )
        return respuesta.content

    return [buscar_reglamento, generar_resumen]

# ─────────────────────────────────────────
# AGENTE — mismo flujo que notebook 05
# ─────────────────────────────────────────
def consultar_agente(pregunta, vectorstore, api_key):
    # Mismo LLM que notebook 05
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key
    )

    herramientas  = crear_herramientas(vectorstore, llm)
    tool_map      = {t.name: t for t in herramientas}

    # Recuperar fuentes para mostrarlas en la UI
    fuentes = vectorstore.similarity_search(pregunta, k=8)

    contexto = "\n\n".join([
        f"[Fragmento {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(fuentes)
    ])

    # Mismo prompt que notebook 05
    mensajes = [HumanMessage(content=f"""Eres un asistente universitario amigable.
Responde siempre en español, en párrafos naturales y fluidos, como si hablaras con un estudiante.
No uses listas ni bullet points en tu respuesta final.
Usa ÚNICAMENTE la información del contexto proporcionado.
Si la respuesta no está en el contexto, di: "No encontré esa información en el reglamento."
NO inventes información que no esté en el contexto.

Contexto del reglamento:
{contexto}

Pregunta: {pregunta}""")]

    respuesta = llm.invoke(mensajes)
    return respuesta.content, fuentes

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Asistente Universitario")
    st.markdown("---")

    api_key = st.text_input(
        "GROQ API Key",
        value= "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",
        type="password",
        help="Obtén tu key gratis en console.groq.com"
    )

    st.markdown("---")

    st.subheader("📄 Cargar Documentos PDF")
    st.caption("Máximo 60 páginas por documento")

    archivos = st.file_uploader(
        "Selecciona uno o más PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if archivos and st.button("⚙️ Procesar documentos", use_container_width=True):
        with st.spinner("Extrayendo texto y generando embeddings..."):
            try:
                st.session_state.vectorstore   = procesar_documentos(archivos)
                st.session_state.docs_cargados = [a.name for a in archivos]
                st.session_state.historial     = []
                total = st.session_state.vectorstore._collection.count()
                st.success(f"✅ {len(archivos)} documento(s) procesado(s) — {total} chunks")
            except Exception as e:
                st.error(f"Error al procesar: {e}")

    if st.session_state.docs_cargados:
        st.markdown("**Documentos activos:**")
        for doc in st.session_state.docs_cargados:
            st.markdown(f"- 📄 {doc}")

    st.markdown("---")

    if st.button("🗑️ Limpiar chat", use_container_width=True):
        st.session_state.historial = []
        st.rerun()

# ─────────────────────────────────────────
# PANTALLA PRINCIPAL
# ─────────────────────────────────────────
st.title("🎓 Asistente de Reglamentos Universitarios")
st.markdown("Carga un PDF y hazle preguntas en lenguaje natural.")

if not api_key:
    st.warning("⚠️ Ingresa tu GROQ API Key en el panel izquierdo.")
    st.stop()

if st.session_state.vectorstore is None:
    st.info("👈 Sube un PDF y presiona **Procesar documentos** para comenzar.")
    st.stop()

# Historial de chat
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])
        if mensaje["rol"] == "assistant" and mensaje.get("fuentes"):
            with st.expander("📚 Ver fuentes recuperadas"):
                for i, doc in enumerate(mensaje["fuentes"]):
                    st.markdown(f"**Fragmento {i+1}** — `{doc.metadata.get('fuente','?')}`")
                    st.text(doc.page_content[:400])
                    st.markdown("---")

# Input
if pregunta := st.chat_input("Escribe tu pregunta sobre el reglamento..."):
    with st.chat_message("user"):
        st.markdown(pregunta)
    st.session_state.historial.append({
        "rol": "user", "contenido": pregunta
    })

    with st.chat_message("assistant"):
        with st.spinner("Consultando el reglamento..."):
            try:
                respuesta, fuentes = consultar_agente(
                    pregunta,
                    st.session_state.vectorstore,
                    api_key
                )
                st.markdown(respuesta)
                with st.expander("📚 Ver fuentes recuperadas"):
                    for i, doc in enumerate(fuentes):
                        st.markdown(f"**Fragmento {i+1}** — `{doc.metadata.get('fuente','?')}`")
                        st.text(doc.page_content[:400])
                        st.markdown("---")

            except Exception as e:
                respuesta = f"Error: {e}"
                fuentes   = []
                st.error(respuesta)

    st.session_state.historial.append({
        "rol": "assistant", "contenido": respuesta, "fuentes": fuentes
    })