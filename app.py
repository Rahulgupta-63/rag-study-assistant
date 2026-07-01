import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from text import Groq
from bs4 import BeautifulSoup
import pdfplumber
from dotenv import load_dotenv

load_dotenv()

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Study Assistant",
    page_icon="🧠",
    layout="centered"
)

# ============================================
# STYLING
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

* { font-family: 'Syne', sans-serif; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1419;
    color: #e4e6eb;
}

[data-testid="stSidebar"] {
    background-color: #1a1f2e;
    border-right: 1px solid #2a3447;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.chat-user {
    background: #1a1f2e;
    border-left: 3px solid #2ecc71;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

.chat-assistant {
    background: #0d1117;
    border-left: 3px solid #3498db;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 0.9rem;
}

.status-box {
    background: rgba(46, 204, 113, 0.1);
    border: 1px solid #2ecc71;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #2ecc71;
    font-family: 'Space Mono', monospace;
}

.stButton > button {
    background: #2ecc71;
    color: #0f1419;
    border: none;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
}

.stButton > button:hover {
    background: #27ae60;
    color: #0f1419;
}

[data-testid="stTextInput"] input {
    background: #1a1f2e;
    border: 1px solid #2a3447;
    color: #e4e6eb;
    font-family: 'Space Mono', monospace;
    border-radius: 6px;
}

[data-testid="stFileUploader"] {
    background: #1a1f2e;
    border: 1px dashed #2a3447;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# INIT SESSION STATE
# ============================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "notes_loaded" not in st.session_state:
    st.session_state.notes_loaded = False
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

# ============================================
# LOAD MODELS (cached)
# ============================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Set it in your environment or Hugging Face Secrets.")
        st.stop()
    return Groq(api_key=api_key)

# ============================================
# FILE PARSERS
# ============================================
def load_html(content):
    soup = BeautifulSoup(content, "html.parser")
    chunks = []
    for code in soup.find_all(["pre", "code"]):
        text = code.get_text(separator=" ").strip()
        if len(text) > 30:
            chunks.append(text)
    for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]):
        text = tag.get_text(separator=" ").strip()
        if len(text) > 50:
            chunks.append(text)
    return chunks

def load_pdf(file):
    chunks = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.split("\n"):
                    if len(line.strip()) > 50:
                        chunks.append(line.strip())
    return chunks

def load_txt(content):
    lines = content.decode("utf-8").split("\n")
    return [l.strip() for l in lines if len(l.strip()) > 50]

# ============================================
# CHROMADB SETUP
# ============================================
def setup_collection(chunks):
    embedding_model = load_embedding_model()
    client = chromadb.Client()  # in-memory for Hugging Face Spaces
    
    try:
        client.delete_collection("study_notes")
    except:
        pass
    
    collection = client.create_collection("study_notes")
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"note{i}" for i in range(len(chunks))]
    
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection

# ============================================
# RAG FUNCTIONS
# ============================================
def retrieve(question, collection):
    embedding_model = load_embedding_model()
    q_embedding = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[q_embedding], n_results=10)
    return results['documents'][0]

def study_assistant(question, collection):
    groq_client = get_groq_client()
    relevant_notes = retrieve(question, collection)
    context = "\n".join(relevant_notes)

    st.session_state.chat_history.append({"role": "user", "content": question})

    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful study assistant.
Answer ONLY from the notes below.
If the answer is not in the notes, say: I don't have that in my notes.

Notes:
{context}"""
        }
    ] + st.session_state.chat_history

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    answer = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    return answer

# ============================================
# UI
# ============================================
st.markdown("# 🧠 Study Assistant")
st.markdown("Upload your notes. Ask anything. Get answers *from your own material*.")
st.divider()

# Sidebar — file upload
with st.sidebar:
    st.markdown("### 📂 Upload Your Notes")
    uploaded_file = st.file_uploader(
        "Supports HTML, PDF, TXT",
        type=["html", "pdf", "txt"]
    )

    if uploaded_file and not st.session_state.notes_loaded:
        with st.spinner("Processing notes..."):
            if uploaded_file.name.endswith(".html"):
                chunks = load_html(uploaded_file.read())
            elif uploaded_file.name.endswith(".pdf"):
                chunks = load_pdf(uploaded_file)
            else:
                chunks = load_txt(uploaded_file.read())

            if chunks:
                st.session_state.collection = setup_collection(chunks)
                st.session_state.notes_loaded = True
                st.markdown(f'<div class="status-box">✅ {len(chunks)} chunks loaded</div>', unsafe_allow_html=True)
            else:
                st.error("No content extracted. Check your file.")

    if st.session_state.notes_loaded:
        st.markdown('<div class="status-box">🟢 Notes ready</div>', unsafe_allow_html=True)
        if st.button("Clear & Upload New"):
            st.session_state.notes_loaded = False
            st.session_state.collection = None
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown("**Built by Rahul Gupta**")
    st.markdown("MCA — KNIT Sultanpur")

# Main — chat
if not st.session_state.notes_loaded:
    st.info("👈 Upload your notes from the sidebar to get started.")
else:
    # Chat history display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    question = st.text_input("Ask a question about your notes...", key="input")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask = st.button("Ask")
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    if ask and question:
        with st.spinner("Thinking..."):
            answer = study_assistant(question, st.session_state.collection)
        st.rerun()
