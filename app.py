import os
import uuid
import sqlite3
from pypdf import PdfReader
import chromadb
import ollama
import streamlit as st
from datetime import datetime, timedelta

# ==============================
# CONFIGURATION
# ==============================
DATABASE_FOLDER = "chroma_db"
KNOWLEDGE_FOLDER = "knowledge"
DB_FILE = "chat_history.db"

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
N_RESULTS = 7
MAX_DISTANCE = 1.6
BATCH_SIZE = 32

st.set_page_config(page_title="Local AI Assistant", page_icon="🤖", layout="wide")

# ==============================
# SQLITE CHAT HISTORY (6-MONTH RETENTION)
# ==============================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT,
            updated_at TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            chat_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP
        )
    ''')
    conn.commit()

    # Auto-delete messages & chats older than 180 days (6 months)
    six_months_ago = datetime.now() - timedelta(days=180)
    c.execute("DELETE FROM messages WHERE timestamp < ?", (six_months_ago,))
    c.execute("DELETE FROM chats WHERE updated_at < ?", (six_months_ago,))
    conn.commit()
    conn.close()

def save_message(chat_id, chat_title, role, content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    now = datetime.now()
    
    c.execute('''
        INSERT INTO chats (id, title, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET updated_at=?
    ''', (chat_id, chat_title, now, now))
    
    c.execute('''
        INSERT INTO messages (chat_id, role, content, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (chat_id, role, content, now))
    
    conn.commit()
    conn.close()

def get_all_chats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, title FROM chats ORDER BY updated_at DESC")
    chats = c.fetchall()
    conn.close()
    return chats

def load_chat_messages(chat_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY timestamp ASC", (chat_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]


# ==============================
# CONNECT TO CHROMADB
# ==============================
@st.cache_resource
def get_chroma_client():
    client = chromadb.PersistentClient(path=DATABASE_FOLDER)
    return client.get_or_create_collection(name="knowledge")

collection = get_chroma_client()


# ==============================
# AUTO-SYNC ENGINE (PYPDF)
# ==============================
def auto_sync_knowledge():
    """Scans 'knowledge' folder and batch-indexes new/un-indexed files using pypdf and nomic-embed-text."""
    if not os.path.exists(KNOWLEDGE_FOLDER):
        os.makedirs(KNOWLEDGE_FOLDER)
        return

    # 1. Get list of files already in the database
    existing_sources = set()
    existing_records = collection.get(include=["metadatas"])
    if existing_records and existing_records.get("metadatas"):
        for meta in existing_records["metadatas"]:
            if meta and "source" in meta:
                existing_sources.add(meta["source"])

    # 2. Find un-indexed files
    new_files = []
    for root, _, files in os.walk(KNOWLEDGE_FOLDER):
        for filename in files:
            if filename.lower().endswith((".txt", ".md", ".pdf")):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, KNOWLEDGE_FOLDER).replace("\\", "/")
                
                if rel_path not in existing_sources:
                    new_files.append((full_path, rel_path))

    if not new_files:
        return

    # 3. Process new files with pypdf and mini-batch embeddings
    with st.spinner(f"⚡ Fast-indexing {len(new_files)} new document(s) with nomic-embed-text..."):
        for full_path, rel_path in new_files:
            text = ""
            if full_path.lower().endswith(".pdf"):
                try:
                    reader = PdfReader(full_path)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                except Exception as e:
                    st.error(f"Error reading {rel_path}: {e}")
                    continue
            else:
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception as e:
                    st.error(f"Error opening {rel_path}: {e}")
                    continue

            if not text.strip():
                continue

            # Split text into chunks (500 words with 50-word overlap)
            words = text.split()
            chunk_size = 500
            overlap = 50
            
            chunks = []
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk)

            if not chunks:
                continue

            # Nomic Prefix for Documents
            formatted_chunks = [f"search_document: {chunk}" for chunk in chunks]

            try:
                all_embeddings = []
                for i in range(0, len(formatted_chunks), BATCH_SIZE):
                    batch = formatted_chunks[i:i + BATCH_SIZE]
                    response = ollama.embed(model=EMBEDDING_MODEL, input=batch)
                    all_embeddings.extend(response["embeddings"])

                ids = [f"{rel_path}_{i}" for i in range(len(chunks))]
                metadatas = [{"source": rel_path} for _ in chunks]

                batch_limit = 100
                for b in range(0, len(chunks), batch_limit):
                    collection.upsert(
                        ids=ids[b:b + batch_limit],
                        documents=chunks[b:b + batch_limit],
                        embeddings=all_embeddings[b:b + batch_limit],
                        metadatas=metadatas[b:b + batch_limit]
                    )

            except Exception as e:
                st.error(f"Error indexing {rel_path}: {e}")

        st.toast("⚡ Knowledge base automatically updated!", icon="✅")

# Run auto-sync on load
auto_sync_knowledge()


# ==============================
# LEFT SIDEBAR - UPLOAD & MANAGEMENT
# ==============================
st.sidebar.title("📁 Document Manager")

subfolder = st.sidebar.text_input("Subfolder (optional)", value="", placeholder="e.g. oops/unit1")
uploaded_files = st.sidebar.file_uploader(
    "Upload new PDF/TXT/MD files", 
    type=["pdf", "txt", "md"], 
    accept_multiple_files=True
)

if uploaded_files:
    target_dir = os.path.join(KNOWLEDGE_FOLDER, subfolder.strip())
    os.makedirs(target_dir, exist_ok=True)
    
    for file in uploaded_files:
        save_path = os.path.join(target_dir, file.name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            st.sidebar.success(f"Saved: {file.name}")
    
    auto_sync_knowledge()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Loaded Knowledge Base")
indexed_data = collection.get(include=["metadatas"])
if indexed_data and indexed_data.get("metadatas"):
    unique_sources = sorted(list(set(m.get("source", "Unknown") for m in indexed_data["metadatas"] if m)))
    for src in unique_sources:
        st.sidebar.text(f"• {src}")
else:
    st.sidebar.caption("No documents loaded yet.")


# ==============================
# RAG RETRIEVAL & STREAMING LOGIC
# ==============================
def get_embedding(text):
    # Nomic Prefix for Search Queries
    response = ollama.embed(model=EMBEDDING_MODEL, input=f"search_query: {text}")
    return response["embeddings"][0]

def ask_question_stream(question):
    auto_sync_knowledge()
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=N_RESULTS,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0] if results["documents"] else []
    metadata = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    relevant_documents = []
    relevant_metadata = []

    for i in range(len(documents)):
        if distances[i] <= MAX_DISTANCE:
            relevant_documents.append(documents[i])
            relevant_metadata.append(metadata[i])

    if not relevant_documents:
        yield "I don't have that information in my knowledge base.", []
        return

    context = "\n\n---\n\n".join(relevant_documents)

    prompt = f"""You are a helpful study assistant analyzing the user's study notes.

Instructions:
1. Synthesize and explain the concepts in the provided KNOWLEDGE BASE to answer the user's question clearly.
2. If asking for a summary, highlight the main topics, definitions, and key phases present in the notes.
3. If the KNOWLEDGE BASE is empty or completely unrelated to the question, state that you don't have details on that specific topic.

KNOWLEDGE BASE:
{context}

QUESTION:
{question}
"""

    response_stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
        stream=True
    )

    for chunk in response_stream:
        yield chunk['message']['content'], relevant_metadata


# ==============================
# INITIALIZE DATABASE & LAYOUT
# ==============================
init_db()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Create 2-column layout: Main Chat (75%) | History Panel (25%)
main_col, right_col = st.columns([3, 1])


# ==============================
# RIGHT COLUMN - CHAT HISTORY PANEL
# ==============================
with right_col:


    st.markdown("""
        <style>
        div[data-testid="stColumn"]:nth-of-type(2) {
            background-color: var(--secondary-background-color);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.subheader("📜 Chat History")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
        

    st.markdown("---")
    st.caption("Saved up to 6 months")

    saved_chats = get_all_chats()
    for chat_id, title in saved_chats:
        label = f"📌 {title[:20]}..." if len(title) > 20 else f"💬 {title}"
        if st.button(label, key=f"hist_{chat_id}", use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.session_state.messages = load_chat_messages(chat_id)
            st.rerun()


# ==============================
# MAIN COLUMN - CHAT UI
# ==============================
with main_col:
    st.title("🤖 Local AI Assistant")
    st.caption("Ask questions about your uploaded documents and notes.")

    # Render Active Chat Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if user_input := st.chat_input("Ask a question about your files..."):
        # Derive chat title
        if not st.session_state.messages:
            chat_title = user_input[:30]
        else:
            chat_title = st.session_state.messages[0]["content"][:30]

        # Display and save user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message(st.session_state.current_chat_id, chat_title, "user", user_input)

        # Stream assistant response
        with st.chat_message("assistant"):
            last_metadata = []

            def text_generator():
                global last_metadata
                for text_chunk, meta in ask_question_stream(user_input):
                    last_metadata = meta
                    yield text_chunk

            full_response = st.write_stream(text_generator())

            # Append source tags
            sources = set(item.get("source", "Unknown") for item in last_metadata)
            if sources:
                source_text = "\n\n**Sources:**\n" + "\n".join(f"- `{src}`" for src in sources)
                st.markdown(source_text)
                full_response += source_text

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        save_message(st.session_state.current_chat_id, chat_title, "assistant", full_response)
        st.rerun()