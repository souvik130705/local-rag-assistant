# local-rag-assistant
A 100% private, locally hosted AI assistant built with **Streamlit**, **Ollama**, **ChromaDB**, and **SQLite**. Upload your PDFs, text files, or Markdown notes and chat with them without sending any data to the cloud.
---

## ✨ Features

- 🔒 **100% Local & Private:** Runs entirely on your machine using Ollama and ChromaDB—no API keys or internet connection required.
- ⚡ **Auto-Syncing Knowledge Base:** Automatically indexes uploaded files (`.pdf`, `.txt`, `.md`) placed into the `knowledge/` folder.
- 🎯 **Optimized Retrieval:** Employs `nomic-embed-text` with search-task prefixes (`search_document:` / `search_query:`) for highly accurate context matching.
- 📜 **Persistent Chat History:** Integrated SQLite database retains past conversations for up to 6 months with auto-cleanup.
- 🚀 **Real-Time Streaming:** Responsive UI powered by Streamlit's `st.write_stream`.
- 🎨 **Modern Dual-Sidebar UI:** Streamlined file management on the left, styled chat history on the right.

---

## 🛠️ Tech Stack

- **UI Framework:** Streamlit
- **LLM Engine:** Ollama (`llama3.2:3b`)
- **Embeddings:** Ollama (`nomic-embed-text`)
- **Vector Store:** ChromaDB
- **Database:** SQLite
- **PDF Parser:** PyPDF (`pypdf`)

---

## ⚙️ Prerequisites

1. **Python 3.10+** installed on your system.
2. **Ollama** installed and running. Download from [ollama.com](https://ollama.com/).

Pull the required models via your terminal:
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
