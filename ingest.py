import os
import chromadb
import ollama
from pypdf import PdfReader

KNOWLEDGE_FOLDER = "knowledge"
DATABASE_FOLDER = "chroma_db"

client = chromadb.PersistentClient(path=DATABASE_FOLDER)
collection = client.get_or_create_collection(name="knowledge")


def get_embedding(text):
    response = ollama.embed(
        model=EMBEDDING_MODEL,    # ✅ GOOD: Uses whatever EMBEDDING_MODEL is set at the top
        input=text
    )
    return response["embeddings"][0]


def split_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def process_file(file_path):
    # Get the relative path (e.g., "oops/unit2/notes.pdf") for cleaner source display
    rel_path = os.path.relpath(file_path, KNOWLEDGE_FOLDER)
    text = ""

    # 1. Handle PDF files
    if file_path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF {rel_path}: {e}")

    # 2. Handle Text and Markdown files
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()

    if not text.strip():
        print(f"\nSkipping empty or unreadable file: {rel_path}")
        return

    chunks = split_text(text)
    print(f"\nProcessing: {rel_path} ({len(chunks)} chunks)")

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        
        # Use rel_path in document_id to avoid conflicts if two folders have files with the same name
        document_id = f"{rel_path}_{i}"

        collection.upsert(
            ids=[document_id],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": rel_path}]
        )
        print(f"Stored chunk {i + 1}/{len(chunks)}")


def main():
    if not os.path.exists(KNOWLEDGE_FOLDER):
        print(f"Folder '{KNOWLEDGE_FOLDER}' does not exist.")
        return

    print(f"Scanning '{KNOWLEDGE_FOLDER}' and all subfolders...\n")

    # os.walk automatically searches through all subdirectories recursively
    for root, dirs, files in os.walk(KNOWLEDGE_FOLDER):
        for filename in files:
            if filename.lower().endswith((".txt", ".md", ".pdf")):
                full_path = os.path.join(root, filename)
                process_file(full_path)

    print("\nKnowledge base updated successfully!")


if __name__ == "__main__":
    main()