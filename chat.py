import chromadb
import ollama
import time

DATABASE_FOLDER = "chroma_db"

LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "nomic-embed-text"

N_RESULTS = 3

# ChromaDB distance above this is considered irrelevant
MAX_DISTANCE = 1.5


# ==============================
# CONNECT TO CHROMADB
# ==============================

print("Connecting to ChromaDB...")

start = time.time()

client = chromadb.PersistentClient(
    path=DATABASE_FOLDER
)

collection = client.get_collection(
    name="knowledge"
)

print(
    "ChromaDB loaded in:",
    round(time.time() - start, 2),
    "seconds"
)


# ==============================
# GET EMBEDDING
# ==============================

def get_embedding(text):

    start = time.time()

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    print(
        "Embedding time:",
        round(time.time() - start, 2),
        "seconds"
    )

    return response["embeddings"][0]


# ==============================
# ASK QUESTION
# ==============================

def ask_question(question):

    total_start = time.time()

    print("\nProcessing question...\n")

    # --------------------------
    # CREATE QUESTION EMBEDDING
    # --------------------------

    question_embedding = get_embedding(question)

    # --------------------------
    # SEARCH CHROMADB
    # --------------------------

    start = time.time()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=N_RESULTS,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    print(
        "ChromaDB search time:",
        round(time.time() - start, 2),
        "seconds"
    )

    documents = results["documents"][0]
    metadata = results["metadatas"][0]
    distances = results["distances"][0]

    # --------------------------
    # FILTER IRRELEVANT RESULTS
    # --------------------------

    relevant_documents = []
    relevant_metadata = []

    print("\nRetrieved results:")

    for i in range(len(documents)):

        print(
            "Result",
            i + 1,
            "distance:",
            round(distances[i], 4),
            "| Source:",
            metadata[i].get("source", "Unknown")
        )

        if distances[i] <= MAX_DISTANCE:

            relevant_documents.append(documents[i])
            relevant_metadata.append(metadata[i])

    # --------------------------
    # NOTHING RELEVANT FOUND
    # --------------------------

    if not relevant_documents:

        print("\nNo sufficiently relevant information found.")

        return (
            "I don't have that information in my knowledge base.",
            []
        )

    # --------------------------
    # BUILD SMALL CONTEXT
    # --------------------------

    context = ""

    for i in range(len(relevant_documents)):

        context += (
            "\n"
            + relevant_documents[i]
            + "\n"
        )

    print(
        "\nRelevant documents:",
        len(relevant_documents)
    )

    print(
        "Context size:",
        len(context),
        "characters"
    )

    # --------------------------
    # SHORT PROMPT
    # --------------------------

    prompt = f"""You are a helpful study assistant analyzing the user's study notes.

Instructions:
1. Synthesize and explain the concepts in the provided KNOWLEDGE BASE to answer the user's question.
2. If asking for a summary, highlight the main topics, definitions, and key phases present in the notes.
3. If the KNOWLEDGE BASE is empty or completely unrelated to the question, state that you don't have details on that specific topic.

KNOWLEDGE BASE:
{context}

QUESTION:
{question}
"""

    # --------------------------
    # QWEN3
    # --------------------------

    print("\nStarting Qwen3:8B...")

    start = time.time()

    try:

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0,
                "num_predict": 100
            },
            think=False,
            stream=False
        )

    except Exception:

        # Fallback for Ollama versions
        # that do not support think=False

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0,
                "num_predict": 100
            },
            stream=False
        )

    llm_time = time.time() - start

    print(
        "Qwen3 generation time:",
        round(llm_time, 2),
        "seconds"
    )

    answer = response["message"]["content"].strip()

    # --------------------------
    # TOTAL TIME
    # --------------------------

    print(
        "\nTOTAL TIME:",
        round(time.time() - total_start, 2),
        "seconds"
    )

    return answer, relevant_metadata


# ==============================
# MAIN
# ==============================

def main():

    print("=================================")
    print("       LOCAL AI ASSISTANT")
    print("=================================")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer, metadata = ask_question(question)

        print("\nAI:")
        print(answer)

        print("\nSources:")

        sources = set()

        for item in metadata:

            sources.add(
                item.get(
                    "source",
                    "Unknown"
                )
            )

        for source in sources:

            print("-", source)

        print()


# ==============================
# START PROGRAM
# ==============================

if __name__ == "__main__":
    main()