# embed_store.py

# Each chunk is:
# Cleaned of unwanted formatting
# Size-limited for token handling
# Associated with original source (e.g.file path)

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from load_and_chunk import load_documents, split_documents
import os


# Check whether to use local or online model
USE_LOCAL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
model_path = ""
if USE_LOCAL:
    model_path = os.path.expanduser("~/models/all-MiniLM-L6-v2")
else:
    model_path = "sentence-transformers/all-MiniLM-L6-v2"  # Hugging Face model ID"


# Constants
PERSIST_DIRECTORY = "./vector_store"

# Step 1: Load and chunk docs
def prepare_chunks():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

# Step 2: Create and store vector DB
def embed_and_store(chunks, persist_dir=PERSIST_DIRECTORY):
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    vector_db.persist()
    print(f"Vector store saved to {persist_dir}")


# Step 3: Test retrieval
def test_retrieval(persist_dir=PERSIST_DIRECTORY, query="How does DVC handle large files?"):
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectordb.as_retriever()

    docs = retriever.get_relevant_documents(query)
    print(f"Query: {query}\n")
    for i, doc in enumerate(docs[:3]):
        print(f"---- Dcoument {i+1} ---")
        print(doc.page_content[:500])  # Print fiest 500 chars
        print("-" * 40)


if __name__ == "__main__":
    chunks = prepare_chunks()
    embed_and_store(chunks)
    test_retrieval()
