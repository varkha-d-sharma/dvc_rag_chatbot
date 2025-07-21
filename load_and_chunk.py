# Use LangChain's DirectoryLoader + MarkdownLoader to read .md file
# Chunk content (use RecursiveCharacterTextSplitter)
# Save chunks to a .jsonl or .pkl for now 


from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import os


# step 1: Load all .md files
def load_documents(doc_path="./data/raw_docs"):
    loader = DirectoryLoader(
        doc_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        use_multithreading=True
    )
    
    
    documents = loader.load()

    # Attach relative file path as source
    for doc in documents:
        full_path = doc.metadata.get("source", "")
        relative_path = os.path.relpath(full_path, doc_path)
        doc.metadata["source"] = relative_path

    return documents



# Step 2: Split into smaller chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    print("Loading documents....")
    docs = load_documents()

    print(f"Loaded {len(docs)} documents")

    print("Splitting into chunks")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # optional: save chunks to file
    with open("data/processed_chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "Unknown")
            f.write(f"--- Chunk {i + 1} | Source: {source} ---\n")
            f.write(chunk.page_content + "\n\n")
