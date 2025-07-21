# 🧠 DVC RAG Assistant

A lightweight Retrieval-Augmented Generation (RAG) assistant that answers questions about [DVC (Data Version Control)](https://dvc.org) using local markdown docs and language models. No internet or cloud APIs required after setup.

---

## 📁 Project Structure

| File              | Purpose                                      |
|-------------------|----------------------------------------------|
| `load_chunk.py`   | Loads & splits DVC `.md` docs into chunks    |      |
| `embed_store.py`  | Converts chunks into vector embeddings using a local model and stores in ChromaDB |
| `rag_infer.py`    | CLI app for asking questions using RAG       |

---

## 🔁 Switching Between Online and Offline Mode

By default, the project uses the Hugging Face model `"sentence-transformers/all-MiniLM-L6-v2"`.

To run in restricted (offline) environments:

1. Download the model from Hugging Face into: `./models/all-MiniLM-L6-v2`
2. Set the environment variable:

```bash
export USE_LOCAL_MODEL=true
```

---

## ⚙️ Setup Instructions

1. Clone DVC docs (`.md` files) into `./data/raw_docs`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Download local embedding model (e.g. `all-MiniLM-L6-v2`) from Hugging Face and place it in `~/models/`
4. Run setup steps:

   ```bash
   python load_chunk.py
   python embed_store.py
   ```
5. Ask questions:

   ```bash
   python rag_infer.py
   ```

---

## 🧠 What This Does

* Loads real DVC documentation
* Chunks and embeds it using a local Sentence-Transformer model
* Stores in a local vector DB (Chroma)
* Runs a RAG pipeline to answer questions with **source references**

---

## 🛠️ Example

```text
❓ Question: How does DVC handle large files?

🧠 Answer:
DVC stores large files in remote storage (like S3, GCS) and tracks them via metafiles in Git.

📚 Sources:
user-guide/large-files.md
```

---

## 📦 Dependencies

* `langchain`
* `chromadb`
* `sentence-transformers`
* `openai` *(optional, replaceable with local models)*

---

## 📄 License

MIT — for educational and experimental use.

