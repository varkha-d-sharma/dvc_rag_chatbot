from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chat_models import ChatOpenAI  # ✅ Use chat-based models
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
import os 
from dotenv import load_dotenv


load_dotenv()

# Check whether to use local or online model
USE_LOCAL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
model_path = ""
if USE_LOCAL:
    model_path = os.path.expanduser("~/models/all-MiniLM-L6-v2")
else:
    model_path = "sentence-transformers/all-MiniLM-L6-v2"  # Hugging Face model ID"

# Constants 
PERSIST_DIRECTORY = "./vector_store"

# Load vector DB
embedding = HuggingFaceBgeEmbeddings(model_name=model_path)
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
retriever = vectordb.as_retriever()

# Load LLM 
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


# Prompt template (optional but useful)
prompt_template = """
You are a helpful assistant for DVC (Data Version Control).
Answer the question based on the context below. If unsure, say "I don't know."

Context:
{context}

Question: {question}

Helpful Answer:
"""


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# Create QA chain with source tracking 
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# CLI interaction 
def run_rag():
    print("Ask about DVC (type 'exit' to quit)")
    while True:
        question = input("\n Question: ")
        if question.lower() in ["exit", "quit"]:
            break

        result = qa_chain.invoke({"question": question})
        print("\n Answer:")
        print(result['answer'])

        if result.get('sources'):
            print("\n Sources:")
            print(result['sources'])
        else:
            print("\n(No sources returned)")

if __name__ == "__main__":
    run_rag()