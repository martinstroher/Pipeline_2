import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

DOCS_DIR = "resources/"  # Update this path if your documents are stored elsewhere
CHROMA_DB_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama2"

def load_documents(docs_dir: str) -> List:
    loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents

def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(documents)
    return split_docs

def create_vector_store(documents: List) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_DB_DIR)
    vector_store.persist()
    return vector_store

def load_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    return vector_store

def get_rag_response(query: str, vector_store: Chroma) -> str:
    llm = Ollama(model=LLM_MODEL)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_store.as_retriever())
    response = qa_chain({"query": query})
    return response["result"]

def setup_rag():
    if not os.path.exists(CHROMA_DB_DIR):
        print("Setting up RAG system...")
        documents = load_documents(DOCS_DIR)
        split_docs = split_documents(documents)
        create_vector_store(split_docs)
        print("RAG system setup complete.")
    else:
        print("RAG system already set up. Loading existing vector store.")

if __name__ == "__main__":
    setup_rag()
