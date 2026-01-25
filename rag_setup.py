import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

DOCS_DIR = "rag_test/"  # Path to the folder with extracted text files including .md
CHROMA_DB_DIR = "chroma_db"
# Upgraded from all-MiniLM-L6-v2 for better semantic quality
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# RAG limits
RAG_SEARCH_K = int(os.environ.get("RAG_SEARCH_K", 10))
RAG_RERANK_K = int(os.environ.get("RAG_RERANK_K", 5))

_BM25_RETRIEVER = None
_CROSS_ENCODER = None

def load_documents(docs_dir: str) -> List:
    # Changed from **/*.txt to **/*.md to use the Markdown output
    loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    return documents

def split_documents(documents: List, chunk_size: int = 500, chunk_overlap: int = 50) -> List:
    # 1. Split by Markdown Headers (Context Preservation)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Process each document
    md_header_splits = []
    for doc in documents:
        splits = markdown_splitter.split_text(doc.page_content)
        for split in splits:
             # Preserve original metadata (source file)
             split.metadata.update(doc.metadata)
        md_header_splits.extend(splits)

    # 2. Split recursively (Size Limit Enforcement)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    final_splits = text_splitter.split_documents(md_header_splits)
    print(f"Split {len(documents)} docs into {len(md_header_splits)} sections, then {len(final_splits)} chunks.")
    return final_splits

def create_vector_store(documents: List) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_DB_DIR)
    vector_store.persist()
    return vector_store

def load_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    return vector_store

def get_bm25_retriever():
    global _BM25_RETRIEVER
    if _BM25_RETRIEVER is None:
        print("Initializing BM25 Retriever (Hybrid Search)...")
        if not os.path.exists(DOCS_DIR):
             print(f"Warning: DOCS_DIR {DOCS_DIR} not found. Hybrid search fallback to Vector only.")
             return None
        docs = load_documents(DOCS_DIR)
        split_docs = split_documents(docs)
        _BM25_RETRIEVER = BM25Retriever.from_documents(split_docs)
        _BM25_RETRIEVER.k = RAG_SEARCH_K
    return _BM25_RETRIEVER

def get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        print("Loading CrossEncoder for Re-ranking...")
        # ms-marco-MiniLM-L-6-v2 is fast and effective for passage ranking
        _CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _CROSS_ENCODER

def get_relevant_documents(query: str, vector_store: Chroma) -> List:
    # 1. Hybrid Search (Vector + Keyword)
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": RAG_SEARCH_K})
    bm25_retriever = get_bm25_retriever()
    
    if bm25_retriever:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.4, 0.6]  # Slight preference to semantic
        )
        initial_docs = ensemble_retriever.invoke(query)
    else:
        initial_docs = chroma_retriever.invoke(query)
        
    # 2. Re-ranking (Cross-Encoder)
    try:
        cross_encoder = get_cross_encoder()
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = cross_encoder.predict(pairs)
        
        # Combine docs with scores and sort
        scored_docs = zip(initial_docs, scores)
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        # Select top N based on RAG_RERANK_K
        return sorted_docs[:RAG_RERANK_K]
    except Exception as e:
        print(f"Warning: Re-ranking failed ({e}), returning initial results.")
        # Return initial results with dummy score 1.0 (since no re-ranking happened)
        return [(doc, 1.0) for doc in initial_docs[:RAG_RERANK_K]]

def setup_rag():
    print("Setting up RAG system...")
    if not os.path.exists(DOCS_DIR):
        print(f"Error: The directory {DOCS_DIR} does not exist. Please run text extraction first.")
        return
    
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"No documents found in {DOCS_DIR}. Please ensure text extraction was successful.")
        return
    
    split_docs = split_documents(documents)
    create_vector_store(split_docs)
    print("RAG system setup complete.")

if __name__ == "__main__":
    setup_rag()
