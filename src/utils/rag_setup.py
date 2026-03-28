import os
import shutil
from typing import List, Tuple
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

DOCS_DIR = os.environ.get("DOCS_DIR", "inputs/")
# Dynamic DB Persistence: We will append parameters to this path if needed, but defaults here.
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "chroma_db")

# --- MODEL CONFIGURATION (BGE-M3 SUITE) ---
# BGE-M3: State-of-the-art Multilingual, Long Context (8192)
EMBED_MODEL = "BAAI/bge-m3"
# BGE-Reranker-v2-m3: Matches the embedding model distribution
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Default params (can be overridden)
DEFAULT_SEARCH_K = 20
DEFAULT_RERANK_K = 5

_BM25_RETRIEVER = None
_CROSS_ENCODER = None

def load_documents(docs_dir: str) -> List:
    loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    return documents

def split_documents(documents: List, chunk_size: int = 1024, chunk_overlap: int = 100) -> List:
    """
    Semantic Chunking:
    1. Split by Markdown Headers (Structure preservation)
    2. Split by Characters (Size limit)
    """
    # 1. Structure Split
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    md_header_splits = []
    for doc in documents:
        splits = markdown_splitter.split_text(doc.page_content)
        for split in splits:
             split.metadata.update(doc.metadata)
        md_header_splits.extend(splits)

    # 2. Size Split (Dynamic)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    final_splits = text_splitter.split_documents(md_header_splits)
    print(f"Split {len(documents)} docs -> {len(md_header_splits)} header sections -> {len(final_splits)} final chunks (Size: {chunk_size}).")
    return final_splits

def get_chroma_path(chunk_size: int) -> str:
    """Returns a unique DB path for a specific chunk size to avoid collisions during Grid Search."""
    return f"{CHROMA_DB_DIR}_{chunk_size}"

def create_vector_store(documents: List, chunk_size: int) -> Chroma:
    """Creates (and persists) a specific vector store for the given chunk size."""
    db_path = get_chroma_path(chunk_size)
    
    # Clear existing if needed to ensure fresh index
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    print(f"Creating Index with BGE-M3 at {db_path}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True}, # 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=db_path)
    vector_store.persist()
    return vector_store

def load_vector_store(chunk_size: int = 1024) -> Chroma:
    db_path = get_chroma_path(chunk_size)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return vector_store

def get_bm25_retriever(docs_list: List = None):
    """
    Singleton for BM25. 
    Note: BM25 depends on the splits. Ideally, we should rebuild it if splits change.
    For Grid Search, we will probably re-initialize this per run.
    """
    global _BM25_RETRIEVER
    if docs_list:
        _BM25_RETRIEVER = BM25Retriever.from_documents(docs_list)
    return _BM25_RETRIEVER

def get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        print(f"Loading Re-ranker: {RERANK_MODEL}...")
        _CROSS_ENCODER = CrossEncoder(RERANK_MODEL, trust_remote_code=True)
    return _CROSS_ENCODER

def get_relevant_documents(
    query: str, 
    vector_store: Chroma, 
    bm25_retriever: BM25Retriever = None,
    search_k: int = DEFAULT_SEARCH_K, 
    rerank_k: int = DEFAULT_RERANK_K
) -> List[Tuple]:
    """
    Full Hybrid Pipeline:
    1. Parallel Retrieval: Dense (BGE-M3) + Sparse (BM25)
    2. Ensemble: Weighted fusion
    3. Re-ranking: BGE-Reranker-v2
    """
    
    # 1. Dense Retriever
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": search_k})
    
    # 2. Hybrid Ensemble (weighted merge of BM25 sparse + dense)
    if bm25_retriever:
        bm25_docs = bm25_retriever.invoke(query)
        dense_docs = chroma_retriever.invoke(query)
        # Weighted rank fusion: assign reciprocal rank scores, combine
        doc_scores: dict[str, tuple[float, object]] = {}
        for rank, doc in enumerate(bm25_docs):
            score = 0.4 / (rank + 1)  # BM25 weight=0.4
            key = doc.page_content
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + score, doc_scores[key][1])
            else:
                doc_scores[key] = (score, doc)
        for rank, doc in enumerate(dense_docs):
            score = 0.6 / (rank + 1)  # Dense weight=0.6
            key = doc.page_content
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + score, doc_scores[key][1])
            else:
                doc_scores[key] = (score, doc)
        sorted_ensemble = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        initial_docs = [doc for _, doc in sorted_ensemble]
    else:
        # Fallback to Dense only
        initial_docs = chroma_retriever.invoke(query)
        
    # Deduplicate (Ensemble might return duplicates)
    unique_docs = {doc.page_content: doc for doc in initial_docs}
    initial_docs = list(unique_docs.values())

    # 3. Re-ranking
    try:
        cross_encoder = get_cross_encoder()
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        if not pairs:
            return []

        scores = cross_encoder.predict(pairs)
        
        # Combine and Sort
        scored_docs = zip(initial_docs, scores)
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        # Top N
        return sorted_docs[:rerank_k]
        
    except Exception as e:
        print(f"Warning: Re-ranking failed ({e}). Returning raw results.")
        return [(doc, 1.0) for doc in initial_docs[:rerank_k]]

def setup_rag(chunk_size: int = 1024, chunk_overlap: int = 100, force_rebuild: bool = False):
    """
    Main setup function.
    Can be called by pipeline.py (default params) or optimize_rag.py (grid search).
    When force_rebuild=False and a DB already exists, loads from disk instead of recreating.
    BM25 is always rebuilt in-memory (not persistable).
    """
    print(f"--- Setting up RAG (Chunk Size: {chunk_size}, Force Rebuild: {force_rebuild}) ---")
    if not os.path.exists(DOCS_DIR):
        print(f"Error: {DOCS_DIR} not found.")
        return None, None

    db_path = get_chroma_path(chunk_size)

    # Check if we can reuse existing ChromaDB
    if not force_rebuild and os.path.exists(db_path) and os.listdir(db_path):
        print(f"Found existing ChromaDB at {db_path}. Loading from disk (use force_rebuild=True to recreate).")
        vector_store = load_vector_store(chunk_size)

        # BM25 must be rebuilt in-memory every time
        documents = load_documents(DOCS_DIR)
        if not documents:
            print("No documents found for BM25.")
            return vector_store, None
        split_docs = split_documents(documents, chunk_size, chunk_overlap)
        bm25 = get_bm25_retriever(split_docs)

        print("RAG Setup Complete (ChromaDB from cache, BM25 rebuilt).")
        return vector_store, bm25

    # Load
    documents = load_documents(DOCS_DIR)
    if not documents:
        print("No documents found.")
        return None, None

    # Split
    split_docs = split_documents(documents, chunk_size, chunk_overlap)

    # Index (Dense) - Returns Chroma Object
    vector_store = create_vector_store(split_docs, chunk_size)

    # Index (Sparse) - Returns BM25 Object
    bm25 = get_bm25_retriever(split_docs)

    print("RAG Setup Complete.")
    return vector_store, bm25

if __name__ == "__main__":
    setup_rag()
