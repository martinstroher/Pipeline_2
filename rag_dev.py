# ingest_definitions_rag_with_classifier.py
import sys
import os
from typing import List
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader, DirectoryLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------
# CONFIGURAÇÕES BÁSICAS
# ---------------------------------------------------------

MD_DIR = "/media/leandro/touro/pos-doc/RAG-Martin/untitled_folder_6/"
CHROMA_DIR = "chroma_papers"
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3"

# ---------------------------------------------------------
# CARREGAR E DIVIDIR DOCUMENTOS
# ---------------------------------------------------------
#******************************************************************************
def load_markdown(md_dir: str):
    loader = DirectoryLoader(
        md_dir,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        recursive=True
    )

    documents = loader.load()
    return documents
#******************************************************************************
def split_docs(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ",],
    )
    return splitter.split_documents(docs)
#******************************************************************************
# ---------------------------------------------------------
# VETOR STORE (CHROMA)
# ---------------------------------------------------------
#******************************************************************************
def create_vectorstore(chunks):
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://localhost:11434",
    )
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    #vectordb.persist()
    return vectordb
#******************************************************************************
def load_vectorstore():
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://localhost:11434",
    )
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb
#******************************************************************************
def get_llm():
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url="http://localhost:11434",
        temperature=0.1,
    )
    return llm
#******************************************************************************
# ---------------------------------------------------------
# === NOVO: CLASSIFICADOR "CONTÉM DEFINIÇÃO / NÃO CONTÉM"
# ---------------------------------------------------------

CLASSIFIER_PROMPT = """
You are a binary classifier that checks if a text chunk
likely contains a DEFINITION of a given scientific term.

TERM: "{term}"

TEXT CHUNK:
\"\"\"{chunk}\"\"\"

Answer with EXACTLY one token: "YES" or "NO".

Rules:
- Answer "YES" if the chunk contains phrases like
  "TERM is defined as", "TERM refers to", "we define TERM as",
  or gives an explicit meaning of TERM.
- Otherwise answer "NO".
"""

classifier_prompt = ChatPromptTemplate.from_template(CLASSIFIER_PROMPT)

#******************************************************************************
def build_definition_classifier_chain():
    llm = get_llm()
    chain = classifier_prompt | llm | StrOutputParser()
    return chain

#******************************************************************************
def classify_chunks_for_term(term: str, docs, classifier_chain, max_yes:int = 8):
    """
    Dado um termo e uma lista de documentos (docs) retornados pelo retriever,
    filtra apenas aqueles que o classificador marcar como YES.
    Limita o máximo de 'YES' em max_yes para não explodir contexto.
    """
    filtered = []
    for d in docs:
        chunk_text = d.page_content[:2000]  # por segurança
        inp = {
            "term": term,
            "chunk": chunk_text
        }
        label = classifier_chain.invoke(inp).strip().upper()
        if "YES" in label:
            filtered.append(d)
            if len(filtered) >= max_yes:
                break
    # fallback: se nenhum chunk foi marcado como YES, usa os originais
    return filtered or docs
#******************************************************************************
# ---------------------------------------------------------
# PROMPT PRINCIPAL DE DEFINIÇÃO
# ---------------------------------------------------------

DEFINITION_PROMPT = """
You are an expert assistant extracting precise scientific DEFINITIONS
from petroleum pre-salt papers.

Your task:
1. Read the CONTEXT (snippets from papers).
2. Extract only sentences that define the target term.
3. Then write a concise, synthesized definition.
4. Always list the sources (source_file and page if available).

Rules:
- Base yourself ONLY on the provided context.
- If there is no clear definition, say:
  "No clear definition found in the provided documents."

TERM: "{term}"

CONTEXT:
{context}

Return in this JSON-like structure (but as plain text):
- term:
- extracted_sentences:
- synthesized_definition:
- sources:
"""

definition_prompt = ChatPromptTemplate.from_template(DEFINITION_PROMPT)

# ---------------------------------------------------------
# BUILD RAG CHAIN COM CLASSIFICAÇÃO
# ---------------------------------------------------------
#******************************************************************************
def build_definition_chain(term,vectordb):
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 15},  # traz um pouco mais para o classificador filtrar
    )
    llm = get_llm()
    classifier_chain = build_definition_classifier_chain()

    #******************************************************************************
    def retrieve_and_filter(term: str):
        # 1) recupera candidatos via similaridade
        docs = retriever.invoke(term)

        # 2) filtra com o classificador "contém definição / não contém"
        filtered_docs = classify_chunks_for_term(
            term=term,
            docs=docs,
            classifier_chain=classifier_chain,
            max_yes=8,
        )

        # 3) concatena para o contexto final
        context_str = "\n\n".join(
            f"[{d.metadata.get('arquivo origem','?')} p.{d.metadata.get('pagina','?')}] {d.page_content}"
            for d in filtered_docs
        )
        return context_str
    #******************************************************************************
    # Encadeia: {context, term} -> prompt -> LLM -> texto
    rag_chain = (
        {
            "context": lambda x: retrieve_and_filter(x[0]),
            "term": RunnablePassthrough()
        }
        | definition_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
#******************************************************************************
# ---------------------------------------------------------
# API: get_definition(term)
# ---------------------------------------------------------
#******************************************************************************
def get_definition(term: str, vectordb=None):
    if vectordb is None:
        vectordb = load_vectorstore()
    chain = build_definition_chain(term,vectordb)
    result = chain.invoke(term)
    return result

# ---------------------------------------------------------
# SCRIPT MAIN
# ---------------------------------------------------------
#******************************************************************************
def main(ingest: bool = True):
    if ingest:
        print("Carregando Arquivos Markdown...")
        raw_docs = load_markdown(MD_DIR)
        print(f"{len(raw_docs)} páginas carregadas.")

        print("Dividindo em chunks...")
        chunks = split_docs(raw_docs)
        print(f"{len(chunks)} chunks criados.")

        print("Criando vetor store (Chroma) com embeddings mxbai-embed-large...")
        vectordb = create_vectorstore(chunks)
        print("Vector store criado e persistido.")
    else:
        print("Carregando vetor store existente...")
        vectordb = load_vectorstore()

    term = "The Itapema interval"
    print(f"\nBuscando definição para o termo: {term}\n")
    answer = get_definition(term, vectordb)
    print(answer)
#******************************************************************************
if __name__ == "__main__":
    if os.path.exists(CHROMA_DIR+"/chroma.sqlite3"):
        main(ingest=False)  # se já tiver o índice, use False
    else: main(ingest=True)
#fim if
