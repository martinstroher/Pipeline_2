import os
import re
import pandas as pd
import json
import asyncio
from typing import List, Dict
from datasets import Dataset

# Ragas & LangChain
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Internal Modules
from src.utils import rag_setup
from src.modules.nld_generator import generate_nld

# --- CONFIGURATION ---
GOLDEN_SET_PATH = "inputs/test_dataset.json"
RESULTS_FILE = "output/optimization_results.csv"

# Grid Search Parameters
CHUNK_SIZES = [1024, 2048, 4000]
RETRIEVAL_CONFIGS = [
    {"search_k": 20, "rerank_k": 5},
    {"search_k": 50, "rerank_k": 10}
]

# Setup Gemini for Ragas (The Judge)
# Using flash for speed/cost, or pro for better reasoning if needed. 
# Ragas uses LangChain LLMs.
JUDGE_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

# Judge Embeddings (for metrics that need them, e.g. answer_relevance)
JUDGE_EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

def load_golden_set():
    with open(GOLDEN_SET_PATH, 'r') as f:
        data = json.load(f)
    return data

def extract_term_from_question(question: str) -> str:
    """
    Extracts the term from single quotes in the question, e.g., 
    "Define 'Grainstone'..." -> "Grainstone"
    Fallback: returns the whole question.
    """
    match = re.search(r"'(.*?)'", question)
    if match:
        return match.group(1)
    return question

def format_docs_for_context(docs):
    return "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" for d in docs])

async def run_experiment():
    print(f"Starting RAG Optimization Grid Search...")
    
    golden_data = load_golden_set()
    all_results = []
    
    # 1. Loop Chunk Sizes (Outer Loop - Re-indexing)
    for chunk_size in CHUNK_SIZES:
        print(f"\n=== Testing Chunk Size: {chunk_size} ===")
        
        # Build Index for this chunk size
        vector_store, bm25_retriever = rag_setup.setup_rag(chunk_size=chunk_size)
        
        if not vector_store:
            print("Skipping due to setup execution...")
            continue

        # 2. Loop Retrieval Configs (Inner Loop)
        for config in RETRIEVAL_CONFIGS:
            search_k = config["search_k"]
            rerank_k = config["rerank_k"]
            experiment_id = f"Sz{chunk_size}_K{search_k}_R{rerank_k}"
            print(f"  >> Running Experiment: {experiment_id}")
            
            questions = []
            answers = []
            contexts = []
            ground_truths = []
            
            # 3. Generate Answers for Golden Set
            for item in golden_data:
                question = item["question"]
                ground_truth = item["ground_truth"]
                
                # Retrieve
                current_term = extract_term_from_question(question)
                
                # Get Docs (Hybrid + Rerank)
                retrieved_docs = rag_setup.get_relevant_documents(
                    query=question, # Query using the full question
                    vector_store=vector_store,
                    bm25_retriever=bm25_retriever,
                    search_k=search_k,
                    rerank_k=rerank_k
                )
                
                # Format Config
                # Note: 'generate_nld' expects a string context
                context_text = format_docs_for_context(retrieved_docs)
                
                # Generate Answer
                # We reuse the actual pipeline function to ensure validity
                try:
                    generated_nld, _ = generate_nld(current_term, context_text)
                except Exception as e:
                    print(f"    Error generating NLD for {current_term}: {e}")
                    generated_nld = "Error during generation."
                
                # Collect Data for Ragas
                questions.append(question)
                answers.append(generated_nld)
                contexts.append([doc.page_content for doc in retrieved_docs]) # Ragas expects list of strings
                ground_truths.append(ground_truth)
            
            # 4. Evaluate with Ragas
            dataset_dict = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths
            }
            dataset = Dataset.from_dict(dataset_dict)
            
            print(f"    Evaluating {len(dataset)} items with Ragas...")
            try:
                scores = evaluate(
                    dataset=dataset,
                    metrics=[
                        faithfulness,
                        answer_relevance,
                        context_precision,
                        context_recall
                    ],
                    llm=JUDGE_LLM,
                    embeddings=JUDGE_EMBEDDINGS
                )
                
                # Store Results
                result_row = {
                    "experiment_id": experiment_id,
                    "chunk_size": chunk_size,
                    "search_k": search_k,
                    "rerank_k": rerank_k,
                    "faithfulness": scores["faithfulness"],
                    "answer_relevance": scores["answer_relevance"],
                    "context_precision": scores["context_precision"],
                    "context_recall": scores["context_recall"],
                    "mean_score": (scores["faithfulness"] + scores["context_recall"] + scores["answer_relevance"]) / 3
                }
                all_results.append(result_row)
                print(f"    Scores: {result_row}")
                
            except Exception as e:
                print(f"    Ragas Evaluation Failed: {e}")
    
    # 5. Save Final Results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nOptimization Complete! Results saved to {RESULTS_FILE}")
    print("\nTop 3 Configurations:")
    print(df.sort_values(by="mean_score", ascending=False).head(3))

if __name__ == "__main__":
    # Ensure asyncio loop for Ragas (which uses async calls)
    try:
        asyncio.run(run_experiment())
    except Exception as e:
        print(f"Fatal execution error: {e}")
