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
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

faithfulness = Faithfulness()
answer_relevance = AnswerRelevancy()
context_precision = ContextPrecision()
context_recall = ContextRecall()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Internal Modules
from src.utils import rag_setup
from src.modules.nld_generator import generate_nld

# --- CONFIGURATION ---
GOLDEN_SET_PATH = "inputs/test_dataset.json"
RESULTS_FILE = "output/optimization_results.csv"

# Grid Search Parameters
CHUNK_SIZES = [512, 1024, 2048]
RETRIEVAL_CONFIGS = [
    {"search_k": 20, "rerank_k": 5},
    {"search_k": 50, "rerank_k": 10}
]

# Setup Gemini for Ragas (The Judge)
# Using flash for speed/cost, or pro for better reasoning if needed. 
# Ragas uses LangChain LLMs.
JUDGE_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
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
    context_parts = []
    for d in docs:
        if isinstance(d, tuple):
             doc_obj = d[0]
        else:
             doc_obj = d
        context_parts.append(f"Source: {doc_obj.metadata.get('source', 'Unknown')}\nContent: {doc_obj.page_content}")
    return "\n\n".join(context_parts)

async def run_experiment():
    print(f"Starting RAG Optimization Grid Search...")
    
    golden_data = load_golden_set()
    # Load existing results if available to resume
    if os.path.exists(RESULTS_FILE):
        try:
            existing_df = pd.read_csv(RESULTS_FILE)
            all_results = existing_df.to_dict('records')
            print(f"Resuming from {len(all_results)} completed experiments.")
        except Exception:
            all_results = []
    else:
        all_results = []
    
    # 1. Loop Chunk Sizes (Outer Loop - Re-indexing)
    for chunk_size in CHUNK_SIZES:
        print(f"\n=== Testing Chunk Size: {chunk_size} ===")
        
        # Build Index for this chunk size
        # Optimization: Only build index if we actually need to run experiments for this chunk size
        # Check if all experiments for this chunk are done
        experiments_for_chunk = [f"Sz{chunk_size}_K{c['search_k']}_R{c['rerank_k']}" for c in RETRIEVAL_CONFIGS]
        completed_ids = [r['experiment_id'] for r in all_results]
        if all(exp_id in completed_ids for exp_id in experiments_for_chunk):
            print(f"Skipping Chunk {chunk_size} (All experiments completed).")
            continue

        vector_store, bm25_retriever = rag_setup.setup_rag(chunk_size=chunk_size)
        
        if not vector_store:
            print("Skipping due to setup execution...")
            continue

        # 2. Loop Retrieval Configs (Inner Loop)
        for config in RETRIEVAL_CONFIGS:
            search_k = config["search_k"]
            rerank_k = config["rerank_k"]
            experiment_id = f"Sz{chunk_size}_K{search_k}_R{rerank_k}"
            
            if experiment_id in completed_ids:
                print(f"  >> Skipping Experiment: {experiment_id} (Already Done)")
                continue

            print(f"  >> Running Experiment: {experiment_id}")
            
            questions = []
            answers = []
            contexts = []
            ground_truths = []
            
            # 3. Generate Answers for Golden Set
            total_q = len(golden_data)
            for i, item in enumerate(golden_data):
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
                # Wrap synchronous call in asyncio to enable timeout
                print(f"    [{i+1}/{total_q}] Generating NLD for: {current_term}")
                try:
                    # Run in thread with 120s timeout
                    generated_nld, _ = await asyncio.wait_for(
                        asyncio.to_thread(generate_nld, current_term, context_text),
                        timeout=120
                    )
                except asyncio.TimeoutError:
                    print(f"    Error: NLD Generation timed out for {current_term}")
                    generated_nld = "Error: Generation Timed Out"
                except Exception as e:
                    print(f"    Error generating NLD for {current_term}: {e}")
                    generated_nld = "Error during generation."
                
                # Collect Data for Ragas
                questions.append(question)
                answers.append(generated_nld)
                
                # Unpack tuple (doc, score) -> doc objects
                doc_objects = [d[0] if isinstance(d, tuple) else d for d in retrieved_docs]
                
                # Prepare Ragas context (list of strings)
                context_strings = [d.page_content for d in doc_objects]
                contexts.append(context_strings)
                ground_truths.append(ground_truth)
                
                # --- THESIS OPTIMIZATION: Instant Save per Question ---
                # Capture Source Metadata for Audit
                sources_list = [d.metadata.get('source', 'Unknown') for d in doc_objects]
                
                DETAILED_FILE = "output/optimization_generations.csv"
                single_row_df = pd.DataFrame([{
                    "experiment_id": experiment_id,
                    "timestamp": pd.Timestamp.now(),
                    "chunk_size": chunk_size,
                    "search_k": search_k,
                    "rerank_k": rerank_k,
                    "question": question,
                    "generated_answer": generated_nld,
                    "ground_truth": ground_truth,
                    "retrieved_sources": json.dumps(sources_list), # Save as JSON string
                    "retrieved_context_snippets": json.dumps(context_strings) # Save text as JSON string to handle newlines
                }])
                
                # Atomic append to CSV
                single_row_df.to_csv(DETAILED_FILE, mode='a', header=not os.path.exists(DETAILED_FILE), index=False)
            
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
                
                # Handle Result Object
                # EvaluationResult object (scores) typically supports __getitem__ (scores['key']) 
                # but might not support .get() or dict() conversion in all versions.
                
                def safe_get_score(result_obj, key):
                    """Safely retireve score from Ragas result object."""
                    val = 0.0
                    try:
                        val = result_obj[key]
                    except (KeyError, TypeError, AttributeError):
                        # Handle key mismatch (answer_relevance vs answer_relevancy)
                        if key == "answer_relevance":
                            try:
                                val = result_obj["answer_relevancy"]
                            except (KeyError, TypeError, AttributeError):
                                print(f"    Warning: Could not retrieve metric '{key}' from result object")
                                return 0.0
                        else:
                             print(f"    Warning: Could not retrieve metric '{key}' from result object")
                             return 0.0
                    
                    # If it's a list (per-sample scores), take the mean
                    if isinstance(val, list):
                        if len(val) == 0: return 0.0
                        return sum(val) / len(val)
                    
                    # If it's a Pandas Series or numpy array
                    if hasattr(val, 'mean'):
                         return float(val.mean())

                    return float(val)

                faithfulness_val = safe_get_score(scores, "faithfulness")
                ans_rel_val = safe_get_score(scores, "answer_relevance")
                precision_val = safe_get_score(scores, "context_precision")
                recall_val = safe_get_score(scores, "context_recall")
                
                result_row = {
                    "experiment_id": experiment_id,
                    "chunk_size": chunk_size,
                    "search_k": search_k,
                    "rerank_k": rerank_k,
                    "faithfulness": faithfulness_val,
                    "answer_relevance": ans_rel_val,
                    "context_precision": precision_val,
                    "context_recall": recall_val,
                    "mean_score": (faithfulness_val + recall_val + ans_rel_val) / 3
                }
                all_results.append(result_row)
                print(f"    Scores: {result_row}")
                
                # INCREMENTAL SAVE
                pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
                print(f"    Saved progress to {RESULTS_FILE}")
                
            except Exception as e:
                print(f"    Ragas Evaluation Failed: {e}")
                try: 
                    print(f"    Result keys available: {list(scores.keys())}") 
                except: 
                    print(f"    Could not list keys from scores object.")
    
    # 5. Save Final Results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nOptimization Complete! Results saved to {RESULTS_FILE}")
    if not df.empty:
        print("\nTop 3 Configurations:")
        print(df.sort_values(by="mean_score", ascending=False).head(3))

if __name__ == "__main__":
    # Ensure asyncio loop for Ragas (which uses async calls)
    try:
        asyncio.run(run_experiment())
    except Exception as e:
        print(f"Fatal execution error: {e}")
