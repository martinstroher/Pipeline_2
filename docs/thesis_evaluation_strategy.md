# Methodology: RAG Optimization Strategy

## 1. Experimental Design (Grid Search)
To empirically determine the optimal configuration for the RAG pipeline, we designed a Grid Search experiment that systematically varies the critical hyperparameters of the retrieval system. The performance of each configuration is evaluated against the "Golden Set" of 22 expert-validated queries.

The experimental grid consists of the following parameters, resulting in **12 distinct experimental runs**:

### 1.1. Chunk Size (Semantic Context Window)
We test three distinct chunk sizes to evaluate the trade-off between outcome precision and context availability:
*   **1024 Tokens**: Standard baseline, adequate for capturing single paragraphs or definitions.
*   **2048 Tokens**: Extended context, suitable for capturing complex geological descriptions that span multiple paragraphs.
*   **4000 Tokens**: Maximum context, effectively treating entire "sections" or "pages" as atomic units. This tests the hypothesis that the large context window of the Generator (Gemini 1.5) benefits from broader, less fragmented input.

### 1.2. Retrieval Depth (Recall vs. Precision)
*   **Top-K = 20**: A conservative retrieval depth, relying on the high precision of the `bge-m3` embedding model.
*   **Top-K = 50**: A broad retrieval depth (high recall strategy), designed to feed a larger candidate pool into the Re-ranker, minimizing the risk of missing relevant documents in the first stage.

### 1.3. Re-Ranking (Refinement)
*   **Top-N = 5**: Strict filtering, providing only the very highest confidence passages to the LLM to reduce noise.
*   **Top-N = 10**: Broader context, providing more supporting evidence to the LLM.

## 2. Chunking Strategy: Semantic vs. Page-Based
A critical methodological decision was the selection of the Text Segmentation (Chunking) strategy.

### 2.1. Rejection of Page-Based Chunking
We evaluated and **rejected** a rigid "Page-Based" chunking strategy (i.e., treating each PDF page as a chunk).
*   **Reasoning**: Geological arguments and definitions frequently cross page boundaries. A rigid page split arbitrarily severs semantic connections (e.g., a sentence starting on Page 1 and ending on Page 2). This "context fragmentation" significantly degrades retrieval quality.

### 2.2. Adoption of Semantic (Header-Aware) Chunking
We adopted a **Semantic Chunking** strategy that leverages the hierarchical structure of the source documents (Markdown).
*   **Mechanism**: The text is first split by logical headers (`# Introduction`, `## Methodology`).
*   **Secondary Split**: Within these logical sections, text is recursively split by character count (1024/2048/4000) only if necessary to fit the window.
*   **Benefit**: This ensures that "Page-sized" content (e.g., 4000 tokens) is maintained as a coherent unit while respecting the logical flow of the scientific argument.
