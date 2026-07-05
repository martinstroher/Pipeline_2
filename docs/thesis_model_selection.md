## 1. Overview and Theoretical Framework
The retrieval subsystem of the RAG pipeline is the critical bottleneck for accuracy. If the relevant context is not retrieved, the LLM cannot generate a correct answer ("garbage in, garbage out"). To maximize retrieval performance for the specific domain of **Brazilian Pre-Salt Geology**, we selected a state-of-the-art **Hybrid Retrieval + Re-ranking** architecture.

### 1.1. Retrieval Paradigms Explained
To understand the model selection, it is necessary to define the three main retrieval approaches:

*   **Sparse Retrieval (Keyword/Lexical)**:
    *   *Mechanism*: Matches exact words between the query and document (e.g., BM25).
    *   *Pros*: Excellent for specific technical identifiers (e.g., "Well 3-BRSA-123", "Stevensite").
    *   *Cons*: Fails on synonyms (e.g., "ancient lake" might not match "lacustrine deposit").
*   **Dense Retrieval (Semantic/Vector)**:
    *   *Mechanism*: Converts text into numerical vectors where similar *meanings* are close together.
    *   *Pros*: Captures conceptual similarity (e.g., understands that "Ostracod" is related to "Crustacean").
    *   *Cons*: Can "hallucinate" relevance or miss exact keyword matches in technical domains.
*   **Hybrid Retrieval**:
    *   *Mechanism*: Fuses the scores of Sparse and Dense retrieval.
    *   *Justification*: In geology, we need **both**: the concept understanding of Dense (to find descriptions of deposition) AND the precision of Sparse (to find specific formation names like "Barra Velha").

## 2. Embedding Model: BAAI/bge-m3
We selected **`BAAI/bge-m3`** as the core embedding model. This decision is driven by three specific requirements of our corpus:

### 2.1. Multilingual Competency (Portuguese/English)
Our geological corpus and ontology terms frequently mix English technical concepts (e.g., "Grainstone") with Portuguese context or source material.
*   **Justification**: Unlike older English-centric models (like `all-mpnet-base-v2`) or API-based commercial models (like Google `text-embedding-004`) which function as "black boxes", `bge-m3` offers **full scientific reproducibility**. For a Master's Thesis, it is critical to use a model with fixed weights that will not change silently over time (as APIs do), ensuring that our results can be replicated in the future.
*   **Performance**: `bge-m3` is trained on massive multilingual datasets (100+ languages) and often outperforms commercial APIs on retrieval benchmarks (MTEB).

### 2.2. Extended Context Window (8192 Tokens)
Scientific geological texts are dense and often require broad context to be understood (e.g., a description of a depositional environment may span several paragraphs).
*   **Justification**: Standard models (BERT-based) are limited to **512 tokens**. Truncating text causes loss of critical semantic information. `bge-m3` supports up to **8192 tokens**, allowing us to embed entire sections or large chunks of geological papers without information loss.

### 2.3. Hybrid-Native Capabilities
*   **Justification**: `bge-m3` is designed to support dense retrieval (semantic), sparse retrieval (lexical/keyword), and multi-vector retrieval simultaneously. This aligns with our Hybrid Retrieval strategy, providing a robust semantic foundation that outperforms older Dense Retrieval methods on the MTEB (Massive Text Embedding Benchmark) leaderboard.

## 3. Re-Ranker: BAAI/bge-reranker-v2-m3
To further refine the retrieval results, we incorporated a Cross-Encoder Re-ranking step using **`BAAI/bge-reranker-v2-m3`**.

### 3.1. Architecture Superiority (Cross-Encoder vs. Bi-Encoder)
*   **Justification**: While the embedding model (Bi-Encoder) is fast for searching millions of documents, it compresses text into a single vector, losing fine-grained interaction details. A Cross-Encoder inputs the Query and Document *together*, allowing the model to perform deep self-attention on the specific interaction between the question and the text.
*   **Performance**: Cross-Encoders consistently fail significantly less than Bi-Encoders on hard negatives.

### 3.2. Model Synergy
*   **Justification**: Choosing a re-ranker from the same model family as the embedding model (`bge` suite) ensures that the training distributions align. `bge-reranker-v2-m3` shares the same massive multilingual training base and long-context capabilities as `bge-m3`, ensuring that it can accurately re-rank the complex, technical, and multilingual passages retrieved by the first stage.

## 4. Operational Feasibility (Local Execution)
Despite their "State-of-the-Art" performance, these models are highly efficient:
*   **Parameter Count**: ~560 Million parameters.
*   **Memory Footprint**: ~2-3 GB of RAM (fp16).
*   **Infrastructure**: This allows the entire pipeline to run **locally** on standard hardware (e.g., Apple M-series chips or consumer GPUs) without requiring external API calls for embedding generation, ensuring data privacy and zero latency overheads from network requests.

## 4. Generation LLM: gpt-5.4 (Azure AI Foundry)
All generative steps (term extraction, NLD generation, categorization, taxonomy construction, relation extraction, and the OntoClean critic) use **OpenAI `gpt-5.4`** served through **Azure AI Foundry (Azure OpenAI)**, accessed via the OpenAI SDK v1 API in [src/utils/llm_client.py](../src/utils/llm_client.py).

### 4.1. Why a single frontier reasoning model
*   **Uniformity for internal validity.** A single model is used for **every** step and, critically, is held **constant across the four ablation conditions (A/B/C/D)**. The ablation's independent variable is the *NLD representation*, not the model; holding the LLM fixed prevents confounding the RAG/NLD effect with model capability. Mixing models per step was evaluated and rejected on defensibility grounds.
*   **Reasoning capability.** `gpt-5.4` is a reasoning model; the OntoClean critic (rigidity/dependence probes, `KEEP_AS_BEARER` logic, six-verdict triage) is the most reasoning-intensive step and benefits from a frontier model.

### 4.2. Settings (pinned)
| Parameter | Value | Notes |
|---|---|---|
| Deployment / version | `gpt-5.4` (2026-03-05) | pinned Azure deployment |
| `reasoning_effort` | `high` (uniform) | GPT-5.4 defaults to `none`; set explicitly. `xhigh` is not available on 5.4 |
| `max_completion_tokens` | 32000 | bounds reasoning + visible tokens; generous to avoid truncation |
| `seed` | 42 | best-effort determinism |
| `temperature` | — | **not supported** by reasoning models; not sent |

### 4.3. Reproducibility caveat
Reasoning models do not accept `temperature`, and they generate hidden reasoning tokens with run-to-run variance even at a fixed `seed`. Determinism is therefore **best-effort** (pinned deployment version + `seed=42`), **not** bit-exact. This is a deliberate, documented trade-off for access to frontier reasoning quality; per-call token usage (including `reasoning_tokens`) is logged to `output/usage_log.csv` for auditability.

*(Migration note: earlier development used Gemini 2.5 Pro; the pipeline was migrated to Azure-hosted gpt-5.4 with no change to the RAG/embedding subsystem, which remains the local `BAAI/bge-m3` stack described above.)*

