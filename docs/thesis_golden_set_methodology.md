# Methodology: Acquisition of the "Golden Set" for RAG Evaluation

## 1. Objective
To rigorously evaluate and optimize the parameters of the Retrieval-Augmented Generation (RAG) pipeline, a high-quality "Golden Set" (Ground Truth dataset) was established. Unlike synthetic datasets generated purely by Large Language Models (LLMs), which may hallucinate or lack domain specificity, this dataset was constructed by consolidating **human expert evaluations**.

The objective was to create a representative benchmark of 20 geological terms that covers both:
1.  **Optimization Targets**: Terms where the initial pipeline failed, but experts provided corrections (to measure improvement).
2.  **Regression Baselines**: Terms where the initial pipeline performed perfectly (to ensure stability).

## 2. Data Source: Expert/Human-in-the-Loop Evaluation
The source data consisted of a comprehensive evaluation table containing approximately 1200 records.
*   **Scope**: 400 geological terms extracted from the Brazilian Pre-Salt corpus.
*   **Evaluators**: Three domain experts (Senior Geologists) evaluated each term.
*   **Attributes Evaluated**:
    *   `Term Relevance`: Whether the term is valid and relevant to the domain.
    *   `NLD Accuracy`: Binary assessment ("Correct"/"Incorrect") of the generated Natural Language Definition.
    *   `Comments`: Qualitative feedback (in English/Portuguese) explaining errors or proposing correct definitions.

## 3. Consolidation Methodology
To transform the 1200 raw evaluation rows into a cohesive "Golden Set" of Question/Answer pairs, we employed a semantic consensus approach using a Large Language Model (LLM) as a reasoning engine.

The logic defined for the consolidation process was as follows:

### 3.1. Selection Strategy (stratified sampling)
The system was instructed to select exactly 20 records based on a 50/50 split:
*   **Group A: "Fixable Failures" (~50%)**: Terms where the majority of experts marked the definition as "Incorrect" but provided constructive scientific feedback. These records constitute the primary target for RAG hyperparameter tuning, as they represent "knowledge gaps" that the improved retrieval settings should resolve.
*   **Group B: "Baseline Benchmarks" (~50%)**: Terms where all three experts unanimously agreed the definition was "Correct". These records serve as a control group to monitor for regression/degradation during the optimization process.

### 3.2. Ground Truth Synthesis
For each selected term, the "Ground Truth" definition was derived via:
*   **For Group A**: Semantic interpretation of the expert comments. The LLM scraped the comments (handling mixed English/Portuguese), isolated the scientific correction (ignoring conversational filler), and synthesized the intended correct definition.
*   **For Group B**: Adoption of the original validated definition, as it was confirmed accurate by the consensus.

## 4. Prompt Engineering
The following prompt was designed and used to guide the LLM in processing the expert data table and generating the Golden Set. It emphasizes semantic interpretation over simple keyword matching to handle the variability in human feedback.

**Prompt Used:**

```markdown
# Role Definitions
You are a **Senior Petroleum Geologist** and **Ontology Expert** specializing in the Brazilian Pre-Salt Carbonate Reservoirs. You are also an expert in **Data Cleaning** and **Multi-lingual Text Analysis**.

# The Task
I will provide you with a dataset (CSV/Excel) containing expert evaluations (~1200 rows, 3 experts per term).
**Your Goal**: Extract a **"Golden Test Set" of exactly 20 high-quality records**.
We need a **balanced mix** to test both *improvement* (on bad terms) and *stability* (on good terms).

# Selection Strategy (The Mix)
You must select exactly 20 terms following this distribution:

1.  **Group A: The "Fixable Failures" (Target: ~10 records)**
    - **Criteria**: The majority of experts marked `NLD Accuracy` = "Incorrect" BUT provided **constructive feedback** in the comments that allows you to deduce the correct definition.
    - **Goal**: Test if the optimized RAG can generate the *corrected* definition that the original pipeline missed.
    - **Ground Truth Source**: Synthesize the *corrected* definition by **interpreting the expert comments**. The correction might be explicitly stated or implied by their critique of what was wrong.

2.  **Group B: The "Baseline Benchmarks" (Target: ~10 records)**
    - **Criteria**: All 3 experts marked `NLD Accuracy` = "Correct" and `Term Relevance` = "Relevant".
    - **Goal**: Ensure the optimized RAG typically maintains high quality and does not "break" what is already working (Regression Testing).
    - **Ground Truth Source**: Use the original `NLD` text (since experts validated it).

# Analysis Logic (Per Selected Term)
1.  **Parse Comments (Crucial)**:
    - **Interpret** the `Comment on Categorization/NLD` column.
    - Do **NOT** rely on finding specific phrases like "Definição sugerida".
    - Instead, read the Portuguese/English commentary to understand the scientific error. Reconstruct the *intended* correct definition based on their technical feedback.
    - Ignore conversational filler (e.g., "eu acho que...", "para mim...", "está errado because..."). Isolate the scientific truth.

2.  **Formulate Output**:
    - `question`: Create a precise technical question.
    - `ground_truth`: The validated or synthesized definition.
    - `group`: Label as "Group A (Correction)" or "Group B (Baseline)".

# Output Format
**CRITICAL**: Output a **JSON List** of exactly 20 items.

[
  {
    "question": "Define 'Grainstone' in the context of reservoir quality.",
    "ground_truth": "Grainstone is a grain-supported carbonate rock with less than 10% matrix...",
    "group": "Group A (Correction)"
  },
  ...
]
```
