# Geological Term Extraction & Ontology Pipeline

A robust AI-powered pipeline that processes geological PDFs to extract terms, generate definitions, and classify them into a formal ontology structure.

## Quick Start

### 1. Prerequisites
*   Python 3.10+
*   A Google AI Studio API Key (Gemini)

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd Pipeline_2

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
1.  Copy `.env.example` (or create new) to `.env`:
    ```bash
    cp .env.example .env
    ```
2.  Edit `.env` and set your `GEMINI_API_KEY`.
    ```bash
    GEMINI_API_KEY=AIzaSy...
    ```

### 4. Running the Pipeline
Place your PDF files in the `inputs/` folder.
```bash
python3 pipeline.py
```
Outputs will be generated in the `output/` folder.

## Running Tests
To verify the full end-to-end flow (including PDF generation and API integration):

```bash
python3 test/run_e2e_test.py
```
This runs a self-contained test that:
1.  Generates a dummy PDF in `test/inputs_test/`.
2.  Runs the pipeline against it.
3.  Verifies the extraction, NLD generation, and categorization artifacts.

## Project Structure
*   `src/modules/`: Core logic (Extractor, Aggregator, Filter, Generator, Categorizer).
*   `src/utils/`: shared utilities (PDF processing, RAG setup).
*   `pipeline.py`: Main orchestration script.
*   `test/`: E2E test runner and test configuration.
