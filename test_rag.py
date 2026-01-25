import sys
import os
import time
from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv()

# Print GEMINI_API_KEY status immediately after loading
print(f"GEMINI_API_KEY at start: {'set' if os.getenv('GEMINI_API_KEY') else 'not set'}")

from rag_setup import setup_rag, load_vector_store, get_relevant_documents
from nld_generator.nld_generator_1_4 import generate_nld, format_docs_for_context
import text_converted

# Print current working directory and .env file path
current_dir = os.getcwd()
env_path = os.path.join(current_dir, '.env')
print(f"Current working directory: {current_dir}")
print(f"Absolute path of .env file: {env_path}")

# Print the contents of the .env file
print("Contents of .env file:")
try:
    with open(env_path, 'r') as env_file:
        env_contents = env_file.read()
        print(env_contents)
except FileNotFoundError:
    print(f".env file not found at {env_path}")
except Exception as e:
    print(f"Error reading .env file: {e}")

# Load environment variables
try:
    load_dotenv(dotenv_path=env_path)
    print("Environment variables loaded successfully")
except Exception as e:
    print(f"Error loading environment variables: {e}")

# Set Gemini API key
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Print all environment variables (for debugging purposes)
print("\nAll environment variables:")
for key, value in os.environ.items():
    print(f"{key}: {value}")

print(f"\nGemini API Key: {os.environ.get('GEMINI_API_KEY')}")

if os.environ.get('GEMINI_API_KEY') is None:
    print("GEMINI_API_KEY is not set in the environment variables")

def test_rag():
    print("Running Robust Text Extraction...")
    text_converted.process_folder("rag_test/")
    
    print("Setting up RAG system...")
    setup_rag()
    
    print("Loading vector store...")
    vector_store = load_vector_store()
    
    test_terms = ["betume", "basin", "evaporite", "carbonatite", "aptian"]
    output_file = "output/rag_test_output.md"
    
    with open(output_file, "w") as f:
        f.write("# RAG Pipeline Test Results\n\n")
        
        for term in test_terms:
            print(f"\nTesting term: {term}")
            f.write(f"## Term: {term}\n\n")
            
            print("Retrieving relevant documents (Hybrid + Rerank)...")
            relevant_docs_with_scores = get_relevant_documents(f"What is the definition of {term}?", vector_store)
            
            # Construct context with metadata (Headers) using shared function
            context = format_docs_for_context(relevant_docs_with_scores)
            
            f.write("### Top 5 Retrieved Segments\n")
            for i, (doc, score) in enumerate(relevant_docs_with_scores, 1):
                source = doc.metadata.get('source', 'Unknown source')
                f.write(f"**{i}. Score: {score:.4f}** | Source: `{os.path.basename(source)}`\n")
                f.write(f"> {doc.page_content}\n\n")
            
            print("Generating NLD...")
            nld, full_prompt = generate_nld(term, context)
            print(f"Generated NLD:\n{nld}\n")
            
            f.write("### Generated Definition (NLD)\n")
            f.write(f"{nld}\n\n")
            
            f.write("<details>\n<summary>Full Prompt</summary>\n\n")
            f.write(f"```text\n{full_prompt}\n```\n")
            f.write("</details>\n\n")
            f.write("---\n\n")
            
            print("-" * 50)
            
            # Respect API rate limits (Free Tier: 15 RPM / 1M TPM)
            print("Waiting 60 seconds to satisfy API rate limits...")
            time.sleep(60)

if __name__ == "__main__":
    test_rag()
