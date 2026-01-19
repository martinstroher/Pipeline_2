import sys
import os
from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv()

# Print GEMINI_API_KEY status immediately after loading
print(f"GEMINI_API_KEY at start: {'set' if os.getenv('GEMINI_API_KEY') else 'not set'}")

from rag_setup import setup_rag, load_vector_store, get_rag_response
from nld_generator.nld_generator_1_4 import generate_nld

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
    print("Setting up RAG system...")
    setup_rag()
    
    print("Loading vector store...")
    vector_store = load_vector_store()
    
    test_terms = ["dolomite", "carbonate", "sedimentary", "petroleum", "rock"]
    
    for term in test_terms:
        print(f"\nTesting term: {term}")
        
        print("Retrieving context...")
        context = get_rag_response(f"Provide context for the term: {term}", vector_store)
        print(f"Retrieved context:\n{context}\n")
        
        print("Generating NLD...")
        nld = generate_nld(term, vector_store)
        print(f"Generated NLD:\n{nld}\n")
        
        print("-" * 50)

if __name__ == "__main__":
    test_rag()
