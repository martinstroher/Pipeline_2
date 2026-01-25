import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    print("Listing models...")
    for m in genai.list_models():
        print(f"{m.name} ({m.supported_generation_methods})")
except Exception as e:
    print(f"Error: {e}")
