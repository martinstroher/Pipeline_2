import os
from dotenv import load_dotenv

print(f"Current working directory: {os.getcwd()}")
env_path = os.path.join(os.getcwd(), '.env')
print(f"Absolute path of .env file: {env_path}")

print("\nContents of .env file:")
try:
    with open(env_path, 'r') as env_file:
        print(env_file.read())
except FileNotFoundError:
    print(f".env file not found at {env_path}")
except Exception as e:
    print(f"Error reading .env file: {e}")

print("\nLoading environment variables...")
load_dotenv(dotenv_path=env_path)

print("\nEnvironment variables after loading:")
for key, value in os.environ.items():
    if key == 'GEMINI_API_KEY':
        print(f"{key}: {'*' * len(value)}")  # Mask the API key
    else:
        print(f"{key}: {value}")

gemini_api_key = os.getenv('GEMINI_API_KEY')
if gemini_api_key:
    print("\nGEMINI_API_KEY is set")
else:
    print("\nGEMINI_API_KEY is not set")
