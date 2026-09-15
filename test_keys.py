from dotenv import load_dotenv
import os

# Load .env from the same folder as this Python file
load_dotenv()

bearer_token = os.getenv("X_BEARER_TOKEN")
api_key = os.getenv("X_API_KEY")
api_key_secret = os.getenv("X_API_KEY_SECRET")

print("Bearer Token:", bool(bearer_token))
print("API Key:", bool(api_key))
print("API Key Secret:", bool(api_key_secret))