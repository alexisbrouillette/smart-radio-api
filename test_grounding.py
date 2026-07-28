"""Quick test — run with: .venv/bin/python test_grounding.py"""
import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
print(f"Using API key: {api_key[:12]}...")

client = genai.Client(api_key=api_key)

models_to_try = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]

for model_name in models_to_try:
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Who sang 'Tout le monde danse autour' and what album is it from? Just answer in one sentence.",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        print(f"✅ SUCCESS!")
        print(f"Response: {response.text}")
        break
    except Exception as e:
        print(f"❌ FAILED: {e}")
