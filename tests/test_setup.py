import openai
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"API key loaded: {api_key[:10]}...")

client = openai.OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Say hello in one word."}]
)
print(f"OpenAI response: {response.choices[0].message.content}")

chroma_client = chromadb.Client()
print("ChromaDB works!")

print("\nSve radi, mozemo da krenemo!")
