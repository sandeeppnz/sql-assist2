import requests
import json

prompt = """
Generate a SQL SELECT query.
Question: Total Internet Sales Amount for 2012
"""

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5:7b-instruct",
        "prompt": prompt,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": True,  # Ollama streams by default
    },
    stream=True  # Enable streaming in requests
)

print("RAW RESPONSE (first few chunks):")
# Ollama returns streaming JSON: one JSON object per line
full_response = ""
chunk_count = 0
for line in resp.iter_lines():
    if line:
        chunk_count += 1
        if chunk_count <= 3:  # Show first 3 chunks
            print(line.decode('utf-8'))
        try:
            chunk = json.loads(line.decode('utf-8'))
            if "response" in chunk:
                full_response += chunk["response"]
            if chunk.get("done", False):
                break
        except json.JSONDecodeError:
            continue

print(f"\n... (total chunks: {chunk_count})")
print()
print("PARSED (full response):")
print(full_response.strip())
