import requests

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
    },
)

print("RAW RESPONSE:")
print(resp.text)
print()
print("PARSED:")
print(resp.json().get("response"))
