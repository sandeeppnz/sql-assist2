import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_store")
STRICT_PREFLIGHT: bool = os.getenv("STRICT_PREFLIGHT", "true").lower() == "true"

GOLD_TEST_FILE: str = os.getenv("GOLD_TEST_FILE", "gold_eval.json")
