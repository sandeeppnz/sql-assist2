from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
from config import OLLAMA_MODEL, OLLAMA_HOST, CHROMA_DIR

class LocalVanna(ChromaDB_VectorStore, Ollama):
    """Vanna instance backed by Ollama + ChromaDB (fully local)."""
    def __init__(self):
        config = {
            "model": OLLAMA_MODEL,
            "host": OLLAMA_HOST,
            "persist_directory": CHROMA_DIR,
        }
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
