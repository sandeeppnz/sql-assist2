# llm_openai_adapter.py
import os
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

from openai_retry import openai_with_retry
from monitoring import monitor_llm_call


class OpenAIVanna(ChromaDB_VectorStore, OpenAI_Chat):
    """Vanna powered by OpenAI Chat + Chroma vectorstore."""

    def __init__(self):
        config = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "persist_directory": os.getenv("CHROMA_DIR", "./chroma_store"),
            "temperature": 0.0,
            "max_tokens": 2048,
        }

        if not config["api_key"]:
            raise RuntimeError("OPENAI_API_KEY missing")

        self.model_name = config["model"]

        # Chroma uses persist_directory, OpenAI_Chat consumes model/temperature/token settings
        chroma_config = {
            "model": config["model"],
            "persist_directory": config["persist_directory"],
        }
        ChromaDB_VectorStore.__init__(self, config=chroma_config)

        openai_config = {
            "model": config["model"],
            "api_key": config["api_key"],
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
        }
        OpenAI_Chat.__init__(self, config=openai_config)

    # ------- SQL Generation -------
    def generate_sql(self, question: str):
        return monitor_llm_call(
            action_name="generate_sql",
            model_name=self.model_name,
            func=lambda question: openai_with_retry(super(OpenAIVanna, self).generate_sql, question),
            question=question
        )

    # ------- Embeddings -------
    def generate_embedding(self, text: str):
        return monitor_llm_call(
            action_name="embedding",
            model_name=self.model_name,
            func=lambda text: openai_with_retry(super(OpenAIVanna, self).generate_embedding, text),
            text=text
        )

    def embed(self, text: str):
        return self.generate_embedding(text)
