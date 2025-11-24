# vanna_provider.py
from typing import Optional
import os

from tqdm import tqdm

from vn_local import LocalVanna              # Ollama
from llm_openai_adapter import OpenAIVanna   # OpenAI adapter


# GLOBAL SINGLETON INSTANCE
_vn_instance: Optional[object] = None

# Toggle this to False to disable debug logs
DEBUG = True


def get_vn():
    """
    Returns a singleton Vanna instance.
    Chooses between Ollama or OpenAI based on VN_PROVIDER env var.
    """
    global _vn_instance

    if _vn_instance is not None:
        return _vn_instance

    provider = os.getenv("VN_PROVIDER", "ollama").lower()

    if provider == "openai":
        tqdm.write("→ Using OpenAI Vanna Adapter")
        _vn_instance = OpenAIVanna()
    else:
        tqdm.write("→ Using Ollama Vanna Adapter (default)")
        _vn_instance = LocalVanna()

    return _vn_instance


# ----------------------------------------------------------
# NORMALIZATION + DEBUGGING
# ----------------------------------------------------------

def _normalize_sql_output(sql) -> str:
    """
    Normalize whitespace and ensure we always return a string.
    Some LLMs return dict/list or multiline formatted SQL.
    """
    if sql is None:
        return ""

    if not isinstance(sql, str):
        sql = str(sql)

    cleaned = " ".join(sql.split())
    return cleaned.strip()


def _debug_log(tag: str, raw_sql: str):
    """Print debug information for SQL model outputs."""
    if not DEBUG:
        return

    tqdm.write("\n" + "=" * 80)
    tqdm.write(f"DEBUG: RAW OUTPUT ({tag})")
    tqdm.write("-" * 80)
    tqdm.write(raw_sql)
    tqdm.write("=" * 80 + "\n")


# ----------------------------------------------------------
# DIRECT SQL GENERATION (REPAIR MODE)
# ----------------------------------------------------------

def generate_sql_from_prompt(prompt: str) -> str:
    """
    SQL repair mode: raw prompt -> SQL using vn.generate_sql().
    """
    vn = get_vn()
    raw = vn.generate_sql(prompt)

    _debug_log("FROM QUESTION", raw)

    return _normalize_sql_output(raw)
