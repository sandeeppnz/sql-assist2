from typing import Optional
from vn_local import LocalVanna

_vn_instance: Optional[LocalVanna] = None

# Toggle this to False to disable debug output
DEBUG = True


def get_vn() -> LocalVanna:
    """
    Returns a singleton LocalVanna instance so the Ollama model
    is loaded only once.
    """
    global _vn_instance
    if _vn_instance is None:
        _vn_instance = LocalVanna()
    return _vn_instance


def _normalize_sql_output(sql) -> str:
    """
    Normalize whitespace and ensure we always return a string.
    Some LLMs return lists, dicts, or multi-line formatted strings.
    """
    if sql is None:
        return ""

    # Convert non-string outputs (rare but can occur)
    if not isinstance(sql, str):
        sql = str(sql)

    cleaned = " ".join(sql.split())
    return cleaned.strip()


def _debug_log(tag: str, raw_sql: str):
    """Print debug info for SQL model outputs."""
    if not DEBUG:
        return

    print("\n" + "=" * 80)
    print(f"DEBUG: RAW OUTPUT ({tag})")
    print("-" * 80)
    print(raw_sql)
    print("=" * 80 + "\n")


def generate_sql_from_question(question: str) -> str:
    """
    NL → SQL generation using Vanna+Ollama.
    """
    vn = get_vn()
    raw = vn.generate_sql(question)   # Vanna decides plan

    _debug_log("FROM QUESTION", raw)

    return _normalize_sql_output(raw)


def generate_sql_from_prompt(prompt: str) -> str:
    """
    SQL repair mode (direct prompt → SQL).
    """
    vn = get_vn()
    raw = vn.generate_sql(prompt)

    _debug_log("FROM REPAIR PROMPT", raw)

    return _normalize_sql_output(raw)
