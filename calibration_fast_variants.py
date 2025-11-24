# fast_self_agreement.py
import re

def fast_sql_normalize(sql: str) -> str:
    """Lowercase, strip, and standardize spaces."""
    sql = sql.lower().strip().rstrip(";")
    sql = " ".join(sql.split())
    return sql

def canonicalize_keywords(sql: str) -> str:
    """Normalize SQL keywords."""
    keywords = [
        "select", "from", "where", "join", "left join", "right join",
        "group by", "order by", "inner join", "outer join"
    ]
    out = sql.lower()
    for kw in keywords:
        out = re.sub(r"\b" + kw + r"\b", kw, out)
    return " ".join(out.split())

def fast_self_agreement_variants(sql: str) -> list:
    """
    Produce 3 very fast, deterministic variants for self-agreement.
    These require NO LLM calls.
    """
    base = fast_sql_normalize(sql)

    v1 = canonicalize_keywords(base)
    v2 = base.replace(",", ", ")
    v3 = re.sub(r"\s*=\s*", " = ", base)

    return [v1, v2, v3]
