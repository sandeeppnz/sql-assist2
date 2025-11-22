import re
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher


# ============================================================
# 1. Schema Validity Score (SVS)
# ============================================================
def schema_validity_score(diagnostics: Dict[str, Any]) -> float:
    """
    Returns 1.0 if SQL uses valid tables/columns and is 'safe',
    otherwise 0.0.
    """
    if not diagnostics:
        return 0.0

    if (diagnostics.get("is_safe") and
        not diagnostics.get("unknown_tables") and
        not diagnostics.get("unknown_columns")):
        return 1.0

    return 0.0


# ============================================================
# 2. Structural Heuristic Score (SHS)
# ============================================================
def structural_heuristic_score(sql: str) -> float:
    """
    Heuristic scoring of SQL structure.
    Checks for SELECT, FROM, JOIN, GROUP BY, ORDER BY.
    Output: value between 0 and 1.
    """
    if not sql:
        return 0.0

    sql_upper = sql.upper()

    features = [
        ("SELECT ", 0.2),
        (" FROM ", 0.2),
        (" JOIN ", 0.2),
        (" GROUP BY ", 0.2),
        (" ORDER BY ", 0.2),
    ]

    score = 0.0
    for pattern, weight in features:
        if pattern in sql_upper:
            score += weight

    return score


# ============================================================
# 3. Self-Agreement Score (SAS)
# ============================================================
def self_agreement_score(main_sql: str, variants: List[str]) -> float:
    """
    Measure how similar the final SQL is to N variant samples.
    Uses difflib string similarity.
    Output: 0 to 1.
    """
    if not variants:
        return 0.0

    sims = []
    for v in variants:
        sm = SequenceMatcher(None, main_sql, v)
        sims.append(sm.ratio())

    return sum(sims) / len(sims)


# ============================================================
# 4. Execution Behavior Score (XBS)
# ============================================================
def execution_behavior_score(
    exec_ok: bool,
    row_count: Optional[int]
) -> float:
    """
    Returns a score based on how cleanly the SQL executed.
    """
    if not exec_ok:
        return 0.0

    # Completely empty or None → low reliability
    if row_count is None:
        return 0.2

    # "Sane" range check
    if 0 <= row_count <= 5_000_000:
        return 1.0

    # too many rows → suspicious
    if row_count > 5_000_000:
        return 0.4

    return 0.0


# ============================================================
# 5. Embedding Similarity Score (ESS)
# ============================================================
def embedding_similarity_score(similarity: Optional[float]) -> float:
    """
    Uses ChromaDB (or other) cosine similarity to reference examples.
    Pass through normalized 0–1.
    """
    if similarity is None:
        return 0.0
    return max(0.0, min(1.0, similarity))


# ============================================================
# 6. Final Calibrated Confidence
# ============================================================
def calibrated_confidence(
    model_sql: str,
    diagnostics: Dict[str, Any],
    exec_ok: bool,
    row_count: Optional[int],
    sql_variants: Optional[List[str]] = None,
    embedding_sim: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Orchestrates all the scoring components and returns:
    {
        "confidence": float,
        "components": {...}
    }
    """

    svs = schema_validity_score(diagnostics)
    shs = structural_heuristic_score(model_sql)
    sas = self_agreement_score(model_sql, sql_variants or [])
    xbs = execution_behavior_score(exec_ok, row_count)
    ess = embedding_similarity_score(embedding_sim)

    # Weighted combination (tunable)
    confidence = (
        0.25 * svs +
        0.20 * shs +
        0.25 * sas +
        0.20 * xbs +
        0.10 * ess
    )

    return {
        "confidence": round(confidence, 4),
        "components": {
            "schema_validity": svs,
            "structure": shs,
            "self_agreement": sas,
            "execution": xbs,
            "embedding_similarity": ess
        }
    }
