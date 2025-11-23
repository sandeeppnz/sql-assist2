import re
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher

# ============================================================
# 1. Schema Validity Score (SVS)
# ============================================================
def schema_validity_score(diagnostics: Dict[str, Any]) -> float:
    if not diagnostics:
        return 0.0

    if (
        diagnostics.get("is_safe") and
        not diagnostics.get("unknown_tables") and
        not diagnostics.get("unknown_columns")
    ):
        return 1.0

    return 0.0


# ============================================================
# 2. Structural Heuristic Score (SHS)
# ============================================================
def structural_heuristic_score(sql: str) -> float:
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
def execution_behavior_score(exec_ok: bool, row_count: Optional[int]) -> float:
    """
    Only used in mode="eval"
    """
    if not exec_ok:
        return 0.0

    if row_count is None:
        return 0.2

    if 0 <= row_count <= 5_000_000:
        return 1.0

    if row_count > 5_000_000:
        return 0.4

    return 0.0


# ============================================================
# 5. Embedding Similarity Score (ESS)
# ============================================================
def embedding_similarity_score(similarity: Optional[float]) -> float:
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
    enable_self_agreement: bool = True,
    enable_ess: bool = True,
    mode: str = "no_exec",  # NEW
) -> Dict[str, Any]:

    svs = schema_validity_score(diagnostics)
    shs = structural_heuristic_score(model_sql)

    sas = (
        self_agreement_score(model_sql, sql_variants)
        if (enable_self_agreement and sql_variants)
        else 0.0
    )

    # EXECUTION SCORE ONLY IN EVAL MODE
    if mode == "eval":
        xbs = execution_behavior_score(exec_ok, row_count)
    else:
        xbs = 0.2  # lightweight signal for validated SQL only

    ess = (
        embedding_similarity_score(embedding_sim)
        if (enable_ess and embedding_sim is not None)
        else 0.0
    )

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
