from fastapi import FastAPI, Query
from pydantic import BaseModel

from confidence_service import get_confidence_service

from calibration import calibrated_confidence
from config import (
    ENABLE_ESS,
    ENABLE_SELF_AGREEMENT,
    ESS_TOP_K,
    SELF_AGREEMENT_VARIANTS
)

from sql_service import (
    embedding_similarity,
    generate_full_pipeline,
    generate_raw,
    generate_sql_variants,
    validate_only
)

app = FastAPI(title="Text-to-SQL Service (Ollama + Vanna)", version="0.1.0")


class QuestionRequest(BaseModel):
    question: str


class SqlRequest(BaseModel):
    sql: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sql/generate")
def generate_endpoint(
    req: QuestionRequest,
    debug: bool = Query(False, description="Include repair attempt history")
):
    """
    Full SQL generation pipeline:
    - generate
    - validate
    - repair (if needed)
    - confidence scoring
    """

    result = generate_full_pipeline(req.question)

    sql = result.get("sql") or ""
    diagnostics = result.get("diagnostics") or {}
    exec_ok = result.get("exec_error") is None
    is_repaired = bool(result.get("repaired"))

    # -------------------------------------------------------------
    # 1) SELF-AGREEMENT (disabled when SQL is repaired)
    # -------------------------------------------------------------
    if ENABLE_SELF_AGREEMENT and not is_repaired:
        sql_variants = generate_sql_variants(
            req.question,
            n=SELF_AGREEMENT_VARIANTS
        )
    else:
        sql_variants = []

    # -------------------------------------------------------------
    # 2) ESS (Embedding Similarity)
    # -------------------------------------------------------------
    if ENABLE_ESS:
        ess_value = embedding_similarity(sql, top_k=ESS_TOP_K)
    else:
        ess_value = None

    # -------------------------------------------------------------
    # 3) CONFIDENCE (same logic as eval)
    # -------------------------------------------------------------
    raw_score = calibrated_confidence(
        model_sql=sql,
        diagnostics=diagnostics,
        exec_ok=exec_ok,
        row_count=None,
        sql_variants=sql_variants,
        embedding_sim=ess_value,
        enable_self_agreement=ENABLE_SELF_AGREEMENT,
        enable_ess=ENABLE_ESS,
        mode="no_exec",
        repaired=is_repaired
    )

    raw_conf = raw_score["confidence"]
    components = raw_score["components"]

    cal = get_confidence_service()
    conf_result = cal.compute_confidence(
        raw_confidence=raw_conf,
        components=components
    )


    # -------------------------------------------------------------
    # 4) BUILD RESPONSE
    # -------------------------------------------------------------
    response = {
        "sql": sql,
        "validated": result.get("validated"),
        "repaired": is_repaired,
        "attempts": result.get("attempts"),
        "diagnostics": diagnostics,
        "exec_error": result.get("exec_error"),

        # Confidence scores
        "confidence_raw": conf_result.raw,
        "confidence": conf_result.calibrated,
        "confidence_used_calibrator": conf_result.used_calibrator,
        "confidence_components": conf_result.components,
    }

    if debug:
        response["history"] = result.get("history") or []

    return response


@app.post("/sql/generate_raw")
def generate_raw_endpoint(req: QuestionRequest):
    """Raw generation with no validation or repair."""
    return generate_raw(req.question)


@app.post("/sql/validate")
def validate_endpoint(req: SqlRequest):
    """Validate SQL without changing it."""
    return validate_only(req.sql)
