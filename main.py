from fastapi import FastAPI, Query
from pydantic import BaseModel

from calibration import calibrated_confidence
from config import ENABLE_ESS, ENABLE_SELF_AGREEMENT, ESS_TOP_K
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
    debug: bool = Query(False, description="Include history of repair attempts")
):
    """Full pipeline: generate + validate + auto-repair."""

    result = generate_full_pipeline(req.question)

    sql = result.get("sql") or ""
    diagnostics = result.get("diagnostics") or {}
    exec_ok = result.get("exec_error") is None

    # Self-agreement
    if ENABLE_SELF_AGREEMENT:
        sql_variants = generate_sql_variants(req.question)
    else:
        sql_variants = []

    # ESS
    if ENABLE_ESS:
        ess_value = embedding_similarity(sql, top_k=ESS_TOP_K)
    else:
        ess_value = None

    # --- SAME CONFIDENCE MODE AS EVAL ---
    score = calibrated_confidence(
        model_sql=sql,
        diagnostics=diagnostics,
        exec_ok=exec_ok,
        row_count=None,
        sql_variants=sql_variants,
        embedding_sim=ess_value,
        enable_self_agreement=ENABLE_SELF_AGREEMENT,
        enable_ess=ENABLE_ESS,
        mode="no_exec" 
    )

    response = {
        "sql": sql,
        "validated": result.get("validated"),
        "repaired": result.get("repaired"),
        "attempts": result.get("attempts"),
        "diagnostics": diagnostics,
        "exec_error": result.get("exec_error"),
        "confidence": score["confidence"],
        "confidence_components": score["components"]
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
