from fastapi import FastAPI, Query
from pydantic import BaseModel

from calibration_fast_variants import fast_self_agreement_variants
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
from db import run_query

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
    # if ENABLE_SELF_AGREEMENT and not is_repaired:
    #     sql_variants = generate_sql_variants(
    #         req.question,
    #         n=SELF_AGREEMENT_VARIANTS
    #     )
    # else:
    #     sql_variants = []

    if ENABLE_SELF_AGREEMENT and not is_repaired:
        # FAST MODE: NO LLM CALLS
        sql_variants = fast_self_agreement_variants(sql)
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


@app.post("/sql/eval_single")
def eval_single_endpoint(
    req: dict
):
    """
    Evaluate a single question + gold SQL pair.
    Returns the same metrics as eval_gold.py but for one item only.
    """
    question = req.get("question", "").strip()
    gold_sql = req.get("gold_sql", "").strip()

    if not question or not gold_sql:
        return {"error": "question and gold_sql are required"}

    # ------------------------------------------------------
    # 1) Generate SQL via full pipeline
    # ------------------------------------------------------
    result = generate_full_pipeline(question)

    model_sql = result.get("sql") or ""
    diagnostics = result.get("diagnostics")
    attempts = result.get("attempts")
    repaired = result.get("repaired")
    history = result.get("history") or []

    # ------------------------------------------------------
    # 2) Execute Gold SQL
    # ------------------------------------------------------
    try:
        gold_result = run_query(gold_sql)
        gold_row_count = len(gold_result)
        gold_error = None
    except Exception as ex:
        gold_result = None
        gold_row_count = None
        gold_error = str(ex)

    # ------------------------------------------------------
    # 3) Execute Model SQL
    # ------------------------------------------------------
    try:
        model_result = run_query(model_sql)
        model_row_count = len(model_result)
        model_exec_ok = True
        model_error = None
    except Exception as ex:
        model_result = None
        model_row_count = None
        model_exec_ok = False
        model_error = str(ex)

    # ------------------------------------------------------
    # 4) Strict match
    # ------------------------------------------------------
    strict_match = (
        gold_error is None
        and gold_result is not None
        and model_result is not None
        and gold_result == model_result
    )

    # ------------------------------------------------------
    # 5) Relaxed match (row count)
    # ------------------------------------------------------
    relaxed_match = (
        gold_error is None
        and model_exec_ok
        and gold_row_count is not None
        and model_row_count is not None
        and gold_row_count == model_row_count
    )

    # ------------------------------------------------------
    # 6) SQL exact match (normalized)
    # ------------------------------------------------------
    norm_gold = " ".join(gold_sql.strip().rstrip(";").split())
    norm_model = " ".join(model_sql.strip().rstrip(";").split())
    sql_exact_match = (norm_gold == norm_model)

    # ------------------------------------------------------
    # 7) Self-agreement (fast or slow)
    # ------------------------------------------------------
    if ENABLE_SELF_AGREEMENT and not repaired:
        # use FAST self-agreement mode (no LLM calls)
        sql_variants = fast_self_agreement_variants(model_sql)
    else:
        sql_variants = []

    # ------------------------------------------------------
    # 8) ESS
    # ------------------------------------------------------
    if ENABLE_ESS:
        ess_value = embedding_similarity(model_sql, top_k=ESS_TOP_K)
    else:
        ess_value = None

    # ------------------------------------------------------
    # 9) Confidence (raw + calibrated)
    # ------------------------------------------------------
    from confidence_service import get_confidence_service
    cs = get_confidence_service()

    raw_score = calibrated_confidence(
        model_sql=model_sql,
        diagnostics=diagnostics or {},
        exec_ok=model_exec_ok,
        row_count=model_row_count,
        sql_variants=sql_variants,
        embedding_sim=ess_value,
        enable_self_agreement=ENABLE_SELF_AGREEMENT,
        enable_ess=ENABLE_ESS,
        mode="no_exec",
        repaired=repaired,
    )

    calibrated = cs.compute_confidence(
        raw_confidence=raw_score["confidence"],
        components=raw_score["components"]
    )

    # ------------------------------------------------------
    # 10) Return full evaluation bundle for the question
    # ------------------------------------------------------
    return {
        "question": question,
        "gold_sql": gold_sql,
        "model_sql": model_sql,

        "validated": result.get("validated"),
        "repaired": repaired,
        "attempts": attempts,
        "diagnostics": diagnostics,
        "history": history,

        "gold_error": gold_error,
        "model_error": model_error,

        "gold_row_count": gold_row_count,
        "model_row_count": model_row_count,

        "strict_match": strict_match,
        "relaxed_match": relaxed_match,
        "sql_exact_match": sql_exact_match,

        "confidence_raw": calibrated.raw,
        "confidence": calibrated.calibrated,
        "confidence_used_calibrator": calibrated.used_calibrator,
        "confidence_components": calibrated.components,
    }
