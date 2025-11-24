import json
from typing import Any, Dict, List, Optional

from config import (
    GOLD_TEST_FILE,
    ENABLE_SELF_AGREEMENT,
    SELF_AGREEMENT_VARIANTS,
    ENABLE_ESS,
    ESS_TOP_K
)

from sql_service import (
    generate_full_pipeline,
    generate_sql_variants,
    embedding_similarity,
)

from db import run_query
from calibration import calibrated_confidence
from confidence_service import get_confidence_service


def _normalize_sql(sql: Optional[str]) -> str:
    """Strip, remove trailing ';', squeeze whitespace."""
    if not sql:
        return ""
    s = sql.strip().rstrip(";")
    return " ".join(s.split())


def eval_gold() -> None:
    with open(GOLD_TEST_FILE, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # ML calibrator singleton
    calibrator = get_confidence_service()

    for item in data:
        q = item["question"]
        gold_sql = item["gold_sql"]

        # Reset fields
        item["model_sql"] = ""
        item["validated"] = False
        item["model_exec_ok"] = False
        item["result_match"] = False
        item["gold_error"] = None
        item["model_error"] = None
        item["diagnostics"] = None
        item["attempts"] = None
        item["repaired"] = None
        item["gold_row_count"] = None
        item["model_row_count"] = None
        item["sql_exact_match"] = False
        item["model_correct"] = False

        # Confidence fields
        item["confidence_raw"] = None
        item["confidence"] = None
        item["confidence_used_calibrator"] = None
        item["confidence_components"] = None

        # =======================================================
        # 1) Generate SQL
        # =======================================================
        try:
            result = generate_full_pipeline(q)

            item["model_sql"] = result.get("sql") or ""
            item["validated"] = bool(result.get("validated"))
            item["diagnostics"] = result.get("diagnostics")
            item["attempts"] = result.get("attempts")
            item["repaired"] = result.get("repaired")
            item["history"] = result.get("history") or []

        except Exception as ex:
            item["model_error"] = f"generation/validation error: {ex}"
            continue  # Cannot continue without SQL

        # =======================================================
        # 2) Execute Gold SQL
        # =======================================================
        gold_result = None
        try:
            gold_result = run_query(gold_sql)
            item["gold_row_count"] = len(gold_result)
            item["gold_error"] = None
        except Exception as ex:
            item["gold_error"] = str(ex)

        # =======================================================
        # 3) Execute Model SQL
        # =======================================================
        model_result = None
        try:
            if item["model_sql"]:
                model_result = run_query(item["model_sql"])
                item["model_row_count"] = len(model_result)
                item["model_exec_ok"] = True
        except Exception as ex:
            item["model_exec_ok"] = False
            item["model_error"] = f"execution error: {ex}"

        # =======================================================
        # 4) Strict comparison
        # =======================================================
        if (
            item["gold_error"] is None
            and gold_result is not None
            and model_result is not None
        ):
            item["result_match"] = (gold_result == model_result)

        # =======================================================
        # 5) Exact SQL match (normalized)
        # =======================================================
        norm_gold = _normalize_sql(gold_sql)
        norm_model = _normalize_sql(item["model_sql"])

        if norm_gold and norm_model and norm_gold == norm_model:
            item["sql_exact_match"] = True

        # =======================================================
        # 6) Relaxed correctness: row-count match
        # =======================================================
        if (
            item["gold_error"] is None
            and item["model_exec_ok"]
            and item["gold_row_count"] is not None
            and item["model_row_count"] is not None
        ):
            item["model_correct"] = (
                item["gold_row_count"] == item["model_row_count"]
            )
        else:
            item["model_correct"] = False

        # =======================================================
        # 7) Confidence Scoring (Raw + ML Calibrated)
        # =======================================================

        # ---- Self-Agreement (skip if repaired) ----
        if ENABLE_SELF_AGREEMENT and not item["repaired"]:
            try:
                sql_variants = generate_sql_variants(
                    q, n=SELF_AGREEMENT_VARIANTS
                )
            except Exception:
                sql_variants = []
        else:
            sql_variants = []

        # ---- Embedding Similarity ----
        if ENABLE_ESS:
            try:
                ess_value = embedding_similarity(
                    item["model_sql"], top_k=ESS_TOP_K
                )
            except Exception:
                ess_value = None
        else:
            ess_value = None

        # ---- RAW heuristic confidence (from calibration.py) ----
        try:
            raw_score = calibrated_confidence(
                model_sql=item["model_sql"],
                diagnostics=item["diagnostics"] or {},
                exec_ok=item["model_exec_ok"],
                row_count=item["model_row_count"],
                sql_variants=sql_variants,
                embedding_sim=ess_value,
                enable_self_agreement=ENABLE_SELF_AGREEMENT,
                enable_ess=ENABLE_ESS,
                mode="no_exec",
                repaired=item["repaired"],
            )

            raw_conf = raw_score["confidence"]
            components = raw_score["components"]

            # ---- ML calibrated confidence ----
            conf_result = calibrator.compute_confidence(
                raw_confidence=raw_conf,
                components=components
            )

            item["confidence_raw"] = conf_result.raw
            item["confidence"] = conf_result.calibrated
            item["confidence_used_calibrator"] = conf_result.used_calibrator
            item["confidence_components"] = conf_result.components

        except Exception as ex:
            item["confidence_raw"] = None
            item["confidence"] = None
            item["confidence_used_calibrator"] = False
            item["confidence_components"] = {"error": str(ex)}

    # =======================================================
    # 8) Save Results
    # =======================================================
    with open("adventureworks_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # =======================================================
    # 9) Reporting Section
    # (unchanged from your original)
    # =======================================================

    total = len(data)
    gold_error_cases = [x for x in data if x.get("gold_error")]
    non_gold = [x for x in data if not x.get("gold_error")]

    strict_ok = [x for x in non_gold if x.get("result_match")]
    strict_fail = [x for x in non_gold if not x.get("result_match")]

    strict_acc = (len(strict_ok) / len(non_gold)) if non_gold else 0.0

    relaxed_base = [x for x in non_gold if x.get("model_exec_ok")]
    relaxed_ok = [x for x in relaxed_base if x.get("model_correct")]
    relaxed_fail = [x for x in relaxed_base if not x.get("model_correct")]

    relaxed_acc = (len(relaxed_ok) / len(relaxed_base)) if relaxed_base else 0.0

    gold_zero_rows = [
        x for x in data
        if x.get("gold_error") is None
        and x.get("gold_row_count") is not None
        and x.get("gold_row_count") == 0
    ]

    print("=== SUMMARY ===")
    print(f"Total questions: {total}")
    print(f"Gold SQL error cases: {len(gold_error_cases)}")
    print(f"Evaluated (no gold error): {len(non_gold)}")
    print()

    print("Strict accuracy (result_match):")
    print(f"  {len(strict_ok)}/{len(non_gold)} = {strict_acc:.2%}")
    print()

    print("Relaxed accuracy (row-count match):")
    if relaxed_base:
        print(f"  {len(relaxed_ok)}/{len(relaxed_base)} = {relaxed_acc:.2%}")
    else:
        print("  (no valid model_exec_ok cases)")
    print()

    print("=== Gold SQL errors ===")
    if not gold_error_cases:
        print("  None")
    else:
        for item in gold_error_cases:
            print(f"- id={item['id']}: {item['question']}")
            print(f"    gold_error: {(item['gold_error'] or '').splitlines()[0]}")
    print()

    print("=== Gold zero-row results ===")
    if not gold_zero_rows:
        print("  None")
    else:
        for item in gold_zero_rows:
            print(f"- id={item['id']}: {item['question']}")
            print(
                f"    gold_row_count={item.get('gold_row_count')} "
                f"model_row_count={item.get('model_row_count')} "
                f"result_match={item.get('result_match')}"
            )
    print()

    strict_relaxed_ok = [x for x in strict_fail if x.get("model_correct")]
    strict_relaxed_fail = [x for x in strict_fail if not x.get("model_correct")]

    print("=== Strict failures — RELAXED_OK ===")
    if not strict_relaxed_ok:
        print("  None")
    else:
        for item in strict_relaxed_ok:
            print(f"- id={item['id']} [RELAXED_OK] {item['question']}")
            print(
                f"    rows: gold={item.get('gold_row_count')} "
                f"model={item.get('model_row_count')}"
            )
    print()

    print("=== Strict failures — RELAXED_FAIL ===")
    if not strict_relaxed_fail:
        print("  None 🎯")
    else:
        for item in strict_relaxed_fail:
            print(f"- id={item['id']} [RELAXED_FAIL] {item['question']}")
            print(
                f"    rows: gold={item.get('gold_row_count')} "
                f"model={item.get('model_row_count')} "
                f"model_exec_ok={item.get('model_exec_ok')}"
            )
            print(
                "    sql_exact_match: "
                + ("True" if item.get("sql_exact_match") else "False")
            )
    print()


if __name__ == "__main__":
    eval_gold()
