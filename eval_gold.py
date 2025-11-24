import json
import glob
import argparse
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

TRAIN_DIR = "data/test/"
RESULT_DIR = "data/result/"

from config import (
    ENABLE_SELF_AGREEMENT,
    SELF_AGREEMENT_VARIANTS,
    ENABLE_ESS,
    ESS_TOP_K,
)

from sql_service import (
    generate_full_pipeline,
    generate_sql_variants,
    embedding_similarity,
)

from db import run_query
from calibration import calibrated_confidence
from confidence_service import get_confidence_service


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _normalize_sql(sql: Optional[str]) -> str:
    if not sql:
        return ""
    s = sql.strip().rstrip(";")
    return " ".join(s.split())


def load_test_files(file_list: Optional[List[str]]) -> List[str]:
    """Return list of test files to evaluate."""
    if file_list:
        resolved = []
        for f in file_list:
            path = os.path.join(TRAIN_DIR, f)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Test file not found: {path}")
            resolved.append(path)
        return resolved

    # fallback: load all JSONs in folder
    return glob.glob(os.path.join(TRAIN_DIR, "*.json"))


def save_results(input_filename: str, data: List[Dict[str, Any]]):
    os.makedirs(RESULT_DIR, exist_ok=True)
    base = os.path.basename(input_filename)
    out_name = f"result_{base}"
    out_path = os.path.join(RESULT_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tqdm.write(f"Saved results → {out_path}")


# ---------------------------------------------------------
# Core evaluation logic for a single file
# ---------------------------------------------------------
def eval_file(test_file: str):
    tqdm.write(f"\n=== Evaluating: {test_file} ===")

    with open(test_file, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    calibrator = get_confidence_service()

    # TQDM PROGRESS BAR FOR ITEMS (WITH POSITION)
    for item in tqdm(
        data,
        desc=f"Processing queries in {os.path.basename(test_file)}",
        position=1,       # inner bar stays on line 1
        leave=False       # auto-clear inner bar after each file
    ):
        q = item["question"]
        gold_sql = item["gold_sql"]

        item.update({
            "model_sql": "",
            "validated": False,
            "model_exec_ok": False,
            "result_match": False,
            "gold_error": None,
            "model_error": None,
            "diagnostics": None,
            "attempts": None,
            "repaired": None,
            "gold_row_count": None,
            "model_row_count": None,
            "sql_exact_match": False,
            "model_correct": False,
            "confidence_raw": None,
            "confidence": None,
            "confidence_used_calibrator": None,
            "confidence_components": None,
        })

        # 1. Generate SQL
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
            continue

        # 2. Execute Gold SQL
        gold_result = None
        try:
            gold_result = run_query(gold_sql)
            item["gold_row_count"] = len(gold_result)
        except Exception as ex:
            item["gold_error"] = str(ex)

        # 3. Execute Model SQL
        model_result = None
        try:
            if item["model_sql"]:
                model_result = run_query(item["model_sql"])
                item["model_row_count"] = len(model_result)
                item["model_exec_ok"] = True
        except Exception as ex:
            item["model_error"] = f"execution error: {ex}"

        # 4. Strict match
        if (
            item["gold_error"] is None
            and gold_result is not None
            and model_result is not None
        ):
            item["result_match"] = (gold_result == model_result)

        # 5. SQL exact match
        if _normalize_sql(gold_sql) == _normalize_sql(item["model_sql"]):
            item["sql_exact_match"] = True

        # 6. Relaxed correctness
        if (
            item["gold_error"] is None
            and item["model_exec_ok"]
            and item["gold_row_count"] is not None
            and item["model_row_count"] is not None
        ):
            item["model_correct"] = (
                item["gold_row_count"] == item["model_row_count"]
            )

        # 7. Confidence scoring
        if ENABLE_SELF_AGREEMENT and not item["repaired"]:
            try:
                sql_variants = generate_sql_variants(q, n=SELF_AGREEMENT_VARIANTS)
            except Exception:
                sql_variants = []
        else:
            sql_variants = []

        ess_value = None
        if ENABLE_ESS:
            try:
                ess_value = embedding_similarity(
                    item["model_sql"], top_k=ESS_TOP_K
                )
            except Exception:
                ess_value = None

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

            conf_result = calibrator.compute_confidence(
                raw_confidence=raw_score["confidence"],
                components=raw_score["components"]
            )

            item["confidence_raw"] = conf_result.raw
            item["confidence"] = conf_result.calibrated
            item["confidence_used_calibrator"] = conf_result.used_calibrator
            item["confidence_components"] = conf_result.components

        except Exception as ex:
            item["confidence_components"] = {"error": str(ex)}

    save_results(test_file, data)


# ---------------------------------------------------------
# Main: multi-file function + top-level TQDM
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-file",
        nargs="*",
        help="One or more test files inside data/test/. Example: -file test1.json test2.json",
    )
    args = parser.parse_args()

    test_files = load_test_files(args.file)
    tqdm.write(f"\nFound {len(test_files)} file(s) to evaluate.\n")

    # Outer bar locked at bottom using position=0
    for file_path in tqdm(
        test_files,
        desc="Evaluating files",
        position=0
    ):
        eval_file(file_path)


if __name__ == "__main__":
    main()
