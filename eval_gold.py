import json
from typing import Any, Dict, List, Optional

from config import GOLD_TEST_FILE
from sql_service import generate_full_pipeline
from db import run_query


def _normalize_sql(sql: Optional[str]) -> str:
    """Simple SQL normalizer: strip, remove trailing ';', squeeze whitespace."""
    if not sql:
        return ""
    s = sql.strip().rstrip(";")
    # Collapse all whitespace to single spaces
    return " ".join(s.split())


def eval_gold() -> None:
    with open(GOLD_TEST_FILE, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    for item in data:
        q = item["question"]
        gold_sql = item["gold_sql"]

        # Reset / ensure fields exist
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
        item["model_correct"] = False  # relaxed correctness

        # --- 1) Generate + validate + (auto-)repair ---
        try:
            result = generate_full_pipeline(q)
            item["model_sql"] = result.get("sql") or ""
            item["validated"] = bool(result.get("validated"))
            item["diagnostics"] = result.get("diagnostics")
            item["attempts"] = result.get("attempts")
            item["repaired"] = result.get("repaired")
        except Exception as ex:
            item["model_error"] = f"generation/validation error: {ex}"
            continue  # can't go further without SQL

        # --- 2) Execute gold SQL ---
        gold_result = None
        try:
            gold_result = run_query(gold_sql)
            item["gold_row_count"] = len(gold_result)
            item["gold_error"] = None
        except Exception as ex:
            # gold is considered "bad"; we will exclude this row from accuracy
            item["gold_error"] = str(ex)

        # --- 3) Execute model SQL ---
        model_result = None
        try:
            if item["model_sql"]:
                model_result = run_query(item["model_sql"])
                item["model_row_count"] = len(model_result)
                item["model_exec_ok"] = True
        except Exception as ex:
            item["model_exec_ok"] = False
            item["model_error"] = f"execution error: {ex}"

        # --- 4) Compare strict results if both ran and gold had no error ---
        if (
            item["gold_error"] is None
            and gold_result is not None
            and model_result is not None
        ):
            item["result_match"] = (gold_result == model_result)

        # --- 5) Exact SQL text match (normalized) ---
        norm_gold = _normalize_sql(gold_sql)
        norm_model = _normalize_sql(item["model_sql"])
        if norm_gold and norm_model and norm_gold == norm_model:
            item["sql_exact_match"] = True

        # --- 6) Relaxed "model_correct" (row-count based, only when gold is valid) ---
        if (
            item["gold_error"] is None
            and item["model_exec_ok"]
            and item["gold_row_count"] is not None
            and item["model_row_count"] is not None
        ):
            # relaxed correctness: same number of rows
            item["model_correct"] = (item["gold_row_count"] == item["model_row_count"])
        else:
            item["model_correct"] = False

    # --- 7) Save per-question results ---
    with open("adventureworks_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # ========== REPORTING SECTION ==========
    total = len(data)
    gold_error_cases = [x for x in data if x.get("gold_error")]
    non_gold = [x for x in data if not x.get("gold_error")]

    # Primary: strict accuracy based on result_match, excluding gold errors
    strict_ok = [x for x in non_gold if x.get("result_match")]
    strict_fail = [x for x in non_gold if not x.get("result_match")]

    strict_acc = (len(strict_ok) / len(non_gold)) if non_gold else 0.0

    # Secondary: relaxed accuracy (row-count based), excluding gold errors
    relaxed_base = [x for x in non_gold if x.get("model_exec_ok")]
    relaxed_ok = [x for x in relaxed_base if x.get("model_correct")]
    relaxed_fail = [x for x in relaxed_base if not x.get("model_correct")]

    relaxed_acc = (len(relaxed_ok) / len(relaxed_base)) if relaxed_base else 0.0

    # Gold queries that ran but returned zero rows
    gold_zero_rows = [
        x
        for x in data
        if x.get("gold_error") is None
        and x.get("gold_row_count") is not None
        and x.get("gold_row_count") == 0
    ]

    print("=== SUMMARY ===")
    print(f"Total questions: {total}")
    print(f"Gold SQL error cases: {len(gold_error_cases)}")
    print(f"Evaluated (no gold error): {len(non_gold)}")
    print()

    print("Strict accuracy (result_match, excluding gold errors):")
    print(f"  {len(strict_ok)}/{len(non_gold)} = {strict_acc:.2%}")
    print()

    print("Relaxed accuracy (row-count match, excluding gold errors):")
    if relaxed_base:
        print(f"  {len(relaxed_ok)}/{len(relaxed_base)} = {relaxed_acc:.2%}")
    else:
        print("  (no cases where model_exec_ok and gold was valid)")
    print()

    # --- Gold SQL issues ---
    print("=== Gold SQL errors (excluded from accuracy) ===")
    if not gold_error_cases:
        print("  None ✅")
    else:
        for item in gold_error_cases:
            err_first_line = (item["gold_error"] or "").splitlines()[0][:140]
            print(f"- id={item['id']}: {item['question']}")
            print(f"    gold_error: {err_first_line}")
    print()

    # --- Gold zero-row cases ---
    print("=== Gold zero-row results (gold_row_count == 0, no gold_error) ===")
    if not gold_zero_rows:
        print("  None")
    else:
        for item in gold_zero_rows:
            print(f"- id={item['id']}: {item['question']}")
            print(
                f"    gold_row_count={item.get('gold_row_count')}, "
                f"model_row_count={item.get('model_row_count')}, "
                f"result_match={item.get('result_match')}"
            )
    print()

    # === Strict failures split by relaxed OK / FAIL ===
    strict_relaxed_ok = [x for x in strict_fail if x.get("model_correct")]
    strict_relaxed_fail = [x for x in strict_fail if not x.get("model_correct")]

    print("=== Strict failures — RELAXED_OK (row-count matched) ===")
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

    print("=== Strict failures — RELAXED_FAIL (row-count different) ===")
    if not strict_relaxed_fail:
        print("  None 🎯 (these are the important ones!)")
    else:
        for item in strict_relaxed_fail:
            print(f"- id={item['id']} [RELAXED_FAIL] {item['question']}")
            print(
                f"    rows: gold={item.get('gold_row_count')} "
                f"model={item.get('model_row_count')} "
                f"model_exec_ok={item.get('model_exec_ok')}"
            )
            if item.get("sql_exact_match"):
                print("    sql_exact_match: True")
            else:
                print("    sql_exact_match: False")
    print()



if __name__ == "__main__":
    eval_gold()
