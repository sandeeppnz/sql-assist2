import json
from typing import Any, Dict, List

from sql_service import generate_full_pipeline
from db import run_query


def eval_gold() -> None:
    with open("gold_eval.json", "r", encoding="utf-8") as f:
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
            item["gold_error"] = str(ex)

        # --- 3) Execute model SQL ---
        model_result = None
        try:
            if item["model_sql"]:
                model_result = run_query(item["model_sql"])
                item["model_row_count"] = len(model_result)
                item["model_exec_ok"] = True
            else:
                model_result = None
        except Exception as ex:
            item["model_exec_ok"] = False
            item["model_error"] = f"execution error: {ex}"
            continue

        # --- 4) Compare results if both ran ---
        if gold_result is not None and model_result is not None:
            item["result_match"] = (gold_result == model_result)

    # --- 5) Save results ---
    with open("adventureworks_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    total = len(data)
    ok = sum(1 for x in data if x.get("result_match"))
    print(f"Total: {total}, Result match: {ok}, Accuracy: {ok/total if total else 0:.2%}")

    # Optional: log the failing IDs/questions to console
    print("\\nFailures:")
    for item in data:
        if not item.get("result_match"):
            print(f"- id={item['id']}: {item['question']}")


if __name__ == "__main__":
    eval_gold()
