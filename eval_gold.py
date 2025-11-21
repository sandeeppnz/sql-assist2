import json
from typing import Any, Dict, List, Optional

from sql_service import generate_full_pipeline
from db import run_query


def _normalize_sql(sql: Optional[str]) -> str:
    """
    Simple SQL normalizer for equality checks:
    - handle None
    - strip leading/trailing whitespace
    - remove trailing semicolon
    - collapse internal whitespace
    - lowercase
    """
    if not sql:
        return ""
    s = sql.strip().rstrip(";")
    # collapse whitespace
    parts = s.split()
    return " ".join(parts).lower()


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

        # New fields
        item["sql_exact_match"] = False
        item["model_correct"] = False  # our new accuracy notion

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
            # No model_sql => can't execute or compare
            continue

        # --- 2) Execute gold SQL ---
        gold_result = None
        try:
            gold_result = run_query(gold_sql)
            item["gold_row_count"] = len(gold_result)
            item["gold_error"] = None
        except Exception as ex:
            # gold is "bad" or at least not executable
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
            # Cannot compare results, but we still may mark gold as bad later.
            # model_correct will remain False in this case.
            continue

        # --- 4) Compare results if both ran successfully ---
        if gold_result is not None and model_result is not None:
            item["result_match"] = (gold_result == model_result)
        else:
            item["result_match"] = False

        # --- 5) Exact SQL match (normalized) ---
        norm_gold = _normalize_sql(gold_sql)
        norm_model = _normalize_sql(item["model_sql"])
        item["sql_exact_match"] = bool(norm_model and norm_model == norm_gold)

        # --- 6) Model correctness (independent of gold being right/wrong) ---
        gold_ok = gold_result is not None and item["gold_error"] is None
        model_ok = model_result is not None and item["model_exec_ok"]

        # Heuristic:
        # - Model must be executable and validated
        # - If gold is good, we require result_match
        # - If gold fails, but model is valid & executable, we still give credit
        if model_ok and item["validated"]:
            if item["result_match"]:
                # Both gold & model ran and agree on result
                item["model_correct"] = True
            elif not gold_ok:
                # Gold is broken / errored, but model ran fine => give credit
                item["model_correct"] = True
            else:
                item["model_correct"] = False
        else:
            item["model_correct"] = False

    # --- 7) Save results ---
    with open("adventureworks_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # --- 8) Aggregate metrics ---
    total = len(data)

    semantic_ok = sum(1 for x in data if x.get("result_match"))
    model_correct_ok = sum(1 for x in data if x.get("model_correct"))
    sql_exact_ok = sum(1 for x in data if x.get("sql_exact_match"))

    def _acc(ok: int) -> float:
        return ok / total if total else 0.0

    print(f"Total examples: {total}")
    print(f"Semantic accuracy (match gold results): {semantic_ok}/{total} = {_acc(semantic_ok):.2%}")
    print(f"Model-correct accuracy (model valid even if gold fails): {model_correct_ok}/{total} = {_acc(model_correct_ok):.2%}")
    print(f"Exact SQL match rate: {sql_exact_ok}/{total} = {_acc(sql_exact_ok):.2%}")

    # --- 9) Failure breakdowns ---
    print("\nSemantic failures (result_match = False):")
    for item in data:
        if not item.get("result_match"):
            print(f"- id={item['id']}: {item['question']}")

    print("\nModel-correct failures (model_correct = False):")
    for item in data:
        if not item.get("model_correct"):
            print(f"- id={item['id']}: {item['question']} (gold_error={item.get('gold_error')}, model_error={item.get('model_error')})")


if __name__ == "__main__":
    eval_gold()
