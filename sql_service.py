from typing import Any, Dict, List, Tuple

from sql_generator import generate_sql
from vanna_provider import generate_sql_from_prompt
from sql_validator import (
    is_safe_select,
    has_unknown_tables,
    has_unknown_columns,
    server_preflight_ok,
)
from schema_service import schema_service

MAX_REPAIR_ATTEMPTS = 3


def _validate_sql(sql: str) -> Tuple[bool, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {}

    is_safe = is_safe_select(sql)
    diagnostics["is_safe"] = is_safe

    has_bad_tables, unknown_tables = has_unknown_tables(sql)
    diagnostics["unknown_tables"] = unknown_tables

    has_bad_cols, unknown_cols = has_unknown_columns(sql)
    diagnostics["unknown_columns"] = [
        {"table": t, "alias": a, "column": c}
        for (t, a, c) in unknown_cols
    ]

    preflight_ok = True
    preflight_error = None
    if is_safe and not has_bad_tables and not has_bad_cols:
        preflight_ok, preflight_error = server_preflight_ok(sql)

    diagnostics["preflight_ok"] = preflight_ok
    diagnostics["preflight_error"] = preflight_error

    ok = (
        is_safe
        and not has_bad_tables
        and not has_bad_cols
        and preflight_ok
    )

    return ok, diagnostics


def _build_error_summary(diagnostics: Dict[str, Any]) -> str:
    unknown_tables = diagnostics.get("unknown_tables") or []
    unknown_cols = diagnostics.get("unknown_columns") or []
    preflight_error = diagnostics.get("preflight_error")

    lines: List[str] = []
    if unknown_tables:
        lines.append("Unknown tables: " + ", ".join(unknown_tables))
    if unknown_cols:
        cols_str = ", ".join(
            f"{item['table']}.{item['column']} (alias {item['alias']})"
            for item in unknown_cols
        )
        lines.append("Unknown columns: " + cols_str)
    if preflight_error:
        lines.append(f"SQL Server compile error: {preflight_error}")

    return "\n".join(lines) or "No explicit error summary available."


def _format_unknown_columns_for_prompt(unknown_cols: List[Dict[str, str]]) -> str:
    """
    Pretty-print the unknown columns from diagnostics["unknown_columns"]
    for use in the repair prompt.
    Each item is: {"table": t, "alias": a, "column": c}
    """
    if not unknown_cols:
        return "None."

    lines: List[str] = []
    for item in unknown_cols:
        table = item.get("table")
        alias = item.get("alias")
        col = item.get("column")
        if alias and alias != table:
            lines.append(f"- {alias}.{col} (table {table})")
        else:
            lines.append(f"- {table}.{col}")
    return "\n".join(lines)


def _repair_sql(question: str, bad_sql: str, diagnostics: Dict[str, Any]) -> str:
    """
    Try to repair an invalid SQL query using Vanna/Ollama.

    This version deliberately keeps the prompt simple and close to the
    original behaviour that gave you ~80% accuracy. It passes along the
    diagnostics as a short error summary, but does NOT over-constrain
    unknown_tables / unknown_columns.
    """
    error_summary = _build_error_summary(diagnostics)

    prompt = f"""You generated invalid SQL for a Microsoft SQL Server database.

        ORIGINAL QUESTION:
        {question}

        INVALID SQL:
        {bad_sql}

        ERROR DIAGNOSTICS:
        {error_summary}

        SCHEMA:
        {schema_service.schema_text}

        INSTRUCTIONS:
        - Generate ONE corrected T-SQL SELECT statement (CTEs allowed).
        - Use ONLY tables and columns from the SCHEMA.
        - Do NOT use LIMIT, OFFSET, CUBE, GROUPING SETS, or non-T-SQL constructs.
        - Always join to DimDate and filter on CalendarYear or FullDateAlternateKey when filtering by year or date.
        - Do NOT invent tables or columns.

        Return ONLY the corrected SQL. No explanations, no comments, no JSON.
        """

    repaired_sql = generate_sql_from_prompt(prompt)
    # Normalize whitespace a bit
    return " ".join(repaired_sql.split())


def generate_full_pipeline(question: str) -> Dict[str, Any]:
    history: List[Dict[str, Any]] = []

    sql = generate_sql(question)
    ok, diag = _validate_sql(sql)
    history.append({"sql": sql, "diagnostics": diag})

    if ok:
        return {
            "sql": sql,
            "validated": True,
            "repaired": False,
            "attempts": 1,
            "diagnostics": diag,
            "history": history,
        }

    attempts = 1
    while attempts < MAX_REPAIR_ATTEMPTS:
        attempts += 1
        sql = _repair_sql(question, sql, diag)
        ok, diag = _validate_sql(sql)
        history.append({"sql": sql, "diagnostics": diag})
        if ok:
            return {
                "sql": sql,
                "validated": True,
                "repaired": True,
                "attempts": attempts,
                "diagnostics": diag,
                "history": history,
            }

    return {
        "sql": None,
        "validated": False,
        "repaired": True,
        "attempts": attempts,
        "diagnostics": diag,
        "history": history,
    }


def generate_raw(question: str) -> Dict[str, Any]:
    sql = generate_sql(question)
    return {"sql": sql}


def validate_only(sql: str) -> Dict[str, Any]:
    ok, diag = _validate_sql(sql)
    return {"sql": sql, "validated": ok, "diagnostics": diag}
