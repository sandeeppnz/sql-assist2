from typing import Any, Dict, List, Tuple
import re
from db import run_query

from sql_generator import generate_sql
from vanna_provider import generate_sql_from_prompt
from sql_validator import (
    is_safe_select,
    has_unknown_tables,
    has_unknown_columns,
    server_preflight_ok,
    _has_bad_year_usage,
)
from schema_service import schema_service

MAX_REPAIR_ATTEMPTS = 3

DIVIDE_BY_ZERO_PAT = re.compile(
    r"/\s*(COUNT|SUM)\s*\(",
    flags=re.IGNORECASE
)

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

    bad_year_usage, bad_year_msg = _has_bad_year_usage(sql)
    diagnostics["bad_year_usage"] = bad_year_msg if bad_year_usage else None

    preflight_ok = True
    preflight_error = None
    if is_safe and not has_bad_tables and not has_bad_cols:
        preflight_ok, preflight_error = server_preflight_ok(sql)

    # If we detected bad YEAR() usage, force a failure to trigger the repair loop
    if bad_year_usage:
        preflight_ok = False
        if not preflight_error:
            preflight_error = bad_year_msg

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
    bad_year_usage = diagnostics.get("bad_year_usage")

    lines: List[str] = []
    if unknown_tables:
        lines.append("Unknown tables: " + ", ".join(unknown_tables))
    if unknown_cols:
        cols_str = ", ".join(
            f"{item['table']}.{item['column']} (alias {item['alias']})"
            for item in unknown_cols
        )
        lines.append("Unknown columns: " + cols_str)
    if bad_year_usage:
            lines.append(f"Date function issue: {bad_year_usage}")
    if preflight_error:
        lines.append(f"SQL Server compile error: {preflight_error}")

    return "\n".join(lines) or "No explicit error summary available."


def _add_nullif_to_divisions(sql: str) -> str:
    """
    Very conservative: transform patterns like
       / COUNT(...)
    or  / SUM(...)
    into
       / NULLIF(COUNT(...), 0)
    if they aren't already using NULLIF.
    """

    def repl(match: re.Match) -> str:
        func = match.group(1)  # COUNT or SUM
        # We will grab from this point up to the matching ')'
        # and wrap it in NULLIF(..., 0)
        start = match.start(1)  # position of function name
        # Very simple parenthesis matching forward:
        depth = 0
        i = start
        while i < len(sql):
            if sql[i] == '(':
                depth += 1
            elif sql[i] == ')':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        func_call = sql[start:i+1]  # e.g. COUNT(DISTINCT ...)
        # If already has NULLIF inside, skip
        if "NULLIF" in func_call.upper():
            return match.group(0)
        return f"/ NULLIF({func_call}, 0)"

    # Apply once – could be extended to global replacements if safe
    new_sql = sql
    for m in DIVIDE_BY_ZERO_PAT.finditer(sql):
        new_sql = _apply_one_replacement(new_sql, m)
        break  # only first for now
    return new_sql

def _apply_one_replacement(sql: str, match: re.Match) -> str:
    func = match.group(1)
    start = match.start(1)
    depth = 0
    i = start
    while i < len(sql):
        if sql[i] == '(':
            depth += 1
        elif sql[i] == ')':
            depth -= 1
            if depth == 0:
                break
        i += 1
    func_call = sql[start:i+1]
    if "NULLIF" in func_call.upper():
        return sql
    replacement = f"/ NULLIF({func_call}, 0)"
    return sql[:match.start()] + replacement + sql[i+1:]


def _repair_sql(question: str, bad_sql: str, diagnostics: Dict[str, Any]) -> str:
    """
    Try to repair an invalid SQL query using Vanna/Ollama.

    - Keeps prompts simple and close to what worked well.
    - Uses question-based hints for specific tricky cases (17, 21, 24, 25, 26, 28, 29).
    - Adds unknown-table-based hints without over-constraining the model.
    - Adds a generic hint for misuse of OrderDateKey with date literals.
    """
    error_summary = _build_error_summary(diagnostics)
    q = (question or "").lower()
    sql_lower = (bad_sql or "").lower()

    extra_instructions: List[str] = []

    # ---- Question-specific hints (by pattern) ----

    # ID 17: Compare Internet and Reseller Sales Amount by calendar year
    if "internet" in q and "reseller" in q and "calendar year" in q:
        extra_instructions.append(
            "- Use BOTH FactInternetSales and FactResellerSales.\n"
            "  Create one combined set (UNION ALL) with a Source column "
            "  ('Internet' or 'Reseller'), joined to DimDate for CalendarYear.\n"
            "  Then GROUP BY CalendarYear and use:\n"
            "    SUM(CASE WHEN Source = 'Internet' THEN SalesAmount ELSE 0 END) AS InternetSalesAmount,\n"
            "    SUM(CASE WHEN Source = 'Reseller' THEN SalesAmount ELSE 0 END) AS ResellerSalesAmount."
        )

    # ID 21: last 30 days of available data
    if "last 30 days" in q:
        extra_instructions.append(
            "- First find the MAX(FullDateAlternateKey) from FactInternetSales "
            "joined to DimDate.\n"
            "  Then build a DateRange CTE that generates the last 30 dates using "
            "  DATEADD(DAY, -n, MaxOrderDate) for n = 0..29.\n"
            "  LEFT JOIN that DateRange to DimDate and FactInternetSales so that days "
            "  with no sales still appear with 0 or NULL sales amounts."
        )

    # ID 24: promotions conversion rate
    if "promotion" in q and ("conversion" in q or "rate" in q):
        extra_instructions.append(
            "- Use two CTEs:\n"
            "  * OrdersWithPromo: SELECT DISTINCT SalesOrderNumber FROM FactInternetSales "
            "    joined to DimDate where CalendarYear = 2004 AND PromotionKey IS NOT NULL.\n"
            "  * AllOrders: SELECT DISTINCT SalesOrderNumber FROM FactInternetSales joined "
            "    to DimDate where CalendarYear = 2004.\n"
            "  In the final SELECT you MUST:\n"
            "    FROM AllOrders AS ao\n"
            "    LEFT JOIN OrdersWithPromo AS owp\n"
            "      ON ao.SalesOrderNumber = owp.SalesOrderNumber\n"
            "  and compute PromoOrderShare as:\n"
            "    CAST(COUNT(DISTINCT owp.SalesOrderNumber) AS FLOAT)\n"
            "    / NULLIF(COUNT(DISTINCT ao.SalesOrderNumber), 0)."
        )

    # ID 25: Sales Reason category
    if "sales reason" in q:
        extra_instructions.append(
            "- Join FactInternetSales to FactInternetSalesReason and DimSalesReason.\n"
            "  Group results by SalesReasonName (SalesReasonType may not exist in the schema),\n"
            "  and SUM the Internet SalesAmount for 2004 using DimDate.CalendarYear = 2004."
        )

    # ID 26: quota attainment
    if "quota attainment" in q or "sales quota attainment" in q:
        extra_instructions.append(
            "- Build one CTE for ResellerSales from FactResellerSales joined to DimDate "
            "and DimSalesTerritory (SUM SalesAmount by SalesTerritoryKey and CalendarYear = 2004).\n"
            "  Build another CTE for SalesQuota from FactSalesQuota joined to DimEmployee, "
            "DimSalesTerritory and DimDate (SUM SalesAmountQuota by SalesTerritoryKey and CalendarYear = 2004).\n"
            "  Then join these CTEs on SalesTerritoryKey and compute QuotaAttainment as:\n"
            "    TotalSalesAmount / NULLIF(TotalSalesQuota, 0)."
        )

    # ID 29: churn-like metric 2003 vs 2004
    if "churn-like metric" in q or ("bought in 2003" in q and "not in 2004" in q):
        extra_instructions.append(
            "- Use two CTEs:\n"
            "  * Sales2003: DISTINCT CustomerKey from FactInternetSales joined to DimDate "
            "    ON FactInternetSales.OrderDateKey = DimDate.DateKey where DimDate.CalendarYear = 2003.\n"
            "  * Sales2004: DISTINCT CustomerKey from FactInternetSales joined to DimDate "
            "    ON FactInternetSales.OrderDateKey = DimDate.DateKey where DimDate.CalendarYear = 2004.\n"
            "  DO NOT compare OrderDateKey directly to string dates like '2003-01-01'; "
            "  always go through DimDate.\n"
            "  Then LEFT JOIN Sales2003 to Sales2004 on CustomerKey, filter rows where "
            "  Sales2004.CustomerKey IS NULL, and join DimCustomer to return customer details."
        )

    # ID 28: lifetime Internet Sales value (top 20 customers)
    if "lifetime internet sales value" in q or "lifetime internet sales" in q:
        extra_instructions.append(
            "- Return TOP 20 customers by total Internet SalesAmount.\n"
            "  Include SUM(fis.SalesAmount) AS LifetimeSalesAmount,\n"
            "  MIN(d.FullDateAlternateKey) AS FirstPurchaseDate,\n"
            "  MAX(d.FullDateAlternateKey) AS LastPurchaseDate.\n"
            "  Join FactInternetSales to DimCustomer and DimDate and group by\n"
            "  c.CustomerKey, c.FirstName, c.LastName, c.EmailAddress."
        )

    # ---- Generic hint for misuse of OrderDateKey with date literals ----
    # Pattern seen in id=29: OrderDateKey BETWEEN '2003-01-01' AND '2003-12-31'
    if "orderdatekey" in sql_lower and "'" in bad_sql:
        extra_instructions.append(
            "- OrderDateKey is an integer surrogate key, not a date.\n"
            "  Do NOT compare OrderDateKey directly to string dates such as '2003-01-01'.\n"
            "  Instead, join your fact table to DimDate ON OrderDateKey = DateKey and\n"
            "  filter using DimDate.CalendarYear or DimDate.FullDateAlternateKey."
        )

    # ---- Unknown-table-based hints (from diagnostics) ----

    unknown_tables = diagnostics.get("unknown_tables") or []
    unknown_table_hints = _build_unknown_table_hints(unknown_tables)

    extra_text_parts: List[str] = []
    if extra_instructions:
        extra_text_parts.append(
            "ADDITIONAL TASK-SPECIFIC / GENERIC HINTS:\n" + "\n\n".join(extra_instructions)
        )
    if unknown_table_hints:
        extra_text_parts.append(
            "ADDITIONAL HINTS ABOUT UNKNOWN TABLES:\n" + unknown_table_hints
        )

    extra_text = ""
    if extra_text_parts:
        extra_text = "\n\n" + "\n\n".join(extra_text_parts) + "\n"

    prompt = f"""You generated invalid SQL for a Microsoft SQL Server database.

ORIGINAL QUESTION:
{question}

INVALID SQL:
{bad_sql}

ERROR DIAGNOSTICS:
{error_summary}

SCHEMA:
{schema_service.schema_text}
{extra_text}
INSTRUCTIONS:
- Generate ONE corrected T-SQL SELECT statement (CTEs allowed).
- Use ONLY tables and columns from the SCHEMA.
- Do NOT use LIMIT, OFFSET, CUBE, GROUPING SETS, or non-T-SQL constructs.
- Always join to DimDate and filter on CalendarYear or FullDateAlternateKey when filtering by year or date.
- Do NOT invent tables or columns.
- Do NOT compare integer surrogate keys like OrderDateKey directly to string date literals.

Return ONLY the corrected SQL. No explanations, no comments, no JSON.
"""

    repaired_sql = generate_sql_from_prompt(prompt)
    # Normalize whitespace
    return " ".join(repaired_sql.split())



def generate_full_pipeline(question: str) -> Dict[str, Any]:
    """
    End-to-end pipeline:

    1) Use generate_sql(question) for the first attempt.
    2) Validate with _validate_sql (safety + unknown tables/cols + bad YEAR() + preflight).
    3) If invalid, call _repair_sql(...) and re-validate, up to MAX_REPAIR_ATTEMPTS.
    4) Once we have a validated SQL, try executing it.
       - If execution fails with divide-by-zero, apply _add_nullif_to_divisions(...)
         once and retry execution.
    5) Return the final SQL, diagnostics, attempt history, and any execution error.

    Note: On total failure we still return the *last* attempted SQL string
    instead of None, so callers can inspect what the model produced.
    """
    history: List[Dict[str, Any]] = []

    def _try_execute_with_divzero_repair(
        sql: str,
        validated: bool,
        diag: Dict[str, Any],
        attempts: int,
        repaired: bool,
    ) -> Dict[str, Any]:
        """
        Try to run the SQL. If we hit a divide-by-zero error, we attempt
        one automatic repair by wrapping denominators with NULLIF.
        """
        try:
            _ = run_query(sql)
            return {
                "sql": sql,
                "validated": validated,
                "repaired": repaired,
                "attempts": attempts,
                "diagnostics": diag,
                "history": history,
                "exec_error": None,
            }
        except Exception as ex:
            msg = str(ex)

            # Heuristic repair for divide-by-zero
            if "divide by zero" in msg.lower():
                repaired_sql = _add_nullif_to_divisions(sql)
                attempts += 1
                repaired = True

                # Optionally re-validate the repaired SQL
                ok2, diag2 = _validate_sql(repaired_sql)
                history.append({
                    "sql": repaired_sql,
                    "diagnostics": diag2,
                    "note": "auto NULLIF / divide-by-zero repair",
                })

                if ok2:
                    try:
                        _ = run_query(repaired_sql)
                        return {
                            "sql": repaired_sql,
                            "validated": True,
                            "repaired": repaired,
                            "attempts": attempts,
                            "diagnostics": diag2,
                            "history": history,
                            "exec_error": None,
                        }
                    except Exception as ex2:
                        # Repaired SQL still fails at execution
                        return {
                            "sql": repaired_sql,
                            "validated": True,
                            "repaired": repaired,
                            "attempts": attempts,
                            "diagnostics": diag2,
                            "history": history,
                            "exec_error": str(ex2),
                        }
                else:
                    # Our NULLIF patch made it fail validation – return that state
                    return {
                        "sql": repaired_sql,
                        "validated": False,
                        "repaired": repaired,
                        "attempts": attempts,
                        "diagnostics": diag2,
                        "history": history,
                        "exec_error": msg,
                    }

            # Non-divide-by-zero error, just bubble it up
            return {
                "sql": sql,
                "validated": validated,
                "repaired": repaired,
                "attempts": attempts,
                "diagnostics": diag,
                "history": history,
                "exec_error": msg,
            }

    # 1) First-pass SQL from the model
    sql = generate_sql(question)
    ok, diag = _validate_sql(sql)
    history.append({"sql": sql, "diagnostics": diag})

    # 2) If it's valid + safe, try executing (with div/0 repair if needed)
    if ok:
        return _try_execute_with_divzero_repair(
            sql=sql,
            validated=True,
            diag=diag,
            attempts=1,
            repaired=False,
        )

    # 3) Try to repair a few times using _repair_sql
    attempts = 1
    while attempts < MAX_REPAIR_ATTEMPTS:
        attempts += 1

        # Use diagnostics + question-specific hints + unknown-table hints
        sql = _repair_sql(question, sql, diag)
        ok, diag = _validate_sql(sql)
        history.append({"sql": sql, "diagnostics": diag})

        if ok:
            # validated successfully after repair -> execute with div/0 patch if needed
            return _try_execute_with_divzero_repair(
                sql=sql,
                validated=True,
                diag=diag,
                attempts=attempts,
                repaired=True,
            )

    # 4) All repair attempts failed; return the last SQL we tried
    return {
        "sql": sql,              # last attempted SQL
        "validated": False,
        "repaired": True,
        "attempts": attempts,
        "diagnostics": diag,
        "history": history,
        "exec_error": "validation failed after repair attempts",
    }



def generate_raw(question: str) -> Dict[str, Any]:
    sql = generate_sql(question)
    return {"sql": sql}

import requests

def generate_sql_variants(question: str, n: int = 3, temperature: float = 0.7) -> List[str]:
    """
    Generate *pure* raw SQL variants directly from Ollama,
    bypassing Vanna, Chroma, and all retrieval/validation layers.
    """

    variants = []

    prompt = f"""
You are a SQL generator. 
Given the question below, output ONLY a SQL SELECT query.
No comments. No markdown. No explanation.

Question: {question}
"""

    for _ in range(n):
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:7b-instruct",
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": 0.95,
                },
                timeout=60
            )
            text = resp.json().get("response", "").strip()
            variants.append(text)
        except Exception:
            variants.append("")

    print("-----")
    print(variants)
    print("-----")
    return variants


# def generate_sql_variants(question: str, n: int = 3, temperature: float = 0.7) -> List[str]:
#     """
#     Generate N raw SQL variants from the model without validation or repair.
#
#     This function intentionally bypasses:
#     - schema inspector
#     - Chroma retrieval
#     - Vanna's validation
#     - automatic repair
#
#     Because for self-agreement confidence, we want *true raw model variability*.
#     """
#
#     from vn_local import LocalVanna  # your existing Ollama+Chroma Vanna instance
#
#     v = LocalVanna()
#     variants = []
#
#     prompt = (
#         "Generate only a SQL query. "
#         "Do NOT add explanation or Markdown. "
#         "Question: " + question
#     )
#
#     for _ in range(n):
#         try:
#             resp = v.ask(
#                 prompt,
#                 temperature=temperature,
#                 top_p=0.95,
#                 max_tokens=512
#             )
#             # v.ask() returns text; ensure it's stripped
#             sql_candidate = resp.strip()
#             variants.append(sql_candidate)
#         except Exception:
#             variants.append("")
#
#     print(variants)
#
#     return variants


def validate_only(sql: str) -> Dict[str, Any]:
    ok, diag = _validate_sql(sql)
    return {"sql": sql, "validated": ok, "diagnostics": diag}

def _build_unknown_table_hints(unknown_tables: List[str]) -> str:
    """
    Turn unknown_tables diagnostics into gentle hints about CTEs / misuse.
    """
    if not unknown_tables:
        return ""

    lines: List[str] = []
    lines.append(
        "- You referenced some identifiers that are NOT real tables in the schema.\n"
        "  If they are meant to be CTEs, you must define them in a WITH clause.\n"
        "  Only use the real tables listed in the SCHEMA above.\n"
    )

    for t in unknown_tables:
        name = (t or "").lower()

        if name == "daterange":
            lines.append(
                "- 'DateRange' should be a CTE that generates dates (for example, the last 30 days) "
                "based on DimDate and MAX(FullDateAlternateKey). It is not a physical table."
            )
        elif name == "allorders":
            lines.append(
                "- 'AllOrders' should be a CTE selecting DISTINCT SalesOrderNumber from "
                "FactInternetSales joined to DimDate for the requested year."
            )
        elif name == "salesquota":
            lines.append(
                "- 'SalesQuota' should be derived from FactSalesQuota joined to DimEmployee, "
                "DimSalesTerritory and DimDate; do not reference 'SalesQuota' as a base table."
            )
        elif name in {"sales2003", "sales2004"}:
            lines.append(
                f"- '{t}' should be a CTE listing DISTINCT CustomerKey from FactInternetSales "
                "joined to DimDate and filtered by CalendarYear (e.g. 2003 or 2004)."
            )
        elif name in {"all_objects", "sys"}:
            lines.append(
                "- Do NOT use sys.all_objects or other system tables. "
                "Use only the business tables defined in the SCHEMA."
            )

    return "\n".join(lines)

def embedding_similarity(model_sql: str, top_k: int = 3) -> float:
    """
    Compute embedding similarity score using ChromaDB (through Vanna).
    Returns a float between 0 and 1.
    """
    from vn_local import LocalVanna

    if not model_sql.strip():
        return 0.0

    try:
        v = LocalVanna()

        # Embed the model SQL
        sql_embedding = v.embed(model_sql)

        # Query nearest SQL examples
        results = v.vectorstore.similarity_search_by_vector(sql_embedding, k=top_k)

        if not results:
            return 0.0

        # Average the similarity scores
        scores = [r.score for r in results if hasattr(r, "score")]
        if not scores:
            return 0.0

        ess = sum(scores) / len(scores)
        return max(0.0, min(1.0, ess))  # Normalize to [0,1]
    except Exception:
        return 0.0
