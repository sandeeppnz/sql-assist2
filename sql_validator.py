import re
from typing import Tuple, List
from db import run_query
from schema_service import schema_service
from config import STRICT_PREFLIGHT

SQL_UPPER = (
    " INSERT ", " UPDATE ", " DELETE ", " ALTER ", " DROP ", " TRUNCATE ",
    " CREATE ", " MERGE ", " EXEC ", " EXECUTE ", " GRANT ", " REVOKE ",
    " BACKUP ", " RESTORE "
)

SQL_START_RE = re.compile(r"(?is)^\s*(with|select)\b")

# Disallow obviously non-T-SQL constructs we see from OSS models
BAD_TSQL_PATTERNS = (
    " limit ",
    " offset ",
    " period for ",
    " grouping sets",
)

TABLE_NAME_RE = re.compile(
    r'\bfrom\s+([\w\.\[\]]+)|\bjoin\s+([\w\.\[\]]+)',
    re.I
)

# First CTE after WITH
CTE_FIRST_RE = re.compile(
    r'\bwith\s+([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(',
    re.IGNORECASE,
)

# Subsequent CTEs after comma
CTE_COMMA_RE = re.compile(
    r',\s*([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(',
    re.IGNORECASE,
)

COLUMN_REF_RE = re.compile(
    r'([A-Za-z0-9_\[\]]+)\.([A-Za-z0-9_\[\]]+)',
    re.IGNORECASE,
)

YEAR_ORDERDATEKEY_RE = re.compile(r'\bYEAR\s*\(\s*OrderDateKey\s*\)', re.IGNORECASE)


def _extract_cte_names(sql: str) -> List[str]:
    """
    Extract all CTE names, including the first one after WITH and any
    additional ones separated by commas:

        WITH MaxDate AS (...),
             DateRange AS (...),
             SomethingElse AS (...)
        SELECT ...

    Returns a list of names with original casing.
    """
    if not sql:
        return []

    names: List[str] = []

    for m in CTE_FIRST_RE.finditer(sql):
        names.append(m.group(1))

    for m in CTE_COMMA_RE.finditer(sql):
        names.append(m.group(1))

    # de-dupe case-insensitively
    seen = set()
    deduped: List[str] = []
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(n)

    return deduped


def extract_tables(sql: str):
    raw = set()
    for m in TABLE_NAME_RE.finditer(sql):
        g = m.group(1) or m.group(2)
        if g:
            ident = g.split('.')[-1].strip('[]')
            raw.add(ident)
    return raw


def is_safe_select(sql: str) -> bool:
    lowered = sql.strip().lower()
    if not SQL_START_RE.search(lowered):
        return False
    padded = " " + lowered + " "
    if any(op.lower() in padded for op in SQL_UPPER):
        return False
    if any(bad in padded for bad in BAD_TSQL_PATTERNS):
        return False
    return True


def has_unknown_tables(sql: str) -> Tuple[bool, List[str]]:
    """
    Tables referenced in FROM/JOIN that are not in schema_service.tables
    and not defined as CTEs are considered unknown.

    This now correctly treats *all* CTEs (including those after commas)
    as known.
    """
    used = extract_tables(sql)
    cte_names = set(n.strip('[]') for n in _extract_cte_names(sql))
    unknown = [
        t for t in used
        if t not in schema_service.tables and t not in cte_names
    ]
    return bool(unknown), unknown


def has_unknown_columns(sql: str):
    """
    Very simple column check: only validates columns when the left-hand side
    of alias.column is an actual table name (not just a query alias).
    """
    unknown = []
    for alias, col in COLUMN_REF_RE.findall(sql):
        table_name = alias.strip('[]')
        if table_name not in schema_service.tables:
            continue
        valid_cols = schema_service.cols_by_table.get(table_name, [])
        col_name = col.strip('[]')
        if col_name not in valid_cols:
            unknown.append((table_name, alias, col_name))
    return bool(unknown), unknown


def _escape_for_tsql_literal(sql: str) -> str:
    return sql.replace("'", "''")


def _has_bad_year_usage(sql: str) -> Tuple[bool, str]:
    """
    Heuristic: detect YEAR(OrderDateKey), which is invalid because OrderDateKey is an int key.
    We want the model to join DimDate on OrderDateKey = DateKey and then filter on CalendarYear.
    """
    if YEAR_ORDERDATEKEY_RE.search(sql):
        return True, (
            "Do not use YEAR(OrderDateKey). OrderDateKey is an integer key; "
            "join to DimDate on OrderDateKey = DateKey and filter on DimDate.CalendarYear "
            "or DimDate.FullDateAlternateKey instead."
        )
    return False, ""


def server_preflight_ok(sql: str):
    if not STRICT_PREFLIGHT:
        return True, "ok"
    try:
        tsql = (
            "DECLARE @q nvarchar(max) = N'" +
            _escape_for_tsql_literal(sql) +
            "'; EXEC sp_describe_first_result_set @tsql = @q;"
        )
        _ = run_query(tsql)
        return True, "ok"
    except Exception as ex:
        return False, str(ex)
