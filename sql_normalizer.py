import sqlparse


def canonicalize_sql(sql: str) -> str:
    """
    Normalizes SQL to a canonical form:
    - uppercase keywords
    - remove comments
    - collapse whitespace
    - remove AS aliases
    """

    if not sql:
        return ""

    # Basic formatting (consistent keyword casing)
    formatted = sqlparse.format(
        sql,
        keyword_case="upper",
        strip_comments=True,
        identifier_case=None,
    )

    # Remove aliases such as "table AS t" or "table t"
    formatted = remove_aliases(formatted)

    # Collapse repeated whitespace
    formatted = " ".join(formatted.split())

    return formatted


def remove_aliases(sql: str) -> str:
    """
    Removes table aliases but keeps table names.
    Example:
        "FactInternetSales AS fis" → "FactInternetSales"
    """
    tokens = sql.split()
    result_tokens = []

    skip_next = False
    for i, t in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue

        # Remove "AS alias"
        if t.upper() == "AS":
            skip_next = True
            continue

        # Remove "<table> <alias>"
        if i > 0:
            prev = tokens[i - 1].upper()
            if prev not in {"FROM", "JOIN"}:
                result_tokens.append(t)
            else:
                # keep table name only
                result_tokens.append(t)
        else:
            result_tokens.append(t)

    return " ".join(result_tokens)
