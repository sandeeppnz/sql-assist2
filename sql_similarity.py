import sqlparse

def structural_similarity(sql1: str, sql2: str) -> float:
    """
    Computes structure similarity based on:
    - FROM tables
    - JOIN tables
    - WHERE columns
    - GROUP BY keys
    - Aggregates (SUM, COUNT)
    """

    parsed1 = sqlparse.parse(sql1)
    parsed2 = sqlparse.parse(sql2)

    if not parsed1 or not parsed2:
        return 0.0

    s1 = extract_structure(parsed1[0])
    s2 = extract_structure(parsed2[0])

    # Jaccard similarity
    return jaccard_similarity(s1, s2)


def extract_structure(stmt):
    text = stmt.value.upper()
    components = set()

    keywords = ["FROM", "JOIN", "WHERE", "GROUP BY", "ORDER BY", "SUM", "COUNT"]

    for kw in keywords:
        if kw in text:
            components.add(kw)

    return components


def jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
