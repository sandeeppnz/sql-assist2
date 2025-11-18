from typing import Optional
from vn_local import LocalVanna

_vn_instance: Optional[LocalVanna] = None

def get_vn() -> LocalVanna:
    global _vn_instance
    if _vn_instance is None:
        _vn_instance = LocalVanna()
    return _vn_instance

def generate_sql_from_question(question: str) -> str:
    vn = get_vn()
    sql = vn.generate_sql(question)
    return " ".join(str(sql).split())

def generate_sql_from_prompt(prompt: str) -> str:
    vn = get_vn()
    sql = vn.generate_sql(prompt)
    return " ".join(str(sql).split())
