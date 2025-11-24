# schema_service.py
from sqlalchemy import inspect
from typing import Dict, List, Set
from db import engine

EXCLUDED_TABLES = {"DatabaseLog", "sysdiagrams"}


class SchemaService:
    def __init__(self, engine):
        self.engine = engine
        self.tables: Set[str] = set()
        self.cols_by_table: Dict[str, List[str]] = {}
        self.schema_text: str = ""
        self.raw_schema: str = ""
        self.compressed_schema: str = ""
        self._load_schema()

    def _load_schema(self) -> None:
        insp = inspect(self.engine)
        tables = [
            t for t in insp.get_table_names()
            if t not in EXCLUDED_TABLES
        ]

        lines = []
        for t in tables:
            cols = [c["name"] for c in insp.get_columns(t)]
            self.cols_by_table[t] = cols
            self.tables.add(t)
            lines.append(f"- {t}: {', '.join(cols)}")

        raw = "\n".join(lines)
        self.raw_schema = raw

        self.compressed_schema = raw
        self.schema_text = raw

schema_service = SchemaService(engine)
