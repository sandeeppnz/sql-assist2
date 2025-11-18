from typing import List, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Check your .env file.")

engine = create_engine(DATABASE_URL, fast_executemany=True)
SessionLocal = sessionmaker(bind=engine)

def run_query(sql: str) -> List[Tuple]:
    """Execute raw SQL and return list of tuples."""
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = result.fetchall()
        return [tuple(r) for r in rows]
