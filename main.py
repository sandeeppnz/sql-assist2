from fastapi import FastAPI
from pydantic import BaseModel

from calibration import calibrated_confidence
from config import ENABLE_ESS, ENABLE_SELF_AGREEMENT, ESS_TOP_K
from sql_service import embedding_similarity, generate_full_pipeline, generate_raw, generate_sql_variants, validate_only

app = FastAPI(title="Text-to-SQL Service (Ollama + Vanna)", version="0.1.0")

class QuestionRequest(BaseModel):
    question: str

class SqlRequest(BaseModel):
    sql: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/sql/generate")
def generate_endpoint(req: QuestionRequest):
    """Full pipeline: generate + validate + auto-repair."""
    return generate_full_pipeline(req.question)

@app.post("/sql/generate_raw")
def generate_raw_endpoint(req: QuestionRequest):
    """Debug: just raw generation (no validation/repair)."""
    return generate_raw(req.question)

@app.post("/sql/validate")
def validate_endpoint(req: SqlRequest):
    """Debug: validate arbitrary SQL without changing it."""
    return validate_only(req.sql)

