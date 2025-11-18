from vanna_provider import generate_sql_from_question

def generate_sql(question: str) -> str:
    """Generate raw SQL from a natural language question using Vanna+Ollama."""
    return generate_sql_from_question(question)
