# Local Text-to-SQL with Vanna + Ollama + AdventureWorksDW2022

## Setup

1. Create and edit `.env` (one is included as a template):

- `DATABASE_URL` pointing to your AdventureWorksDW2022 SQL Server
- `OLLAMA_HOST` (default `http://localhost:11434`)
- `OLLAMA_MODEL` (e.g. `qwen2.5:7b-instruct`)
- `STRICT_PREFLIGHT=true` to enable SQL Server compile-only checks

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure Ollama is running and the model is pulled:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

4. Train Vanna on schema + gold SQL:

```bash
python train_vn.py
```

5. Run the FastAPI app:

```bash
uvicorn main:app --reload
```

Endpoints:

- `GET /health` – simple health check
- `POST /sql/generate` – full pipeline (generate + validate + auto-repair)
- `POST /sql/generate_raw` – raw generation only
- `POST /sql/validate` – validate arbitrary SQL

6. Evaluate against the gold set:

```bash
python eval_gold.py
```

Results will be written to `adventureworks_eval_results.json`.
