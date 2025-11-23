import json
from config import GOLD_TRAIN_FILE
from vanna_provider import get_vn
from schema_service import schema_service
from sql_normalizer import canonicalize_sql


def main():
    vn = get_vn()

    print("Training on schema text...")
    vn.train(documentation=schema_service.schema_text)

    print("Training on gold Q/A examples...")
    with open(GOLD_TRAIN_FILE, "r", encoding="utf-8") as f:
        gold_items = json.load(f)

    for i, item in enumerate(gold_items, start=1):
        q = item["question"]
        sql = item["gold_sql"]

        # --- ORIGINAL SQL ---
        vn.train(question=q, sql=sql)

        # --- CANONICAL VERSION ---
        canonical = canonicalize_sql(sql)
        if canonical and canonical != sql:
            vn.train(question=q, sql=canonical)

        # --- LOWERCASE CANONICAL VERSION (helps embeddings a lot) ---
        lower = canonical.lower()
        vn.train(question=q, sql=lower)

        print(f" Trained {i}: {q[:60]}...")

    print("Training complete.")


if __name__ == "__main__":
    main()
