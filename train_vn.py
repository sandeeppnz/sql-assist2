import json
from config import GOLD_TRAIN_FILE
from vanna_provider import get_vn
from schema_service import schema_service

def main():
    vn = get_vn()
    print("Training on schema text...")
    vn.train(documentation=schema_service.schema_text)

    print("Training on gold Q/A examples...")
    with open(GOLD_TRAIN_FILE, "r", encoding="utf-8") as f:
        gold_items = json.load(f)

    for i, item in enumerate(gold_items, start=1):
        vn.train(question=item["question"], sql=item["gold_sql"])
        print(f" Trained {i}: {item['question'][:60]}...")

    print("Training complete.")

if __name__ == "__main__":
    main()
