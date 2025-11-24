import json
import glob
import os

from tqdm import tqdm

from vanna_provider import get_vn
from schema_service import schema_service
from sql_normalizer import canonicalize_sql


TRAIN_DIR = "data/train"


def load_training_files():
    """Returns a list of all JSON training files inside data/train/."""
    pattern = os.path.join(TRAIN_DIR, "*.json")
    return glob.glob(pattern)


def load_items_from_file(path: str):
    """Reads and returns JSON array from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_item(vn, item, i, file_name):
    """Trains Vanna on a single Q/A item."""
    q = item["question"]
    sql = item["gold_sql"]

    # --- ORIGINAL SQL ---
    vn.train(question=q, sql=sql)

    # --- CANONICAL NORMALIZED SQL ---
    canonical = canonicalize_sql(sql)
    if canonical and canonical != sql:
        vn.train(question=q, sql=canonical)

    # --- LOWERCASE CANONICAL VERSION ---
    vn.train(question=q, sql=canonical.lower())

    tqdm.write(f"  Trained {i}: {q[:60]}...  ({file_name})")


def main():
    vn = get_vn()

    tqdm.write("\nTraining on schema text...")
    vn.train(documentation=schema_service.schema_text)

    # ------------------------------------------------------------
    # Load all training JSONs from data/train/
    # ------------------------------------------------------------
    files = load_training_files()

    if not files:
        tqdm.write("No training files found in data/train/*.json")
        return

    tqdm.write(f"\nFound {len(files)} training files:")
    for f in files:
        tqdm.write(" -", f)

    tqdm.write("\nTraining on gold Q/A examples...")


    for file_path in files:
        file_name = os.path.basename(file_path)

        tqdm.write(f"\nProcessing file: {file_name}")

        gold_items = load_items_from_file(file_path)

        for i, item in enumerate(gold_items, start=1):
            train_item(vn, item, i, file_name)

    tqdm.write("\nTraining complete.")


if __name__ == "__main__":
    main()
