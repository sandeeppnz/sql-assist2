# calibrator_train.py
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import joblib
from tqdm import tqdm

INPUT = "adventureworks_eval_results.json"
MODEL_FILE = "calibrator.pkl"

FEATURE_KEYS = [
    "schema_validity",
    "structure",
    "self_agreement",
    "execution",
    "embedding_similarity"
]

def sanitize(value):
    """Convert None or NaN to 0.0"""
    try:
        if value is None:
            return 0.0
        if isinstance(value, float) and np.isnan(value):
            return 0.0
        return float(value)
    except:
        return 0.0

def extract_features(item):
    comp = item.get("confidence_components", {}) or {}
    return [sanitize(comp.get(k, 0.0)) for k in FEATURE_KEYS]

def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    X = []
    y = []

    for row in data:
        # Skip gold-error cases (no training signal)
        if row.get("gold_error"):
            continue

        # label
        correct = row.get("model_correct")
        if correct is None:
            continue

        y.append(1 if correct else 0)

        # features
        X.append(extract_features(row))

    X = np.array(X)
    y = np.array(y)

    # ---- safety: remove rows with leftover NaN or inf ----
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- scaler + logistic regression ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=2000)
    calibrated = CalibratedClassifierCV(lr, cv="prefit")

    # Fit base LR, then calibrator
    lr.fit(X_scaled, y)
    calibrated.fit(X_scaled, y)

    # Save
    joblib.dump({
        "scaler": scaler,
        "model": lr,
        "calibrator": calibrated,
        "feature_keys": FEATURE_KEYS
    }, MODEL_FILE)

    tqdm.write(f"Saved → {MODEL_FILE}")
    tqdm.write(f"Training accuracy: {lr.score(X_scaled, y):.4f}")

if __name__ == "__main__":
    main()
