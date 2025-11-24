# calibrator_predict.py
import joblib
import numpy as np
from tqdm import tqdm

MODEL_FILE = "calibrator.pkl"

def predict_calibrated_confidence(components: dict) -> float:
    bundle = joblib.load(MODEL_FILE)
    scaler = bundle["scaler"]
    calibrator = bundle["calibrator"]
    keys = bundle["feature_keys"]

    X = np.array([[components[k] for k in keys]])
    X_scaled = scaler.transform(X)

    # Calibrated probability
    prob = calibrator.predict_proba(X_scaled)[0, 1]
    return float(prob)

# Example usage
if __name__ == "__main__":
    example = {
        "schema_validity": 1.0,
        "structure": 0.60,
        "self_agreement": 0.456,
        "execution": 0.2,
        "embedding_similarity": 0.667,
    }

    tqdm.write("Calibrated:", predict_calibrated_confidence(example))
