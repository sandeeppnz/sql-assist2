# confidence_service.py
import os
import joblib
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

DEFAULT_CALIBRATOR_PATH = "calibrator.pkl"


# -----------------------------
# Confidence Result Object
# -----------------------------
@dataclass
class ConfidenceResult:
    raw: float
    calibrated: float
    components: Dict[str, float]
    used_calibrator: bool


# -----------------------------
# Confidence Service
# -----------------------------
class ConfidenceService:
    def __init__(self, calibrator_path: str = DEFAULT_CALIBRATOR_PATH):
        self.calibrator_path = calibrator_path
        self.calibrator = None
        self.feature_keys = None
        self.scaler = None

        if os.path.exists(calibrator_path):
            self._load_calibrator()

    def _load_calibrator(self):
        bundle = joblib.load(self.calibrator_path)

        self.scaler = bundle["scaler"]
        self.model = bundle["model"]
        self.calibrator = bundle["calibrator"]
        self.feature_keys = bundle["feature_keys"]

        print(f"[ConfidenceService] Loaded calibrator from {self.calibrator_path}")

    # ---------------------------------------
    # Predict calibrated probability
    # ---------------------------------------
    def _predict_calibrated(self, features: Dict[str, float]) -> Optional[float]:
        if self.calibrator is None:
            return None  # calibrator missing

        try:
            X = np.array([[features[k] for k in self.feature_keys]])
        except KeyError:
            # Missing a feature → fallback
            return None

        X_scaled = self.scaler.transform(X)
        prob = self.calibrator.predict_proba(X_scaled)[0, 1]
        return float(prob)

    # ---------------------------------------
    # Main entry: compute confidence
    # ---------------------------------------
    def compute_confidence(
        self,
        raw_confidence: float,
        components: Dict[str, float]
    ) -> ConfidenceResult:

        calibrated = self._predict_calibrated(components)

        return ConfidenceResult(
            raw=raw_confidence,
            calibrated=calibrated if calibrated is not None else raw_confidence,
            components=components,
            used_calibrator=calibrated is not None,
        )


# ----------------------------------------------
# Singleton accessor (optional convenience)
# ----------------------------------------------
_confidence_service_instance: Optional[ConfidenceService] = None

def get_confidence_service() -> ConfidenceService:
    global _confidence_service_instance

    if _confidence_service_instance is None:
        _confidence_service_instance = ConfidenceService()

    return _confidence_service_instance