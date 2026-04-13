import os
import joblib
import pandas as pd
from typing import Dict, Tuple

MODEL_PATH = os.path.join("models", "career_direction_model.pkl")

FEATURE_COLUMNS = [
    "direction.it",
    "direction.medicine",
    "direction.education",
    "direction.business",
    "direction.creative",
    "direction.engineering",
    "thinkingStyle.analytic",
    "thinkingStyle.creative",
    "thinkingStyle.practical",
    "temperament.introvert",
    "temperament.extrovert",
    "studyProfile.stem",
    "studyProfile.humanities",
    "values.stability",
    "values.income",
    "values.helping",
    "values.freedom",
    "anti.it",
    "anti.medicine",
    "anti.education",
    "anti.business",
    "anti.creative",
    "anti.engineering",
]

DEFAULT_LABEL = "education"


class CareerDirectionPredictor:
    def __init__(self):
        self.model = None
        self.labels = []

    def load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. Run train_model.py first."
            )
        payload = joblib.load(MODEL_PATH)
        self.model = payload["model"]
        self.labels = payload["labels"]

    def _vectorize(self, raw_scores: Dict[str, float]) -> pd.DataFrame:
        row = {}
        for feature in FEATURE_COLUMNS:
            row[feature] = float(raw_scores.get(feature, 0.0))
        return pd.DataFrame([row])

    def predict(self, raw_scores: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        if self.model is None:
            self.load()

        X = self._vectorize(raw_scores)
        prediction = self.model.predict(X)[0]

        probabilities = {}
        confidence = 0.0

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            probabilities = {
                label: round(float(prob), 4)
                for label, prob in zip(self.labels, proba)
            }
            confidence = probabilities.get(prediction, 0.0)

        return prediction, round(float(confidence), 4), probabilities