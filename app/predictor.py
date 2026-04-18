import os
import joblib
import pandas as pd
from typing import Dict, Tuple, List, Any

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
        self.labels: List[str] = []
        self.model_metrics: Dict[str, Any] = {}

    def load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. Run train_model.py first."
            )

        payload = joblib.load(MODEL_PATH)
        self.model = payload["model"]
        self.labels = payload.get("labels", [])
        self.model_metrics = payload.get("metrics", {})

    def _vectorize(self, raw_scores: Dict[str, float]) -> pd.DataFrame:
        row = {}
        for feature in FEATURE_COLUMNS:
            row[feature] = float(raw_scores.get(feature, 0.0))
        return pd.DataFrame([row])

    def predict(
        self, raw_scores: Dict[str, float]
    ) -> Tuple[str, float, int, Dict[str, float], List[Dict[str, float]]]:
        if self.model is None:
            self.load()

        X = self._vectorize(raw_scores)
        prediction = self.model.predict(X)[0]

        probabilities: Dict[str, float] = {}
        confidence = 0.0
        match_percent = 0
        top_predictions: List[Dict[str, float]] = []

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            probabilities = {
                label: round(float(prob), 4)
                for label, prob in zip(self.labels, proba)
            }

            confidence = float(probabilities.get(prediction, 0.0))
            match_percent = int(round(confidence * 100))

            sorted_predictions = sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            top_predictions = [
                {
                    "direction": label,
                    "confidence": round(float(prob), 4),
                    "matchPercent": int(round(float(prob) * 100)),
                }
                for label, prob in sorted_predictions[:3]
            ]
        else:
            prediction = prediction or DEFAULT_LABEL
            confidence = 0.0
            match_percent = 0
            probabilities = {prediction: 1.0}
            top_predictions = [
                {
                    "direction": prediction,
                    "confidence": 1.0,
                    "matchPercent": 100,
                }
            ]

        return (
            prediction,
            round(confidence, 4),
            match_percent,
            probabilities,
            top_predictions,
        )

    def predict_full(self, raw_scores: Dict[str, float]) -> Dict[str, Any]:
        prediction, confidence, match_percent, probabilities, top_predictions = self.predict(
            raw_scores
        )

        return {
            "predictedDirection": prediction,
            "confidence": confidence,
            "matchPercent": match_percent,
            "probabilities": probabilities,
            "topPredictions": top_predictions,
            "modelMetrics": self.model_metrics,
        }