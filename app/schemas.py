from pydantic import BaseModel
from typing import Dict, List, Optional


class PredictRequest(BaseModel):
    rawScores: Dict[str, float]
    topDirectionFromRules: Optional[str] = None


class TopPrediction(BaseModel):
    direction: str
    confidence: float
    matchPercent: int


class PredictResponse(BaseModel):
    predictedDirection: str
    confidence: float = 0.0
    matchPercent: int = 0
    probabilities: Dict[str, float] = {}
    topPredictions: List[TopPrediction] = []
    modelVersion: str = "unknown"