from pydantic import BaseModel
from typing import Dict, Optional


class PredictRequest(BaseModel):
    rawScores: Dict[str, float]
    topDirectionFromRules: Optional[str] = None


class PredictResponse(BaseModel):
    predictedDirection: str
    confidence: float
    probabilities: Dict[str, float]
    modelVersion: str