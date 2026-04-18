from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.predictor import CareerDirectionPredictor

app = FastAPI(title="BotaDiplom ML Service")

predictor = CareerDirectionPredictor()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        result = predictor.predict_full(payload.rawScores)

        return PredictResponse(
            predictedDirection=result["predictedDirection"],
            confidence=result["confidence"],
            matchPercent=result["matchPercent"],
            probabilities=result["probabilities"],
            topPredictions=result["topPredictions"],
            modelVersion="v2-ml"
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )