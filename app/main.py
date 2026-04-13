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
        predicted_direction, confidence, probabilities = predictor.predict(payload.rawScores)
        return PredictResponse(
            predictedDirection=predicted_direction,
            confidence=confidence,
            probabilities=probabilities,
            modelVersion="v1-logreg"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")