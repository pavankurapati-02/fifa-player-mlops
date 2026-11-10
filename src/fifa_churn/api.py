from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fifa_churn.modeling import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    TARGET_DEFINITION,
    predict_probabilities,
)

app = FastAPI(
    title="FIFA Player Risk Proxy API",
    description=TARGET_DEFINITION,
    version="1.0.0",
)


class PlayerFeatures(BaseModel):
    age: int = Field(ge=0, le=100)
    overall: int = Field(ge=0, le=100)
    potential: int = Field(ge=0, le=100)
    height_cm: float | None = None
    weight_kg: float | None = None
    weak_foot: int | None = Field(default=None, ge=1, le=5)
    skill_moves: int | None = Field(default=None, ge=1, le=5)
    international_reputation: int | None = Field(default=None, ge=1, le=5)
    player_positions: str | None = None
    preferred_foot: str | None = None


class BatchPredictionRequest(BaseModel):
    players: list[PlayerFeatures] = Field(min_length=1, max_length=500)


@lru_cache(maxsize=1)
def _load_model() -> Any:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run pipelines/run_all_labs.py first."
        )
    return joblib.load(MODEL_PATH)


def _predict(players: list[PlayerFeatures]) -> list[dict[str, Any]]:
    try:
        model = _load_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    features = pd.DataFrame(
        [player.model_dump() for player in players],
        columns=FEATURE_COLUMNS,
    )
    probabilities = predict_probabilities(model, features)
    return [
        {
            "churn_proxy": int(probability >= 0.5),
            "risk_probability": round(probability, 6),
            "warning": "synthetic age-based proxy only; not observed player churn.",
        }
        for probability in probabilities
    ]


@app.get("/health")
def health() -> dict[str, str]:
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Trained model is unavailable.")
    return {"status": "ok"}


@app.post("/predict")
def predict(player: PlayerFeatures) -> dict[str, Any]:
    return _predict([player])[0]


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest) -> dict[str, Any]:
    return {"predictions": _predict(request.players)}
