
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

DELIVERY_MODEL_PATH = BASE_DIR / "delivery_model.joblib"
INSTALLATION_MODEL_PATH = BASE_DIR / "installation_model.joblib"

FEATURES = [
    "inventoryAvailableAtCreation",
    "deliveryZone",
    "distanceKm",
    "carrierId",
    "dispatchStatusAtCutoff",
    "hoursToPromisedAtCutoff",
    "installationRequired",
    "installationSlotAvailableAtCutoff",
    "installationSchedulingAtCutoff",
]

NUMERIC_FEATURES = {
    "distanceKm",
    "hoursToPromisedAtCutoff",
}

BOOLEAN_FEATURES = {
    "inventoryAvailableAtCreation",
    "installationRequired",
    "installationSlotAvailableAtCutoff",
}

CATEGORICAL_FEATURES = {
    "deliveryZone",
    "carrierId",
    "dispatchStatusAtCutoff",
    "installationSchedulingAtCutoff",
}

app = FastAPI(
    title="SYNOVA AI Prediction API",
    version="1.0.0",
    description=(
        "Prototype delivery and installation delay models. "
        "Models were trained on synthetic data."
    ),
)

delivery_model = None
installation_model = None


@app.on_event("startup")
def load_models():
    global delivery_model, installation_model

    if not DELIVERY_MODEL_PATH.exists():
        raise RuntimeError("delivery_model.joblib is missing")

    if not INSTALLATION_MODEL_PATH.exists():
        raise RuntimeError("installation_model.joblib is missing")

    # Load only trusted model artifacts created by this project.
    delivery_model = joblib.load(DELIVERY_MODEL_PATH)
    installation_model = joblib.load(INSTALLATION_MODEL_PATH)


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Prediction-time order features",
    )


def prepare_features(payload: dict[str, Any]) -> pd.DataFrame:
    unknown = set(payload) - set(FEATURES)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown feature fields: {sorted(unknown)}",
        )

    row = {}

    for name in FEATURES:
        value = payload.get(name)

        if value is not None and name in NUMERIC_FEATURES:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(
                    status_code=422,
                    detail=f"{name} must be a number or null",
                )

        if value is not None and name in BOOLEAN_FEATURES:
            if not isinstance(value, bool):
                raise HTTPException(
                    status_code=422,
                    detail=f"{name} must be true, false, or null",
                )

        if value is not None and name in CATEGORICAL_FEATURES:
            if not isinstance(value, str):
                raise HTTPException(
                    status_code=422,
                    detail=f"{name} must be a string or null",
                )

        if name == "distanceKm" and value is not None and value < 0:
            raise HTTPException(
                status_code=422,
                detail="distanceKm cannot be negative",
            )

        row[f"features.{name}"] = value

    return pd.DataFrame([row], columns=[f"features.{name}" for name in FEATURES])


def predict(model, payload: dict[str, Any]) -> dict[str, Any]:
    X = prepare_features(payload)
    prediction = str(model.predict(X)[0])

    # This is an uncalibrated model score, not a verified real-world probability.
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)
    class_scores = {
        str(label): round(float(score), 4)
        for label, score in zip(classes, probabilities)
    }

    return {
        "predicted_label": prediction,
        "model_scores": class_scores,
        "score_note": (
            "Scores are model outputs from synthetic training data; "
            "they are not validated real-world probabilities."
        ),
        "synthetic_data_only": True,
    }


@app.get("/")
def root():
    return {
        "service": "SYNOVA AI Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "delivery_model_loaded": delivery_model is not None,
        "installation_model_loaded": installation_model is not None,
    }


@app.post("/predict/delivery")
def predict_delivery(request: PredictionRequest):
    if delivery_model is None:
        raise HTTPException(status_code=503, detail="Delivery model unavailable")

    return predict(delivery_model, request.features)


@app.post("/predict/installation")
def predict_installation(request: PredictionRequest):
    if installation_model is None:
        raise HTTPException(
            status_code=503,
            detail="Installation model unavailable",
        )

    if request.features.get("installationRequired") is not True:
        raise HTTPException(
            status_code=422,
            detail="Installation prediction requires installationRequired=true",
        )

    return predict(installation_model, request.features)
