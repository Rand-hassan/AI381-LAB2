
"""Churn prediction service with MLflow model lineage."""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src import schema

MODEL_DIR = Path("artifacts")
model = joblib.load(MODEL_DIR / "model.joblib")
meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))

# Use the Git commit SHA provided by Docker.
meta["model_version"] = os.getenv("GIT_SHA", "local")

app = FastAPI(title="AI381 churn API", version=meta["model_version"])


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(gt=0, le=1000)
    contract: str
    internet_service: str
    support_calls: int = Field(ge=0, le=50)


@app.get("/health")
def health():
    """Show which model is running and its MLflow lineage."""
    return {"status": "ok", **meta}


@app.post("/predict")
def predict(customer: Customer):
    X = pd.DataFrame([customer.model_dump()])[schema.FEATURES]
    probability = float(model.predict_proba(X)[0, 1])

    return {
        "churn_probability": round(probability, 3),
        "model_version": meta["model_version"],
    }
