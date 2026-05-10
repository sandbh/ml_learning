"""
FastAPI inference API for the heart-disease classifier (containerisation / Task 6).

**File:** ``src/api/api.py``

**Endpoints**

- ``POST /predict`` — JSON body matches ``PatientData``; returns prediction, label,
  confidence (probability of class 1), and risk band.
- ``GET /health`` — Liveness; confirms the sklearn pipeline loaded.
- ``GET /metrics`` — Prometheus text exposition (monitoring / Task 8).

**Usage (local dev, repo root)**

.. code-block:: bash

    source .venv/bin/activate
    python src/api/api.py
    # or: uvicorn api.api:app --app-dir src --host 0.0.0.0 --port 8000

**Usage (Docker)** — image CMD runs ``uvicorn api.api:app`` with ``PYTHONPATH=/app/src``;
ensure ``models/best_model.pkl`` and ``models/feature_names.pkl`` exist (train first or mount ``./models``).

**Environment**

- ``MODEL_PATH`` — joblib pipeline (default: ``<repo>/models/best_model.pkl``).
- ``FEATURE_PATH`` — pickled feature name list (default: ``<repo>/models/feature_names.pkl``).
- ``API_LOG_PATH`` — optional request log file (default: ``<repo>/api_requests.log``).
- ``PORT`` — bind port when running ``python src/api/api.py`` (default: ``8000``).

**Author.** SANDIP BHATTACHARYYA — BITS Pilani ID 2025cs05025
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

_SRC_ROOT = Path(__file__).resolve().parent.parent  # .../src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from config.paths import MODELS_DIR, PROJECT_ROOT

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "best_model.pkl")))
FEATURE_PATH = Path(os.getenv("FEATURE_PATH", str(MODELS_DIR / "feature_names.pkl")))

# ─────────────────────────────────────────────
# LOGGING (Task 8)
# ─────────────────────────────────────────────

_log_handlers: list[logging.Handler] = [logging.StreamHandler()]
_log_file = Path(os.getenv("API_LOG_PATH", str(PROJECT_ROOT / "api_requests.log")))
try:
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    _log_handlers.append(logging.FileHandler(_log_file))
except OSError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=_log_handlers,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PROMETHEUS METRICS (Task 8)
# ─────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Total number of /predict requests",
    ["status"],
)
REQUEST_LATENCY = Histogram(
    "predict_request_latency_seconds",
    "Latency of /predict requests in seconds",
)
PREDICTION_DIST = Counter(
    "prediction_label_total",
    "Total predictions by label",
    ["label"],
)

# Expose "last prediction" style gauges for dashboarding.
# Note: Prometheus is pull-based, so these represent the most recently observed
# values at scrape time (not a full event log).
LAST_PREDICTION = Gauge(
    "last_prediction",
    "Most recent prediction (0=Healthy, 1=Heart Disease)",
)
LAST_CONFIDENCE = Gauge(
    "last_prediction_confidence",
    "Most recent confidence score (probability of class 1)",
)
RISK_LEVEL_TOTAL = Counter(
    "prediction_risk_level_total",
    "Total predictions by risk level",
    ["risk_level"],
)
CONFIDENCE_HIST = Histogram(
    "prediction_confidence",
    "Distribution of confidence scores (probability of class 1)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS (before load_model: schema validation uses PatientData)
# ─────────────────────────────────────────────


class PatientData(BaseModel):
    age: float = Field(..., description="Age in years", examples=[63.0])
    sex: int = Field(..., description="1=male, 0=female", examples=[1])
    cp: int = Field(..., description="Chest pain type 0–3", examples=[3])
    trestbps: float = Field(
        ..., description="Resting blood pressure (mmHg)", examples=[145.0]
    )
    chol: float = Field(..., description="Serum cholesterol (mg/dl)", examples=[233.0])
    fbs: int = Field(
        ...,
        description="Fasting blood sugar >120: 1=true, 0=false",
        examples=[1],
    )
    restecg: int = Field(..., description="Resting ECG results 0–2", examples=[2])
    thalach: float = Field(
        ..., description="Maximum heart rate achieved", examples=[150.0]
    )
    exang: int = Field(
        ...,
        description="Exercise-induced angina: 1=yes, 0=no",
        examples=[0],
    )
    oldpeak: float = Field(
        ..., description="ST depression induced by exercise", examples=[2.3]
    )
    slope: int = Field(
        ...,
        description="Slope of peak exercise ST segment 0–2",
        examples=[2],
    )
    ca: float = Field(..., description="Number of major vessels (0–3)", examples=[0.0])
    thal: float = Field(
        ...,
        description="Thal: 0=normal, 1=fixed defect, 2=reversible",
        examples=[1.0],
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=Healthy, 1=Heart Disease")
    label: str = Field(..., description="Human-readable label")
    confidence: float = Field(
        ...,
        description="Probability of heart disease (class 1)",
    )
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH")


# ─────────────────────────────────────────────
# LOAD MODEL & APP
# ─────────────────────────────────────────────

model = None
feature_names: list[str] | None = None


def load_model() -> None:
    """Load sklearn pipeline and training feature order from disk."""
    global model, feature_names
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH.resolve()}")
    if not FEATURE_PATH.is_file():
        raise FileNotFoundError(f"Feature list not found at {FEATURE_PATH.resolve()}")

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("feature_names.pkl must be a non-empty list of column names.")

    expected = set(PatientData.model_fields.keys())
    missing = set(feature_names) - expected
    if missing:
        raise ValueError(
            "feature_names.pkl contains columns not exposed by PatientData: "
            f"{sorted(missing)}. Update PatientData or retrain."
        )

    logger.info("Model loaded from %s", MODEL_PATH.resolve())
    logger.info("Feature order (%d): %s", len(feature_names), feature_names)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment — Heart Disease Risk Classifier",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────


@app.get("/health", summary="Health check")
def health_check():
    """Return liveness JSON and whether the sklearn pipeline is loaded."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, summary="Predict heart disease risk")
def predict(patient: PatientData):
    """Run the saved pipeline on one patient row; update Prometheus metrics and logs."""
    if model is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        row = patient.model_dump()
        X = pd.DataFrame([row])[feature_names]

        prediction = int(model.predict(X)[0])
        confidence = float(model.predict_proba(X)[0][1])

        label = "Heart Disease" if prediction == 1 else "Healthy"
        if confidence < 0.35:
            risk = "LOW"
        elif confidence < 0.65:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        latency = time.perf_counter() - start

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(latency)
        PREDICTION_DIST.labels(label=label).inc()
        RISK_LEVEL_TOTAL.labels(risk_level=risk).inc()
        LAST_PREDICTION.set(prediction)
        LAST_CONFIDENCE.set(confidence)
        CONFIDENCE_HIST.observe(confidence)

        logger.info(
            "PREDICT | pred=%s conf=%.3f risk=%s latency=%.3fs | age=%s sex=%s cp=%s",
            prediction,
            confidence,
            risk,
            latency,
            patient.age,
            patient.sex,
            patient.cp,
        )

        return PredictionResponse(
            prediction=prediction,
            label=label,
            confidence=round(confidence, 4),
            risk_level=risk,
        )

    except HTTPException:
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(status="error").inc()
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics", summary="Prometheus metrics")
def metrics():
    """Expose Prometheus metrics (scrape target for Grafana)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", summary="API info")
def root():
    """Minimal discovery payload with links to docs and main routes."""
    return {
        "name": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "metrics": "/metrics",
    }


# ─────────────────────────────────────────────
# RUN (local dev without Docker)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
