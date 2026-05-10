# Dockerfile — production-style image for the FastAPI inference API (assignment Task 6).
# Author: SANDIP BHATTACHARYYA — BITS Pilani ID 2025cs05025
#
# Expects trained artifacts under models/ at build time (best_model.pkl, feature_names.pkl).
# Build:  docker build -t heart-disease-api:latest .
# Run:    docker run --rm -p 8000:8000 heart-disease-api:latest
# Stack:  prefer docker-compose.yml for API + Prometheus + Grafana.
#
# ── Stage 1: base image ───────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# PYTHONPATH so ``api.api`` resolves to ``src/api/api.py``
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/best_model.pkl \
    FEATURE_PATH=/app/models/feature_names.pkl \
    API_LOG_PATH=/app/logs/api_requests.log \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/logs

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health >/dev/null || exit 1

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
