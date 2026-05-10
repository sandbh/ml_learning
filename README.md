# Heart Disease Prediction — MLOps End-to-End Pipeline

**Author:** SANDIP BHATTACHARYYA — BITS Pilani ID `2025cs05025`  
**Course:** MLOps (S2-25_AMLCSZG523)

An end-to-end ML pipeline for predicting heart disease risk using the UCI Heart Disease dataset: preprocessing, EDA, training with MLflow, batch inference, FastAPI, Docker, Prometheus/Grafana monitoring, GitHub Actions CI/CD, and optional Kubernetes (`k8s/`).

**Jump to:** [Prerequisites](#prerequisites) · [Architecture](#architecture) · [Project structure](#project-structure) · [YAML files (purpose)](#yaml-files-purpose) · [Data (`data/`)](#data) · [Quick start](#quick-start-end-to-end) · [Kubernetes](#kubernetes-deployment-minikube) · [Clean stopping](#clean-stopping) · [Model details](#model-details) · [API](#api-endpoints) · [CI/CD](#cicd-pipeline-github-actions) · [Monitoring](#monitoring-prometheus-and-grafana) · [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python** 3.9+ (CI uses 3.11; tested locally on 3.9–3.12)
- **Docker** + **Docker Compose** — use **`docker compose`** (V2 plugin) or **`docker-compose`** (standalone CLI); this repo’s examples often use `docker-compose`
- Free ports for the monitoring demo: **8000** (API — local Python, Docker, Compose), **8080** (API — Kubernetes [port-forward](#kubernetes-deployment-minikube), step **7**, maps `8080:80`), **9090** (Prometheus), **3000** (Grafana)
- **Git**
- **kubectl** + a cluster (optional — for `k8s/`)
- **minikube** (optional — for local Kubernetes, same idea as course reference repos)
- macOS / Linux (Windows WSL2 should work)

---

## Architecture

High-level data and serving flow:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Data (CSV) │────▶│  EDA + clean │────▶│  Train + MLflow │────▶│  Model (.pkl)│
└─────────────┘     └──────────────┘     └─────────────────┘     └──────┬───────┘
                                                                        │
                    ┌──────────────┐      ┌─────────────────┐           │
                    │  Prometheus  │◀──── │ FastAPI + Docker│ ◀─────────┘
                    │  /metrics    │      │  /predict       │
                    └──────────────┘      └────────┬────────┘
                                                   │
                                                   ▼
                                         ┌──────────────────┐
                                         │ Kubernetes (opt) │
                                         │ k8s/ manifests   │
                                         └──────────────────┘
```

**Docker Compose** adds **Grafana** (port **3000**); Prometheus scrapes the API using `monitoring/prometheus.yml`.

---

## Project structure

```
ml_learning/                       # repository root (clone folder name may differ)
├── .github/workflows/ci.yml       # CI/CD Pipeline (4 jobs: lint → testing → training-model → docker-build-smoke)
├── data/
│   ├── heart_disease_UCI_dataset.csv      # Bundled raw UCI-style input
│   ├── heart_disease_processed_dataset.csv  # Clean numeric table for training (EDA / preprocess)
│   └── batch_predictions.csv      # Optional output from batch inference
├── models/                        # best_model.pkl, feature_names.pkl, training_metadata.json
├── mlruns/                        # MLflow tracking store
├── screenshots/                   # EDA and training plots
├── src/
│   ├── api/api.py                 # FastAPI: /health, /predict, /metrics
│   ├── config/paths.py
│   ├── data_preprocessing/pre_processing_data.py  # load_data, clean_data, save_cleaned_csv
│   ├── eda/eda.py                 # CLI EDA → clean CSV + plots
│   └── model_training/            # train.py, inference.py (batch scoring)
├── monitoring/
│   ├── prometheus.yml             # Prometheus scrape config for the API metrics endpoint
│   └── grafana/
│       ├── heart_disease_api.json # Grafana dashboard (panels for API metrics)
│       └── provisioning/
│           ├── dashboards/dashboards.yml   # Grafana: load dashboards from files
│           └── datasources/datasource.yml # Grafana: Prometheus connection
├── k8s/
│   ├── kustomization.yaml         # kubectl apply -k k8s/ — API + Prometheus + Grafana
│   ├── namespace.yaml
│   ├── deployment.yaml            # API
│   ├── service.yaml
│   ├── prometheus-*.yaml          # ConfigMap + Deployment + Service
│   ├── grafana-*.yaml             # ConfigMaps + Deployment + Service
│   ├── hpa.yaml                   # Optional
│   └── ingress.yaml               # Optional
├── scripts/test_full_flow.sh      # Optional end-to-end: lint, tests, pipeline, Compose
├── docker-compose.yml             # Local stack: API + Prometheus + Grafana
├── Dockerfile
├── requirements.txt
├── pytest.ini
└── README.md
```

**Note:** There is **no** `notebooks/` directory in this repo; EDA is **script-driven** via `src/eda/eda.py`. You can still open a Jupyter kernel and import the same modules if your coursework allows notebooks alongside scripts.

The **Data** heading later in this file explains raw vs cleaned CSV files and how they relate to `RAW_DATA_CSV` and `CLEAN_DATA_CSV`.

---

## YAML files (purpose)

Paths are relative to the repository root. Each file’s job is separate: Compose and monitoring configs support the local Docker stack; `k8s/` describes Kubernetes objects; the workflow file is only for GitHub’s runners.

- **`docker-compose.yml`** — Declares the local three-service setup (build and run the API container, Prometheus, Grafana), including mounts for trained models, API log directory, Prometheus config, Grafana provisioning, and the dashboard JSON so Grafana and Prometheus start with the right settings.

- **`monitoring/prometheus.yml`** — Configures how Prometheus scrapes the FastAPI `/metrics` endpoint (interval, job name, and target on the Compose network).

- **`monitoring/grafana/provisioning/datasources/datasource.yml`** — Registers Prometheus as a Grafana data source so dashboards can run queries against metric series.

- **`monitoring/grafana/provisioning/dashboards/dashboards.yml`** — Enables Grafana’s “load dashboards from disk” behavior for the dashboards directory.

- **`monitoring/grafana/heart_disease_api.json`** — Defines the dashboard layout and queries (not YAML; shipped beside the Grafana provisioning files).

- **`k8s/namespace.yaml`** — Creates the `heart-disease` namespace so Deployments, Services, and related objects are grouped together.

- **`k8s/deployment.yaml`** — Describes the API workload: container image, replica count, environment for model paths, and health checks using `/health`.

- **`k8s/service.yaml`** — Provides stable in-cluster networking to the API pods and maps external-facing port **80** to port **8000** in the container.

- **`k8s/hpa.yaml`** — Expresses autoscaling policy (CPU and memory targets) for the API Deployment when the cluster exposes those metrics.

- **`k8s/ingress.yaml`** — Declares HTTP routing from a hostname to the Service for environments that run an ingress controller (host and annotations are meant to be edited for your cluster).

- **`k8s/prometheus-configmap.yaml`**, **`prometheus-deployment.yaml`**, **`prometheus-service.yaml`** — In-cluster Prometheus that scrapes the API Service at **`heart-disease-api-service:80/metrics`**.

- **`k8s/grafana-datasource-configmap.yaml`**, **`grafana-dashboard-provider-configmap.yaml`**, **`grafana-dashboard-json-configmap.yaml`**, **`grafana-deployment.yaml`**, **`grafana-service.yaml`** — Grafana with the same datasource + dashboard JSON as Compose (admin **`admin`** / **`admin123`**).

- **`k8s/kustomization.yaml`** — Lists manifests so you can run **`kubectl apply -k k8s/`** for the **full** namespace stack (API + Prometheus + Grafana).

- **`.github/workflows/ci.yml`** — Defines the continuous integration pipeline: lint, tests, preprocessing and training, artifact upload, Docker image build, and container smoke checks on GitHub Actions.

The files under `monitoring/` are wired into **`docker-compose.yml`**. For Kubernetes, scrape targets and Grafana provisioning are duplicated into **`k8s/prometheus-*.yaml`** and **`k8s/grafana-*.yaml`** (dashboard JSON is copied into **`grafana-dashboard-json-configmap.yaml`**). If you edit **`monitoring/grafana/heart_disease_api.json`**, regenerate that ConfigMap with:  
`kubectl create configmap grafana-dashboard-heart-api -n heart-disease --from-file=heart_disease_api.json=monitoring/grafana/heart_disease_api.json -o yaml --dry-run=client > k8s/grafana-dashboard-json-configmap.yaml`

---

## Quick start (end-to-end)

All commands assume the **repository root** (directory containing this `README.md`).

### Recommended order (what to run, in sequence)

| Order | Step | Needs from earlier steps |
|------:|------|---------------------------|
| 1 | [Clone and setup](#1-clone-and-setup) | — |
| 2 | [EDA](#2-explore-the-data-eda) | 1 |
| 3 | [Train](#3-train-the-model) | 1, 2 (cleaned CSV) |
| 4 | [MLflow UI](#4-mlflow-ui-optional) (optional) | 1, 3 |
| 5 | [Unit tests](#5-unit-tests) | 1 only (can run right after 1, or after 3) |
| 6 | [Batch inference](#6-batch-inference-local) (optional) | 1, 3 (`models/`) |
| 7 | [API locally (Python)](#7-start-the-api-locally) | 1, 3 (`models/`) |
| 8 | [Docker (one container)](#8-docker-api-image-only) | 1, 3 + **Docker daemon**; **stop step 7 first** so port **8000** is free |
| 9 | [Docker Compose](#9-monitoring-stack-api--prometheus--grafana) (API + Prometheus + Grafana) | 1, 3 + Docker; **stop step 7 or 8** if they still hold **8000** |
| 10 | [One-shot script](#10-one-shot-script-optional) (optional) | Default: needs Docker for Compose. Use `SKIP_COMPOSE=1` or `--skip-compose` for steps 1–5 only |

**Rule:** Steps **2** then **3** are required before **6, 7, 8, or 9**. Step **7** (uvicorn in your venv) and steps **8–9** (containers) all want port **8000** by default—only run **one** of them at a time unless you change ports.

### 1. Clone and setup

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_CLONE_DIRECTORY>   # often ml_learning

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 2. Explore the data (EDA)

**Script / CLI (primary in this repo):**

```bash
python3 src/eda/eda.py all
```

This loads `data/heart_disease_UCI_dataset.csv`, writes `data/heart_disease_processed_dataset.csv`, and saves plots under `screenshots/`.

Individual steps: `load`, `inspect`, `preprocess`, `eda`, `all` — see `python3 src/eda/eda.py --help`.

### 3. Train the model

```bash
python3 src/model_training/train.py
```

This trains **Logistic Regression** and **Random Forest** with **GridSearchCV** (ROC-AUC), logs runs to **MLflow** under `mlruns/`, and saves the best pipeline to `models/best_model.pkl` plus `models/feature_names.pkl` and `models/training_metadata.json`.

### 4. MLflow UI (optional)

```bash
python3 -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 -p 5050
```

Open **http://127.0.0.1:5050** (use another port if busy). Run any time after **step 3**.

### 5. Unit tests

```bash
python3 -m pytest tests/ -v
```

Covers preprocessing, model pipelines, and API **`PatientData`** validation (see `tests/test_pipeline.py`). Only **step 1** is required; run early for a fast check, or after **step 3** once training artifacts exist.

**JUnit XML (matches CI artifact):**

```bash
python3 -m pytest tests/ -v --tb=short --junitxml=pytest-results.xml
```

### 6. Batch inference (local)

```bash
python3 src/model_training/inference.py --output data/batch_predictions.csv
```

Uses the same **raw → preprocess** path as training (`load_data` + `clean_data`), then the saved sklearn **Pipeline**. Requires **steps 1–3** (trained `models/`).

### 7. Start the API locally

Use this to verify the API **without** Docker. **Stop the server** (Ctrl+C in that terminal) before **step 8** or **9**, which also bind to port **8000**.

```bash
python3 src/api/api.py
# or:
python3 -m uvicorn api.api:app --app-dir src --host 0.0.0.0 --port 8000
```

- **Swagger:** http://127.0.0.1:8000/docs  
- **Health:** http://127.0.0.1:8000/health  

### 8. Docker (API image only)

Requires **steps 1–3** so `models/best_model.pkl` and `models/feature_names.pkl` exist at **image build** time. Do **not** run this at the same time as **step 7** on the same port.

```bash
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
```

Smoke checks:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":1}'
```

### 9. Monitoring stack (API + Prometheus + Grafana)

From repo root after **`models/`** exists (Compose mounts `./models` read-only). Stop anything else using **8000** (e.g. **step 7** or **step 8**) before starting.

```bash
docker-compose build
docker-compose up -d --build
# same with Compose V2:  docker compose build && docker compose up -d --build
```

- **API docs:** http://127.0.0.1:8000/docs  
- **Prometheus:** http://127.0.0.1:9090  
- **Grafana:** http://127.0.0.1:3000 — login **admin** / **admin123**  

Send a few `POST /predict` requests, then check targets in Prometheus and the provisioned Grafana dashboard JSON at `monitoring/grafana/heart_disease_api.json`.

Stop:

```bash
docker-compose down
# or:  docker compose down
```

More shutdown cases (K8s, port-forwards, volumes): **[Clean stopping](#clean-stopping)**.

**Compose services (fixed container names):** `heart-disease-api`, `prometheus`, `grafana`.

### 10. One-shot script (optional)

```bash
bash scripts/test_full_flow.sh
```

Runs **the same lint scope as CI** (`flake8 src tests`), tests (writes `pytest-results.xml`), EDA, training, batch inference, then brings up Compose. **On success the stack is left running** on ports 8000 / 9090 / 3000; stop with `docker compose down` or `docker-compose down`. **Docker must be running** for step **[6/6]** (e.g. Docker Desktop, or **Colima** with `colima start` — confirm with `docker info`).

If Docker is not available, run **steps 1–5 only** (no Compose):

```bash
SKIP_COMPOSE=1 bash scripts/test_full_flow.sh
# or:
bash scripts/test_full_flow.sh --skip-compose
```

---

## Kubernetes deployment (Minikube)

**Prerequisites:** A working **Docker daemon** (Docker Desktop or **Colima**), **minikube**, **kubectl**. Minikube’s **docker** driver talks to that same daemon, so Docker can stay running.

**After Docker Compose:** Stop the Compose stack so nothing is still bound to **8000** on the host and you are not hitting the wrong API by mistake:

```bash
docker compose down
# or:  docker-compose down
```

You do **not** need to quit Colima or Docker—only stop the Compose project. Then follow the steps below.

**1. Start cluster**

```bash
minikube start --driver=docker
```

**2. Build image (repo root)**

```bash
docker build -t heart-disease-api:latest .
```

**3. Load image into Minikube**

```bash
minikube image load heart-disease-api:latest
```

**4. Apply manifests (full stack: API + Prometheus + Grafana)**

```bash
kubectl apply -k k8s/
```

This uses **`k8s/kustomization.yaml`**. API-only apply is still possible if you omit monitoring files:  
`kubectl apply -f k8s/namespace.yaml -f k8s/deployment.yaml -f k8s/service.yaml`  
Optional extras: **`kubectl apply -f k8s/hpa.yaml`**, **`kubectl apply -f k8s/ingress.yaml`**.

**5. Wait for rollout**

```bash
kubectl rollout status deployment/heart-disease-api -n heart-disease --timeout=180s
kubectl rollout status deployment/prometheus -n heart-disease --timeout=120s
kubectl rollout status deployment/grafana -n heart-disease --timeout=180s
```

**6. Check resources**

```bash
kubectl get pods,svc,deployment,endpoints -n heart-disease
```

Pods for **`heart-disease-api`**, **`prometheus`**, and **`grafana`** should be **Running**. **Endpoints** for `heart-disease-api-service` should list pod IPs. If something fails: `kubectl describe pod -n heart-disease -l app=heart-disease-api` (and similarly for **`app=prometheus`** / **`app=grafana`**), plus **`kubectl logs`** for the failing pod.

**7. Port-forward (easiest way to reach Services on Minikube)**

Inside Minikube, Services run **inside the cluster**. From your Mac, the straightforward way to open the API, Prometheus, and Grafana is **`kubectl port-forward`** — stable localhost ports, no dependence on `minikube service` tunnels (which are fiddly on **macOS + Docker driver**).

Open **one terminal per forward** (or keep only the service you need), leave each command running:

```bash
# Terminal 1 — API (Service port 80 → app on 8000 in pod)
kubectl port-forward -n heart-disease svc/heart-disease-api-service 8080:80

# Terminal 2 — Prometheus (after kubectl apply -k k8s/)
kubectl port-forward -n heart-disease svc/prometheus 9090:9090

# Terminal 3 — Grafana
kubectl port-forward -n heart-disease svc/grafana 3000:3000
```

| Local URL | What |
|-----------|------|
| **http://127.0.0.1:8080/docs** | FastAPI |
| **http://127.0.0.1:9090** | Prometheus |
| **http://127.0.0.1:3000** | Grafana (**admin** / **admin123**) |

**8. Test the API and generate metrics for Grafana** (with **Terminal 1** port-forward still running)

Many dashboard panels (e.g. **`predict_requests_total`**) only appear **after** at least one **`POST /predict`** — the Python client creates those time series on first use. Prometheus scrapes every **5s**, so wait **~15–30 seconds** after traffic, then refresh Grafana (**time range: Last 15 minutes**).

```bash
curl -fsS http://127.0.0.1:8080/health
PRED='{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":1}'
for i in 1 2 3 4 5; do
  curl -fsS -X POST http://127.0.0.1:8080/predict \
    -H "Content-Type: application/json" -d "$PRED" >/dev/null
  echo "predict $i ok"
done
```

With **Terminal 2** (Prometheus) open, check **Status → Targets**: job **`heart-disease-api`** should be **UP**. In **Graph**, try: **`up{job="heart-disease-api"}`** (should be **1**) then **`predict_requests_total`**.

**Optional — `minikube service … --url`** can print a URL for the LoadBalancer Service, but on macOS the tunnel often **must stay open** in another window. Prefer **`kubectl port-forward`** for predictable **127.0.0.1** ports.

**Cleanup (example)** — symmetric with **`kubectl apply -k k8s/`**; frees in-cluster resources. Stop **port-forwards** first (**Ctrl+C** in each `kubectl port-forward` terminal). See **[Clean stopping](#clean-stopping)** for the full checklist (Compose, Minikube, volumes).

```bash
kubectl delete -k k8s/
# If anything remains:  kubectl delete namespace heart-disease
minikube stop   # optional; use minikube delete only if you want to wipe the cluster
```

### Kubernetes manifest summary

- **`k8s/namespace.yaml`** — namespace `heart-disease`
- **`k8s/deployment.yaml`** — Deployment `heart-disease-api`, image `heart-disease-api:latest`, probes on `/health`, **1** replica by default (avoids per-pod metrics confusion with Prometheus; scale up if you need HA)
- **`k8s/service.yaml`** — LoadBalancer Service `heart-disease-api-service`, port **80** → container **8000**
- **`k8s/prometheus-*.yaml`** — Prometheus Deployment + ConfigMap scrape target **`heart-disease-api-service:80`**, Service port **9090**
- **`k8s/grafana-*.yaml`** — Grafana Deployment (admin **`admin`** / **`admin123`**), dashboard ConfigMaps, Service port **3000**
- **`k8s/kustomization.yaml`** — **`kubectl apply -k k8s/`** applies API + Prometheus + Grafana together
- **`k8s/hpa.yaml`**, **`k8s/ingress.yaml`** — optional autoscaling / ingress (apply separately if needed)

Use **step 7** port-forwards for Prometheus and Grafana as well as the API. Send **`POST /predict`** to **http://127.0.0.1:8080** so Prometheus gets metrics and Grafana dashboards fill in.

### Prometheus and Grafana: Docker Compose vs Kubernetes

| Where | Command | Stack |
|--------|---------|--------|
| **Docker** | `docker compose up -d --build` | **`docker-compose.yml`** — API + Prometheus + Grafana |
| **Kubernetes** | `kubectl apply -k k8s/` after **`minikube image load`** | Same three roles in-cluster; Prometheus scrapes **`heart-disease-api-service`** in **`heart-disease`**. |

**Compose** is the quickest full stack on one host; **Minikube + `kubectl apply -k k8s/`** deploys the **same services** as Kubernetes workloads.

---

## Clean stopping

Shut down what you started so **ports** (8000, 8080, 9090, 3000, 5050) and **clusters** are free. Order below is safe for typical use; you can skip rows that do not apply.

| How you ran it | Clean stop |
|----------------|------------|
| **Local API** ([Quick start](#7-start-the-api-locally) step **7**) | **Ctrl+C** in the terminal running `api.py` or `uvicorn`. |
| **MLflow UI** ([step **4**](#4-mlflow-ui-optional)) | **Ctrl+C** in that terminal. |
| **Single-container Docker** ([step **8**](#8-docker-api-image-only)) | **Ctrl+C**; the example `docker run` uses **`--rm`**, so the container is removed on exit. |
| **Docker Compose** ([step **9**](#9-monitoring-stack-api--prometheus--grafana), [script](#10-one-shot-script-optional)) | From repo root: **`docker compose down`** or **`docker-compose down`**. Frees **8000**, **9090**, **3000**. |
| **`kubectl port-forward`** ([Kubernetes step **7**](#kubernetes-deployment-minikube)) | **Ctrl+C** in each forward terminal (**8080**, **9090**, **3000**). |
| **Kubernetes workloads** | From repo root: **`kubectl delete -k k8s/`** (matches **`apply -k`**). If the namespace lingers: **`kubectl delete namespace heart-disease`**. |
| **Minikube** (optional) | After deletes: **`minikube stop`**. Use **`minikube delete`** only if you want to remove the cluster VM entirely. |

**Switching Docker → Kubernetes:** run **`docker compose down`** (or **`docker-compose down`**) before Minikube so **8000** is not still bound by Compose and you do not hit the wrong API.

**Compose volumes:** By default, **`docker compose down`** removes containers and the Compose network but **keeps** the named volume **`grafana_data`** (Grafana’s local DB). To remove that volume too: **`docker compose down -v`** — dashboard JSON still comes from the repo mount; you only reset Grafana’s stored state.

---

## Model details

- **Dataset:** UCI Heart Disease (bundled as `data/heart_disease_UCI_dataset.csv`; ~303 rows, 13 clinical features, binary `target`)
- **Preprocessing:** shared in `src/data_preprocessing/pre_processing_data.py`; EDA writes `data/heart_disease_processed_dataset.csv`
- **Models:** **Logistic Regression** (with scaling) and **Random Forest** (with imputation), tuned via **GridSearchCV** with stratified CV, scoring **ROC-AUC**
- **Selection:** best model by **hold-out test ROC-AUC**; saved to `models/best_model.pkl` with `models/feature_names.pkl` for inference
- **Experiment tracking:** **MLflow** under `mlruns/` (experiment name default `heart_disease_prediction`)

Exact metrics for your run are in `models/training_metadata.json` and in the MLflow UI.

---

## API endpoints

**Base URL** depends on how you run the API:

| Setup | Base URL |
|--------|----------|
| Local `uvicorn` / **Docker** / **Compose** | `http://127.0.0.1:8000` |
| **Kubernetes** (this README’s `kubectl port-forward … 8080:80`) | `http://127.0.0.1:8080` |

- **`GET /docs`** — Swagger UI  
- **`GET /health`** — Liveness; JSON includes model load status  
- **`POST /predict`** — JSON body `PatientData` → prediction, label, confidence, risk level  
- **`GET /metrics`** — Prometheus text exposition  

### `/predict` request body (example)

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 2,
  "ca": 0,
  "thal": 1
}
```

### Sample response shape

```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "confidence": 0.7081,
  "risk_level": "HIGH"
}
```

Field names match `PredictionResponse` in `src/api/api.py`. Numeric values are **examples**; `confidence`, `prediction`, `label`, and `risk_level` depend on the trained model and the request body. **`risk_level` bands:** confidence below 0.35 → LOW, below 0.65 → MEDIUM, otherwise HIGH.

**Shell example** (swap the host/port for your setup — **8000** local/Compose, **8080** with K8s port-forward):

```bash
PRED='{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":1}'
curl -fsS -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "$PRED"
# Kubernetes:  http://127.0.0.1:8080/predict
```

If **zsh** prints a lone **`%`** after the JSON line, that only means the response had no trailing newline — the request still succeeded.

---

## CI/CD pipeline (GitHub Actions)

Workflow: **`.github/workflows/ci.yml`** — in the GitHub UI the workflow name is **CI/CD Pipeline**.

**Triggers:** push and pull request to **`main`** / **`master`**, plus **workflow_dispatch** (manual).

**Jobs (sequential):**

1. **`lint`** — `flake8 src tests`
2. **`testing`** — `pytest tests/` (writes `pytest-results.xml`), uploads artifact **`test-results`** (`if: always()` on the upload step)
3. **`training-model`** — `python src/eda/eda.py preprocess`, `python src/model_training/train.py`, uploads **`ml-training`** (`models/` + `mlruns/`) on success
4. **`docker-build-smoke`** — downloads **`ml-training`**, `docker build -t heart-disease-api:ci .`, smoke **`/health`** and **`POST /predict`**

CI **does not** start Grafana or Prometheus; use **Docker Compose** locally for that.

**Artifacts:** **`test-results`**, **`ml-training`** (14-day retention) — download from **Actions → workflow run → Artifacts**.

**Local parity (core ML steps):**

```bash
python3 -m flake8 src tests
python3 -m pytest tests/ -v
python3 src/eda/eda.py preprocess
python3 src/model_training/train.py
```

---

## Monitoring (Prometheus and Grafana)

### Compose quick start

```bash
docker-compose up -d --build
```

**Services**

- **heart-disease-api** — http://localhost:8000  
- **prometheus** — http://localhost:9090  
- **grafana** — http://localhost:3000 — **admin** / **admin123**  

### Dashboard

`monitoring/grafana/heart_disease_api.json` holds the dashboard panels and PromQL queries. The YAML files under `monitoring/grafana/provisioning/` tell Grafana to attach to Prometheus and to import dashboards from the configured folder at startup.

### Metrics (API)

Prometheus counters/histograms/gauges include (names from `src/api/api.py`):

- **`predict_requests_total`** — `/predict` requests by `status`
- **`predict_request_latency_seconds`** — latency histogram
- **`prediction_label_total`** — counts by predicted label
- **`prediction_risk_level_total`** — counts by risk band
- **`last_prediction`**, **`last_prediction_confidence`** — gauges for latest request
- **`prediction_confidence`** — confidence histogram

### Logging

Structured logging to **stdout** and, when configured, **`API_LOG_PATH`** (under Compose often `./logs/api_requests.log` via `docker-compose.yml`).

### Cleanup

```bash
docker-compose down
# or:  docker compose down
```

For a full shutdown checklist (port-forwards, K8s, volumes), see **[Clean stopping](#clean-stopping)**.

---

## Troubleshooting

- **Port 8000 in use** — `lsof -nP -iTCP:8000 -sTCP:LISTEN`; stop the conflicting process or use only Compose for the API.
- **`Clean data not found` when training** — Run `python3 src/eda/eda.py all` or `preprocess` first.
- **Grafana “No data” (Compose or Minikube)** — (1) **Generate traffic**: several **`POST /predict`** to the API (panels like `predict_requests_total` stay empty until then). (2) **Wait** one or two Prometheus scrape intervals (~5–15s here), set Grafana time range to **Last 15 minutes**, refresh. (3) **Prometheus** (port-forward **9090** on K8s): **Status → Targets** → **`heart-disease-api`** must be **UP**; **Graph** → query **`up{job="heart-disease-api"}`**. If **DOWN**, test from a throwaway pod:  
`kubectl run -n heart-disease curl-metrics --rm -i --restart=Never --image=curlimages/curl -- curl -sS http://heart-disease-api-service/metrics | head`  
(expect `# HELP` / metric lines; **ENDPOINTS** for the Service must not be empty). (4) In Grafana **Connections → Data sources → Prometheus → Save & test** should succeed. (5) **Explore** → run `predict_requests_total` after step (1).
- **`docker-compose` vs `docker compose`** — Install the [Compose CLI](https://docs.docker.com/compose/install/) or use the Docker Desktop plugin; subcommands are the same.
- **Docker permission denied (e.g. Colima)** — `colima stop && colima start`; verify `docker info`.
- **`test_full_flow.sh` step [6/6] — Docker daemon not available** — Start your engine (`colima start` with Colima, or Docker Desktop), then `docker info` until a **Server** section appears. Or finish without Compose: `SKIP_COMPOSE=1 bash scripts/test_full_flow.sh` or `--skip-compose`.
- **`/predict` 500 in container** — Rebuild image; this repo pins **`scikit-learn==1.6.1`** in `requirements.txt` for pickle compatibility.
- **Training parallel errors** — `export MLOPS_N_JOBS=1` before `train.py` in restricted environments.
- **Minikube (macOS, Docker driver) — `minikube service --url` works once then `curl` fails** — Prefer **`kubectl port-forward`** for the API, Prometheus, and Grafana (see **Kubernetes step 7** in this README). That gives fixed **127.0.0.1** ports and no tunnel flakiness.
- **Minikube — `ImagePullBackOff` / image not found** — Build on the host, then **`minikube image load heart-disease-api:latest`**, or build inside Minikube’s Docker: `eval $(minikube docker-env)` then `docker build -t heart-disease-api:latest .`
- **Prometheus query empty but `curl …:8080/metrics` shows `predict_requests_total`** — Often **multiple API replicas**: metrics live **per pod**; scrapes via the Service may hit a pod that never got `/predict`. **`k8s/deployment.yaml`** defaults to **1** replica for demos. If you use **2+**, send enough traffic or **`kubectl scale deployment/heart-disease-api -n heart-disease --replicas=1`**, wait for rollout, send predicts again, wait ~15s, re-query Prometheus.
- **zsh shows `%` after `curl` JSON** — Harmless: the shell marks a line that did not end with a newline. Use **`curl -fsS … | cat`** or ignore.

---

## Data

### Files in `data/`

- **`heart_disease_UCI_dataset.csv`** — Bundled **raw** UCI-style heart disease data. Default input for EDA and for `load_data()` when scoring raw rows in batch inference.
- **`heart_disease_processed_dataset.csv`** — **Generated** by `python src/eda/eda.py preprocess` or `all`. Numeric features + binary `target`; **`train.py` loads this** for modelling. Not required for the HTTP API (the API scores **raw** rows via the same cleaning code in memory).
- **`batch_predictions.csv`** — **Optional.** Created only if you run batch inference **with** `--output`, e.g. `python src/model_training/inference.py --output data/batch_predictions.csv`. It is **not** needed for training, CI, or serving. **Purpose:** save offline predictions (class, probability, and `actual_target` when the input still has labels) for reports, coursework, or archiving—without calling `POST /predict`. Omit `--output` if you only want a terminal preview.

### Path constants (`src/config/paths.py`)

- **`RAW_DATA_CSV`** → `data/heart_disease_UCI_dataset.csv`
- **`CLEAN_DATA_CSV`** → `data/heart_disease_processed_dataset.csv` (same file EDA writes and training reads by default)

### Citation

- [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

---


