#!/usr/bin/env bash
#
# =============================================================================
# test_full_flow.sh — one-command end-to-end run
# =============================================================================
# Author: SANDIP BHATTACHARYYA — BITS Pilani ID 2025cs05025
#
# Default behavior (no args):
#   flake8 → pytest → EDA → train → batch predict → Docker Compose monitoring
#   (API :8000 + Prometheus :9090 + Grafana :3000). On success the stack stays up;
#   stop manually: docker compose down  (or docker-compose down).
#
# Without Docker (steps 1–5 only):  SKIP_COMPOSE=1 bash scripts/test_full_flow.sh
#   or:  bash scripts/test_full_flow.sh --skip-compose
#
set -euo pipefail

SKIP_COMPOSE="${SKIP_COMPOSE:-0}"
for _arg in "$@"; do
  case "${_arg}" in
    --skip-compose) SKIP_COMPOSE=1 ;;
    -h|--help)
      echo "Usage: bash scripts/test_full_flow.sh [--skip-compose]"
      echo "  --skip-compose   Lint, tests, EDA, train, batch inference only (no Compose)."
      echo "  Environment:     SKIP_COMPOSE=1 bash scripts/test_full_flow.sh"
      exit 0
      ;;
  esac
done

# If `docker` is not on PATH (common when Docker Desktop is installed but the CLI dir was never linked),
# prepend typical locations so step [6/6] works without manual PATH edits.
ensure_docker_on_path() {
  if command -v docker >/dev/null 2>&1; then
    return 0
  fi
  local d
  for d in \
    "/Applications/Docker.app/Contents/Resources/bin" \
    "${HOME}/.docker/bin" \
    "/usr/local/bin" \
    "/opt/homebrew/bin"
  do
    if [[ -x "${d}/docker" ]]; then
      export PATH="${d}:${PATH}"
      hash -r 2>/dev/null || true
      echo "Note: added Docker CLI to PATH → ${d}"
      return 0
    fi
  done
  return 1
}

docker_daemon_ready() {
  command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ensure_docker_on_path || true

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
  echo "Using virtualenv: .venv"
else
  echo "Tip: create and activate .venv first (see README → Quick start)." >&2
fi

echo "==> [1/6] Lint (flake8)"
flake8 src tests

echo "==> [2/6] Unit tests (pytest) → pytest-results.xml"
pytest tests/ -v --tb=short --junitxml=pytest-results.xml

echo "==> [3/6] EDA — load, clean CSV, plots (python src/eda/eda.py all)"
python src/eda/eda.py all

echo "==> [4/6] Train models + MLflow + save models/"
python src/model_training/train.py

echo "==> [5/6] Batch inference sample"
python src/model_training/inference.py --output data/batch_predictions.csv

PREDICT_JSON='{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":1}'

# Colima / some Homebrew docker CLIs misparse `docker compose -f path`; run from repo root
# so Compose finds docker-compose.yml by name. Fallback to `docker-compose` if plugin missing.
docker_compose() {
  (
    cd "$ROOT" || exit 1
    if docker compose version >/dev/null 2>&1; then
      docker compose "$@"
    elif command -v docker-compose >/dev/null 2>&1; then
      docker-compose "$@"
    else
      echo "ERROR: Docker Compose not found. Install the Compose plugin or standalone binary:" >&2
      echo "  brew install docker-compose" >&2
      exit 1
    fi
  )
}

compose_down_safe() {
  docker_compose down --remove-orphans 2>/dev/null || true
}

# Returns 0 when Prometheus TSDB has at least one predict_requests_total series.
prometheus_has_predict_metrics() {
  curl -fsS -G "http://127.0.0.1:9090/api/v1/query" --data-urlencode "query=predict_requests_total" 2>/dev/null \
    | python -c "
import sys, json
j = json.load(sys.stdin)
if j.get('status') != 'success':
    sys.exit(1)
series = j.get('data', {}).get('result') or []
sys.exit(0 if len(series) >= 1 else 1)
" 2>/dev/null
}

monitoring_stack() {
  echo "==> [6/6] Docker Compose — API (8000) + Prometheus (9090) + Grafana (3000)"
  compose_down_safe

  if ! docker_compose up -d --build; then
    compose_down_safe
    exit 1
  fi

  local api_ok=0 prom_ok=0 graf_ok=0
  for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1 && api_ok=1
    curl -fsS "http://127.0.0.1:9090/-/ready" >/dev/null 2>&1 && prom_ok=1
    curl -fsS "http://127.0.0.1:3000/api/health" >/dev/null 2>&1 && graf_ok=1
    if [[ "$api_ok" -eq 1 && "$prom_ok" -eq 1 && "$graf_ok" -eq 1 ]]; then
      break
    fi
    sleep 3
  done

  if [[ "$api_ok" -ne 1 ]]; then
    echo "ERROR: API not healthy on :8000" >&2
    docker_compose logs --tail=80 heart-disease-api >&2 || true
    compose_down_safe
    exit 1
  fi
  if [[ "$prom_ok" -ne 1 ]]; then
    echo "ERROR: Prometheus not ready on :9090 (waited ~3m; http://127.0.0.1:9090/-/ready never succeeded)." >&2
    echo "  Common causes: host port 9090 already in use; Prometheus container exited (bad config/mount); slow image pull." >&2
    echo "  DEBUG — curl:" >&2
    curl -sS -o /dev/null -w "    HTTP %{http_code} connect=%{time_connect}s total=%{time_total}s\n" "http://127.0.0.1:9090/-/ready" >&2 || echo "    (connection failed)" >&2
    echo "  DEBUG — docker compose ps:" >&2
    docker_compose ps -a >&2 || true
    echo "  DEBUG — prometheus logs (last 80 lines):" >&2
    docker_compose logs --tail=80 prometheus >&2 || true
    compose_down_safe
    exit 1
  fi
  if [[ "$graf_ok" -ne 1 ]]; then
    echo "ERROR: Grafana not healthy on :3000" >&2
    compose_down_safe
    exit 1
  fi

  echo "  • API /health OK"
  echo "  • Prometheus /-/ready OK"
  echo "  • Grafana /api/health OK"

  echo "  • Grafana — waiting for file-provisioned dashboards…"
  local dash_ok=0
  for _ in $(seq 1 25); do
    if curl -fsS -u admin:admin123 "http://127.0.0.1:3000/api/search?type=dash-db" 2>/dev/null \
      | python -c "import sys,json; u=json.load(sys.stdin); sys.exit(0 if isinstance(u,list) and len(u)>=1 else 1)" 2>/dev/null; then
      dash_ok=1
      break
    fi
    sleep 2
  done
  if [[ "$dash_ok" -ne 1 ]]; then
    echo "ERROR: no dashboards in Grafana — check monitoring/grafana/*.json and provisioning YAML." >&2
    compose_down_safe
    exit 1
  fi
  curl -fsS -u admin:admin123 "http://127.0.0.1:3000/api/search?type=dash-db" \
    | python -c "import sys,json; u=json.load(sys.stdin); print('    found %d dashboard(s), e.g. %r' % (len(u), u[0].get('title','?')))"

  echo "  • Sending 3× POST /predict (for Prometheus metrics)…"
  for _ in 1 2 3; do
    curl -fsS -X POST "http://127.0.0.1:8000/predict" \
      -H "Content-Type: application/json" \
      -d "$PREDICT_JSON" >/dev/null
  done

  echo "  • Waiting for Prometheus to ingest predict_requests_total (5s scrape interval; retry up to ~90s)…"
  local metrics_ok=0
  local attempt
  for attempt in $(seq 1 18); do
    if prometheus_has_predict_metrics; then
      metrics_ok=1
      break
    fi
    sleep 5
  done

  if [[ "$metrics_ok" -ne 1 ]]; then
    echo "ERROR: Prometheus has no predict_requests_total series yet." >&2
    echo "  DEBUG — host /metrics (sample):" >&2
    curl -sS "http://127.0.0.1:8000/metrics" 2>/dev/null | grep -E '^[^#].*predict_|predict_requests' | head -15 >&2 || true
    echo "  DEBUG — Prometheus targets (active):" >&2
    curl -sS "http://127.0.0.1:9090/api/v1/targets?state=active" 2>/dev/null | python -c "
import sys, json
j=json.load(sys.stdin)
for t in j.get('data',{}).get('activeTargets',[]) or []:
    print('   ', t.get('labels',{}).get('job'), t.get('health'), t.get('lastError',''), file=sys.stderr)
" 2>/dev/null || true
    echo "  DEBUG — from Prometheus container → API /metrics (first lines):" >&2
    docker_compose exec -T prometheus wget -qO- "http://heart-disease-api:8000/metrics" 2>/dev/null | grep -E '^[^#].*predict_|predict_requests' | head -10 >&2 || echo "   (wget from prometheus container failed — network/DNS)" >&2
    compose_down_safe
    exit 1
  fi

  curl -fsS -G "http://127.0.0.1:9090/api/v1/query" --data-urlencode "query=predict_requests_total" \
    | python -c "
import sys, json
j = json.load(sys.stdin)
series = j.get('data', {}).get('result') or []
print('    OK — predict_requests_total: %d series (sample labels: %s)' % (len(series), series[0].get('metric', {}) if series else {}))
"

  echo "  • Prometheus query: up{job=\"heart-disease-api\"}"
  curl -fsS -G "http://127.0.0.1:9090/api/v1/query" --data-urlencode 'query=up{job="heart-disease-api"}' \
    | python -c "
import sys, json
j = json.load(sys.stdin)
assert j.get('status') == 'success'
r = j.get('data', {}).get('result') or []
assert r and float(r[0]['value'][1]) == 1.0, 'target not up'
print('    OK — scrape target UP')
"

  echo "  • Stack is healthy and will be left running."
  echo "  • Stop it when you’re done:"
  echo "      docker compose down   (or: docker-compose down)"
}

# Pick up Docker CLI again if Docker Desktop was started while steps 1–5 were running.
ensure_docker_on_path || true

if [[ "${SKIP_COMPOSE}" == "1" ]]; then
  echo ""
  echo "SKIP_COMPOSE=1 — skipping step [6/6] (Docker Compose / monitoring)."
  echo "=== Pipeline steps 1–5 completed ==="
  exit 0
fi

if ! docker_daemon_ready; then
  echo ""
  echo "ERROR: Docker daemon not available — cannot run step [6/6] (Compose monitoring)." >&2
  echo "  Fix: start a Docker engine, then verify with:  docker info" >&2
  echo "  • macOS — Docker Desktop: open the app and wait until it says Docker is running." >&2
  echo "  • macOS/Linux — Colima:  colima start    then:  docker info" >&2
  echo "  • Linux — service:  sudo systemctl start docker   (or your distro equivalent)" >&2
  echo "  Or run without Docker (steps 1–5 only):" >&2
  echo "    SKIP_COMPOSE=1 bash scripts/test_full_flow.sh" >&2
  echo "    bash scripts/test_full_flow.sh --skip-compose" >&2
  echo "  Diagnostics (docker):" >&2
  docker context ls 2>&1 | sed 's/^/    /' >&2 || true
  docker info 2>&1 | tail -n 8 | sed 's/^/    /' >&2 || true
  echo "  Re-run full script after Docker works:  bash scripts/test_full_flow.sh" >&2
  exit 1
fi

monitoring_stack
echo ""
echo "=== Full MLOps pipeline completed (including monitoring stack test) ==="

echo ""
echo "Done."
