#!/usr/bin/env bash
# PaperMind local workspace: one command from a clean clone to a cited answer.
#
#   ./papermind.sh up [--seed] [--smoke] [--no-frontend]   start everything
#   ./papermind.sh down                                    stop, keep data
#   ./papermind.sh status                                  report readiness
#   ./papermind.sh smoke                                   prove setup
#
# Idempotent: rerunning `up` reuses secrets, volumes, migrations, and live
# processes, and recovers from an interrupted run without deleting user data.
# Shutdown stops processes and containers while preserving persistent volumes
# and user files (volumes are never deleted here).
set -euo pipefail

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BACKEND_DIR="$REPO_ROOT/backend"
FRONTEND_DIR="$REPO_ROOT/frontend"
COMPOSE_FILE="$BACKEND_DIR/compose.yaml"
PID_DIR="$BACKEND_DIR/data/.local"
BACKEND_PID="$PID_DIR/backend.pid"
WORKER_PID="$PID_DIR/worker.pid"
FRONTEND_PID="$PID_DIR/frontend.pid"
API_URL="${PAPERMIND_API_URL:-http://127.0.0.1:3000}"

log() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
recovery() { printf 'Fix: %s\n' "$*" >&2; }

need_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "missing tools: '$1' is not installed. Fix: install $1 (${2:-see README.md}), then rerun ./papermind.sh up."
  fi
}

check_prerequisites() {
  need_tool docker "Docker Desktop (https://docs.docker.com/get-docker)"
  need_tool uv "uv (https://docs.astral.sh/uv/getting-started/installation)"
  need_tool python3 "Python 3.11+"
  need_tool node "Node 20+ (https://nodejs.org)"
  need_tool npm "Node 20+ (ships with node)"
  need_tool curl "curl"
  if ! docker info >/dev/null 2>&1; then
    die "unavailable services: Docker daemon is not running. Fix: start Docker Desktop and rerun ./papermind.sh up."
  fi
}

check_ports() {
  # Occupied ports: only fail when the listener is not one of our own processes.
  for mapping in "5432:Postgres" "6333:Qdrant HTTP" "6334:Qdrant gRPC" "3000:backend API" "5173:frontend"; do
    port="${mapping%%:*}"; name="${mapping##*:}"
    if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
      case "$port" in
        3000|5173)
          pidfile=""; [ "$port" = "3000" ] && pidfile="$BACKEND_PID"
          [ "$port" = "5173" ] && pidfile="$FRONTEND_PID"
          if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then continue; fi
          ;;
      esac
      die "occupied ports: port $port ($name) is already in use. Fix: stop the process holding it (lsof -iTCP:$port -sTCP:LISTEN) or free the port, then rerun ./papermind.sh up."
    fi
  done
}

ensure_secrets() {
  # Creates local configuration and generates required secrets (idempotent:
  # a healthy password is never rotated, so reruns keep the database working).
  # Covers invalid secrets: a placeholder password aborts with guidance.
  bash "$BACKEND_DIR/scripts/bootstrap-local.sh"
  if grep -q '^POSTGRES_PASSWORD=replace-me' "$BACKEND_DIR/.env" 2>/dev/null; then
    die "invalid secrets: $BACKEND_DIR/.env still holds the placeholder password. Fix: delete the POSTGRES_PASSWORD line in backend/.env and rerun ./papermind.sh up to generate a random one."
  fi
  if [ ! -f "$BACKEND_DIR/.env" ] || ! grep -q '^APP_SECRET=' "$BACKEND_DIR/.env"; then
    secret=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    printf '\nAPP_SECRET=%s\n' "$secret" >> "$BACKEND_DIR/.env"
    log "Generated APP_SECRET in backend/.env (encrypts stored provider keys)."
  fi
  POSTGRES_USER=$(grep -E '^POSTGRES_USER=' "$BACKEND_DIR/.env" 2>/dev/null | cut -d= -f2- | tail -n 1)
  POSTGRES_PASSWORD=$(grep -E '^POSTGRES_PASSWORD=' "$BACKEND_DIR/.env" 2>/dev/null | cut -d= -f2- | tail -n 1)
  export POSTGRES_USER
  if [ -z "${POSTGRES_PASSWORD:-}" ]; then
    die "invalid secrets: POSTGRES_PASSWORD is empty in backend/.env. Fix: delete the POSTGRES_PASSWORD line in backend/.env and rerun ./papermind.sh up."
  fi
  if [ ! -f "$FRONTEND_DIR/.env.local" ] && [ -z "${VITE_API_URL:-}" ]; then
    cp "$FRONTEND_DIR/.env.example" "$FRONTEND_DIR/.env.local"
    log "Created frontend/.env.local from .env.example (VITE_API_URL=$API_URL)."
  fi
}

compose_up() {
  (cd "$REPO_ROOT" && docker compose -f backend/compose.yaml up -d)
}

wait_for_db() {
  log "Waiting for Postgres..."
  for _ in $(seq 1 60); do
    if docker compose -f "$COMPOSE_FILE" exec -T db pg_isready -U "${POSTGRES_USER:-papermind}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  die "unavailable services: Postgres never became ready. Fix: run 'docker compose -f backend/compose.yaml logs db' and confirm port 5432 is free, then rerun ./papermind.sh up."
}

wait_for_qdrant() {
  log "Waiting for Qdrant..."
  for _ in $(seq 1 60); do
    if curl -sf http://127.0.0.1:6333/collections >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  die "unavailable services: Qdrant never answered at http://127.0.0.1:6333. Fix: run 'docker compose -f backend/compose.yaml logs qdrant', then rerun ./papermind.sh up."
}

run_migrations() {
  (cd "$BACKEND_DIR" && uv sync --frozen >/dev/null && uv run alembic upgrade head)
}

start_process() {
  name="$1"; pidfile="$2"; logfile="$3"; shift 3
  mkdir -p "$PID_DIR"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    log "Keeping existing $name (pid $(cat "$pidfile"))."
    return 0
  fi
  log "Starting $name..."
  # shellcheck disable=SC2068
  nohup "$@" >"$logfile" 2>&1 &
  echo $! >"$pidfile"
  sleep 1
  if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    die "unavailable services: $name exited immediately. Fix: read $logfile for the traceback, fix backend/.env (see backend/.env.example), then rerun ./papermind.sh up."
  fi
}

wait_for_api() {
  log "Waiting for the backend API..."
  for _ in $(seq 1 60); do
    if curl -sf "$API_URL/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  die "unavailable services: backend never answered $API_URL/health. Fix: read $PID_DIR/backend.log, confirm DATABASE_URL and QDRANT_URL in backend/.env, then rerun ./papermind.sh up."
}

readiness() {
  # Confirms the database, migration state, Qdrant, and worker before success.
  log "Checking readiness (database, migrations, Qdrant, worker)..."
  (cd "$BACKEND_DIR" && uv run alembic current) | grep -q . \
    || die "invalid secrets or unavailable services: cannot read migration state. Fix: confirm DATABASE_URL in backend/.env matches POSTGRES_PASSWORD in the same file, then rerun ./papermind.sh up."
  curl -sf http://127.0.0.1:6333/collections >/dev/null \
    || die "unavailable services: Qdrant unreachable. Fix: run 'docker compose -f backend/compose.yaml ps', then rerun ./papermind.sh up."
  curl -sf "$API_URL/health" >/dev/null \
    || die "unavailable services: backend unreachable. Fix: read $PID_DIR/backend.log, then rerun ./papermind.sh up."
  if [ -f "$WORKER_PID" ] && kill -0 "$(cat "$WORKER_PID")" 2>/dev/null; then
    log "  worker alive (pid $(cat "$WORKER_PID"))"
  else
    die "unavailable services: ingestion worker is not running. Fix: read $PID_DIR/worker.log, then rerun ./papermind.sh up."
  fi
  log "Ready: database reachable, migrations applied, Qdrant answering, worker alive."
}

cmd_up() {
  seed=0; smoke=0; frontend=1
  for arg in "$@"; do
    case "$arg" in
      --seed) seed=1 ;;
      --smoke) smoke=1 ;;
      --no-frontend) frontend=0 ;;
      --help) usage; exit 0 ;;
      *) die "unknown flag '$arg'. Fix: run ./papermind.sh --help for the supported flags." ;;
    esac
  done
  check_prerequisites
  check_ports
  ensure_secrets
  compose_up
  wait_for_db
  wait_for_qdrant
  run_migrations
  start_process "worker" "$WORKER_PID" "$PID_DIR/worker.log" env -C "$BACKEND_DIR" uv run python worker.py
  start_process "backend" "$BACKEND_PID" "$PID_DIR/backend.log" env -C "$BACKEND_DIR" uv run python app.py
  wait_for_api
  if [ "$frontend" = "1" ]; then
    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
      log "Installing frontend dependencies (one-time)..."
      (cd "$FRONTEND_DIR" && npm install) || die "missing tools or unavailable services: 'npm install' failed. Fix: confirm Node 20+ and network access, then rerun ./papermind.sh up."
    fi
    start_process "frontend" "$FRONTEND_PID" "$PID_DIR/frontend.log" env -C "$FRONTEND_DIR" npm run dev -- --port 5173 --strictPort --host 127.0.0.1
    log "Frontend: http://127.0.0.1:5173 (VITE_API_URL=$API_URL)"
  fi
  readiness
  if [ "$seed" = "1" ]; then
    log "Seeding licensed sample documents (no model key needed)..."
    (cd "$BACKEND_DIR" && PAPERMIND_API_URL="$API_URL" uv run python scripts/seed_sample_docs.py) \
      || die "seed failed. Fix: confirm the worker is alive (./papermind.sh status) and rerun ./papermind.sh up --seed."
  fi
  if [ "$smoke" = "1" ]; then
    cmd_smoke
  fi
  log ""
  log "PaperMind is up: API $API_URL/health, frontend http://127.0.0.1:5173"
  log "Stop cleanly with ./papermind.sh down (volumes and uploads are kept)."
}

cmd_down() {
  log "Stopping PaperMind (volumes and user files are preserved)..."
  for pidfile in "$FRONTEND_PID" "$BACKEND_PID" "$WORKER_PID"; do
    if [ -f "$pidfile" ]; then
      pid=$(cat "$pidfile")
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$pid" 2>/dev/null || true
        log "  stopped pid $pid"
      fi
      rm -f "$pidfile"
    fi
  done
  if docker compose -f "$COMPOSE_FILE" ps -q 2>/dev/null | grep -q .; then
    (cd "$REPO_ROOT" && docker compose -f backend/compose.yaml stop)
  fi
  log "Stopped. Data kept in Docker volumes and backend/data/storage."
}

cmd_status() {
  ok=0
  curl -sf "$API_URL/health" >/dev/null 2>&1 && { log "  backend: up ($API_URL)"; } || { log "  backend: down"; ok=1; }
  curl -sf http://127.0.0.1:6333/collections >/dev/null 2>&1 && { log "  qdrant: up"; } || { log "  qdrant: down"; ok=1; }
  docker compose -f "$COMPOSE_FILE" exec -T db pg_isready >/dev/null 2>&1 && { log "  database: up"; } || { log "  database: down"; ok=1; }
  if [ -f "$WORKER_PID" ] && kill -0 "$(cat "$WORKER_PID")" 2>/dev/null; then log "  worker: up"; else log "  worker: down"; ok=1; fi
  if [ -f "$FRONTEND_PID" ] && kill -0 "$(cat "$FRONTEND_PID")" 2>/dev/null; then log "  frontend: up"; else log "  frontend: down (or --no-frontend)"; fi
  return $ok
}

cmd_smoke() {
  (cd "$BACKEND_DIR" && PAPERMIND_API_URL="$API_URL" uv run python scripts/smoke_setup.py)
}

usage() {
  cat <<'EOF'
Usage: ./papermind.sh <command> [flags]

  up [--seed] [--smoke] [--no-frontend]  verify prerequisites, generate
      secrets, start pinned services, migrate, start worker + API + frontend.
      --seed adds licensed sample documents and a safe sample conversation
      without requiring a hosted model key.
  down          stop processes and containers, keep volumes and uploads.
  status        report database, Qdrant, worker, backend, frontend readiness.
  smoke         prove setup with a cited answer or deterministic abstention.
EOF
}

cmd="${1:- --help}"
case "$cmd" in
  up) shift; cmd_up "$@" ;;
  down) cmd_down ;;
  status) cmd_status ;;
  smoke) cmd_smoke ;;
  --help|-h|help) usage ;;
  *) die "unknown command '$cmd'. Fix: run ./papermind.sh --help." ;;
esac
