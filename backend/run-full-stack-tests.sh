#!/usr/bin/env bash
set -euo pipefail

backend_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_name="papermind-full-stack-$$"

# Isolated, ephemeral test credentials: never the development password from
# backend/.infra.env and never committed. A fresh password per run keeps one
# run's database unreachable from any other checkout.
export POSTGRES_TEST_USER="${POSTGRES_TEST_USER:-papermind_test}"
export POSTGRES_TEST_DB="${POSTGRES_TEST_DB:-papermind_test}"
export POSTGRES_TEST_PASSWORD="${POSTGRES_TEST_PASSWORD:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')}"
export FULL_STACK_DATABASE_URL="${FULL_STACK_DATABASE_URL:-postgresql+psycopg://${POSTGRES_TEST_USER}:${POSTGRES_TEST_PASSWORD}@127.0.0.1:55432/${POSTGRES_TEST_DB}}"

cleanup() {
  docker compose --project-directory "$backend_dir" -p "$project_name" -f "$backend_dir/compose.test.yaml" down --volumes --remove-orphans
}

trap cleanup EXIT

docker compose --project-directory "$backend_dir" -p "$project_name" -f "$backend_dir/compose.test.yaml" up -d --wait
(
  cd "$backend_dir"
  RUN_FULL_STACK_TESTS=1 uv run pytest -m full_stack -q
)
