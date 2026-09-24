#!/usr/bin/env bash
set -euo pipefail

backend_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_name="papermind-full-stack-$$"

cleanup() {
  docker compose --project-directory "$backend_dir" -p "$project_name" -f "$backend_dir/compose.test.yaml" down --volumes --remove-orphans
}

trap cleanup EXIT

docker compose --project-directory "$backend_dir" -p "$project_name" -f "$backend_dir/compose.test.yaml" up -d --wait
(
  cd "$backend_dir"
  RUN_FULL_STACK_TESTS=1 uv run pytest -m full_stack -q
)
