#!/usr/bin/env bash
# Generate local-only infrastructure secrets (idempotent, never committed).
#
# Writes a random POSTGRES_PASSWORD and a matching DATABASE_URL into
# backend/.env. Re-running never rotates a healthy password. A known shared
# default is rotated instead of carried forward, and a local DATABASE_URL
# that no longer matches is rewritten so the app and the database cannot
# drift apart silently. A leftover backend/.infra.env from the old split is
# migrated into backend/.env once, then removed.
set -euo pipefail

backend_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
app_env="$backend_dir/.env"
legacy_infra="$backend_dir/.infra.env"

# Passwords that ship in public documentation: never reused for a local install.
shared_defaults="papermind papermind_test postgres password"

# DATABASE_URL shape: postgresql+psycopg://user:password@host:port/db
url_password() {
  printf '%s' "$1" | sed -n 's#.*://[^:/]*:\([^@]*\)@.*#\1#p' || true
}

url_is_local() {
  case "$1" in
    *://*@localhost:* | *://*@127.0.0.1:*) return 0 ;;
    *) return 1 ;;
  esac
}

is_shared_default() {
  local candidate="$1" known
  for known in $shared_defaults; do
    [ "$candidate" = "$known" ] && return 0
  done
  return 1
}

read_env_value() {
  grep -E "^$2=" "$1" | cut -d= -f2- | tail -n 1 || true
}

ensure_line() {
  # Append KEY=VALUE to a file when no line starting with KEY= exists yet.
  local file="$1" key="$2" value="$3"
  if [ -f "$file" ] && grep -qE "^${key}=" "$file"; then
    return 0
  fi
  if [ -f "$file" ]; then
    printf '\n%s=%s\n' "$key" "$value" >> "$file"
  else
    printf '%s=%s\n' "$key" "$value" > "$file"
  fi
}

# One-time migration from the retired split (backend/.infra.env held the
# Postgres password, backend/.env held the app config). Values already in
# backend/.env win; the legacy file is removed once its values have a home.
if [ -f "$legacy_infra" ]; then
  if [ ! -f "$app_env" ]; then
    touch "$app_env"
  fi
  for key in POSTGRES_USER POSTGRES_PASSWORD QDRANT_API_KEY; do
    if ! grep -qE "^${key}=" "$app_env"; then
      legacy_value=$(read_env_value "$legacy_infra" "$key")
      if [ -n "${legacy_value:-}" ]; then
        printf '\n%s=%s\n' "$key" "$legacy_value" >> "$app_env"
        echo "Migrated $key from backend/.infra.env to backend/.env."
      fi
    fi
  done
  rm -f "$legacy_infra"
  echo "Removed backend/.infra.env (superseded by backend/.env)."
fi

existing_url=""
if [ -f "$app_env" ]; then
  existing_url=$(read_env_value "$app_env" DATABASE_URL)
fi
existing_password=$(url_password "$existing_url")

stored_password=""
stored_user=""
if [ -f "$app_env" ]; then
  stored_password=$(read_env_value "$app_env" POSTGRES_PASSWORD)
  stored_user=$(read_env_value "$app_env" POSTGRES_USER)
fi
stored_user=${stored_user:-papermind}

if [ -z "${stored_password:-}" ] || [ "$stored_password" = "replace-me-run-bootstrap-local-sh" ]; then
  if [ -n "$existing_password" ] && ! is_shared_default "$existing_password"; then
    stored_password="$existing_password"
    echo "Reusing the password already in DATABASE_URL in $app_env."
  else
    stored_password=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    if [ -n "$existing_password" ]; then
      echo "Wrote a random POSTGRES_PASSWORD to $app_env: the previous one was a"
      echo "documented default ('$existing_password') shared by every clone."
      echo "An existing database volume still holds the old password, so either"
      echo "reset it once (docker compose -f backend/compose.yaml down -v) or run"
      echo "  ALTER USER papermind WITH PASSWORD '<new password>';"
    else
      echo "Wrote a random POSTGRES_PASSWORD to $app_env."
    fi
  fi
  if grep -qE '^POSTGRES_PASSWORD=' "$app_env" 2>/dev/null; then
    python3 - "$app_env" "$stored_password" <<'PY'
import pathlib
import re
import sys

path, password = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
path.write_text(re.sub(r"(?m)^POSTGRES_PASSWORD=.*$", f"POSTGRES_PASSWORD={password}", text))
PY
  else
    ensure_line "$app_env" POSTGRES_PASSWORD "$stored_password"
  fi
  chmod 600 "$app_env" 2>/dev/null || true
else
  echo "Keeping existing POSTGRES_PASSWORD in $app_env."
fi

ensure_line "$app_env" POSTGRES_USER "$stored_user"
ensure_line "$app_env" QDRANT_API_KEY ""

# Read the password back (without sourcing secrets into the shell history).
pg_password=$(read_env_value "$app_env" POSTGRES_PASSWORD)
pg_user=$(read_env_value "$app_env" POSTGRES_USER)
pg_user=${pg_user:-papermind}

if [ -z "${pg_password:-}" ] || [ "$pg_password" = "replace-me-run-bootstrap-local-sh" ]; then
  echo "ERROR: $app_env has no real POSTGRES_PASSWORD." >&2
  exit 1
fi

expected_url="postgresql+psycopg://${pg_user}:${pg_password}@localhost:5432/papermind"

if [ ! -f "$app_env" ]; then
  printf 'DATABASE_URL=%s\n' "$expected_url" > "$app_env"
  echo "Created $app_env with a matching DATABASE_URL."
elif [ -z "$existing_url" ]; then
  printf '\nDATABASE_URL=%s\n' "$expected_url" >> "$app_env"
  echo "Added DATABASE_URL to $app_env."
elif [ "$existing_url" != "$expected_url" ]; then
  if url_is_local "$existing_url"; then
    # A local URL drifting from the generated password would fail to
    # authenticate; the generated password is the source of truth.
    python3 - "$app_env" "$expected_url" <<'PY'
import pathlib
import re
import sys

path, url = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
path.write_text(re.sub(r"(?m)^DATABASE_URL=.*$", f"DATABASE_URL={url}", text))
PY
    echo "Updated DATABASE_URL in $app_env to match POSTGRES_PASSWORD."
  else
    echo "WARNING: $app_env points DATABASE_URL at a non-localhost database, so it"
    echo "         was left alone. Make sure its password matches POSTGRES_PASSWORD,"
    echo "         or drop the variable to use the local one."
  fi
fi
