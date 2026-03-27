#!/usr/bin/env bash

set -euo pipefail

PORT=8000
HOST="0.0.0.0"
APP_MODULE="app:app"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PID=""

log() {
  printf '%s\n' "$1"
}

cleanup() {
  trap - EXIT INT TERM
  log ""
  log "Shutting down..."

  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" 2>/dev/null || true
  fi

  log "Stopped backend."
}

port_in_use() {
  local port=$1

  lsof -ti "tcp:${port}" >/dev/null 2>&1
}

find_available_port() {
  local port=$1

  while port_in_use "${port}"; do
    printf 'Port %s is in use, trying %s...\n' "${port}" "$((port + 1))" >&2
    port=$((port + 1))
  done

  printf '%s\n' "${port}"
}

wait_for_http() {
  local url="$1"

  for i in {1..30}; do
    if curl --silent --fail "$url" >/dev/null 2>&1; then
      log "Backend ready."
      return 0
    fi
    sleep 1
  done

  log "Backend failed to start."
  return 1
}

start_backend() {
  log "Starting server..."

  cd "${PROJECT_DIR}"

  PORT=$(find_available_port "${PORT}")

  if [[ ! -d "venv" ]]; then
    log "ERROR: venv not found. Run:"
    log "python3 -m venv venv"
    exit 1
  fi

  # 🚀 Run server using venv
  venv/bin/uvicorn "${APP_MODULE}" --host "${HOST}" --port "${PORT}" &
  BACKEND_PID=$!

  wait_for_http "http://127.0.0.1:${PORT}/"
  log "Server running on port ${PORT}"
}

main() {
  trap cleanup EXIT INT TERM

  start_backend

  log "Server ready at: http://127.0.0.1:${PORT}"

  wait "${BACKEND_PID}"
}

main "$@"
