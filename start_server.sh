#!/usr/bin/env bash

set -euo pipefail

PORT="${PORT:-8001}"
HOST="0.0.0.0"
APP_MODULE="app:app"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PID=""
TUNNEL_PID=""
TUNNEL_LOG=""
PYTHON_BIN=""
AUTO_RELOAD="${AUTO_RELOAD:-1}"
FORCE_PORT_TAKEOVER="${FORCE_PORT_TAKEOVER:-1}"

log() {
  printf '%s\n' "$1"
}

cleanup() {
  trap - EXIT INT TERM
  log ""
  log "Shutting down..."

  if [[ -n "${TUNNEL_PID}" ]] && kill -0 "${TUNNEL_PID}" 2>/dev/null; then
    kill "${TUNNEL_PID}" 2>/dev/null || true
    wait "${TUNNEL_PID}" 2>/dev/null || true
  fi

  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" 2>/dev/null || true
  fi

  if [[ -n "${TUNNEL_LOG}" && -f "${TUNNEL_LOG}" ]]; then
    rm -f "${TUNNEL_LOG}" || true
  fi

  log "Stopped backend."
}

ensure_port_available() {
  local port=$1

  if ! lsof -ti tcp:$port >/dev/null 2>&1; then
    return 0
  fi

  if [[ "${FORCE_PORT_TAKEOVER}" != "1" ]]; then
    log "ERROR: Port $port is already in use."
    log "Run with another port, for example: PORT=8001 ./start_server.sh"
    log "Or allow takeover: FORCE_PORT_TAKEOVER=1 ./start_server.sh"
    return 1
  fi

  force_free_port "${port}"
}

force_free_port() {
  local port=$1
  local pids=""

  pids="$(lsof -ti tcp:${port} 2>/dev/null | tr '\n' ' ' | xargs 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi

  log "Port ${port} is in use. Stopping existing process(es): ${pids}"
  kill ${pids} 2>/dev/null || true

  for _ in {1..10}; do
    if ! lsof -ti tcp:$port >/dev/null 2>&1; then
      log "Port ${port} is free now."
      return 0
    fi
    sleep 1
  done

  pids="$(lsof -ti tcp:${port} 2>/dev/null | tr '\n' ' ' | xargs 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    log "Force-killing remaining process(es) on port ${port}: ${pids}"
    kill -9 ${pids} 2>/dev/null || true
  fi

  for _ in {1..5}; do
    if ! lsof -ti tcp:$port >/dev/null 2>&1; then
      log "Port ${port} is free now."
      return 0
    fi
    sleep 1
  done

  log "ERROR: Unable to free port ${port}."
  return 1
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

get_lan_ip() {
  local ip=""

  if command -v ipconfig >/dev/null 2>&1; then
    ip=$(ipconfig getifaddr en0 2>/dev/null || true)
    if [[ -z "${ip}" ]]; then
      ip=$(ipconfig getifaddr en1 2>/dev/null || true)
    fi
  fi

  if [[ -z "${ip}" ]] && command -v hostname >/dev/null 2>&1; then
    ip=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
  fi

  printf '%s' "${ip}"
}

start_cloudflare_tunnel() {
  local public_url=""

  if ! command -v cloudflared >/dev/null 2>&1; then
    log "Cloudflare tunnel skipped: cloudflared is not installed."
    return 0
  fi

  TUNNEL_LOG="$(mktemp)"

  log "Starting Cloudflare tunnel..."
  cloudflared tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate >"${TUNNEL_LOG}" 2>&1 &
  TUNNEL_PID=$!

  for _ in {1..30}; do
    if ! kill -0 "${TUNNEL_PID}" 2>/dev/null; then
      break
    fi

    public_url=$(grep -Eo 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' "${TUNNEL_LOG}" | head -n 1 || true)
    if [[ -n "${public_url}" ]]; then
      printf '%s' "${public_url}"
      return 0
    fi

    sleep 1
  done

  log "Cloudflare tunnel started, but no public URL was detected yet."
  return 0
}

start_backend() {
  log "Starting server..."

  cd "${PROJECT_DIR}"

  ensure_port_available "${PORT}"

  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    log "Using active virtualenv: ${VIRTUAL_ENV}"
  elif [[ -x "xtts-env/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_DIR}/xtts-env/bin/python"
    log "Using XTTS virtualenv: ${PROJECT_DIR}/xtts-env"
  elif [[ -x "venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
    log "Using project virtualenv: ${PROJECT_DIR}/venv"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    log "Using system python: ${PYTHON_BIN}"
  else
    log "ERROR: No usable Python interpreter found."
    exit 1
  fi

  if ! "${PYTHON_BIN}" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
    log "ERROR: FastAPI/Uvicorn are not installed in ${PYTHON_BIN}."
    log "Activate the correct environment, or install dependencies there first."
    exit 1
  fi

  if ! "${PYTHON_BIN}" -c "from TTS.api import TTS" >/dev/null 2>&1; then
    log "ERROR: TTS is not installed in ${PYTHON_BIN}."
    log "Activate your XTTS environment before running the server."
    log "Example: source xtts-env/bin/activate"
    exit 1
  fi

  local uvicorn_args=("${APP_MODULE}" "--host" "${HOST}" "--port" "${PORT}")

  if [[ "${AUTO_RELOAD}" != "0" ]]; then
    uvicorn_args+=("--reload")
    log "Auto-reload enabled. File changes will restart the server automatically."
  else
    log "Auto-reload disabled."
  fi

  "${PYTHON_BIN}" -m uvicorn "${uvicorn_args[@]}" &
  BACKEND_PID=$!

  wait_for_http "http://127.0.0.1:${PORT}/"
  log "Server running on port ${PORT}"
}

main() {
  local lan_ip=""
  local lan_url=""
  local public_url=""

  trap cleanup EXIT INT TERM

  start_backend

  lan_ip="$(get_lan_ip)"
  if [[ -n "${lan_ip}" ]]; then
    lan_url="http://${lan_ip}:${PORT}"
  else
    lan_url="http://127.0.0.1:${PORT}"
  fi

  public_url="$(start_cloudflare_tunnel || true)"

  log "Server ready."
  log "Local computer: http://127.0.0.1:${PORT}"
  log "Office network: ${lan_url}"
  if [[ -n "${public_url}" ]]; then
    log "Work from home: ${public_url}"
  else
    log "Work from home: Cloudflare link unavailable right now"
  fi

  wait "${BACKEND_PID}"
}

main "$@"
