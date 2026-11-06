#!/usr/bin/env bash
# ==============================================================================
# CystoDS Cloudflare Tunnel Launcher
# Exposes the local FastAPI backend (port 8000) securely to the internet.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TUNNEL_LOG="${LOG_DIR}/tunnel.log"
BACKEND_PORT="${PORT:-8000}"

# Colors
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[0;33m"
BOLD="\033[1m"
RESET="\033[0m"

if ! command -v cloudflared >/dev/null 2>&1; then
    echo -e "${YELLOW}cloudflared is not installed. Installing via Homebrew...${RESET}"
    brew install cloudflared
fi

echo -e "${CYAN}${BOLD}[INFO] Starting Cloudflare Tunnel for http://localhost:${BACKEND_PORT}...${RESET}"

if [[ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
    echo -e "${GREEN}[INFO] Running with persistent Tunnel Token...${RESET}"
    exec cloudflared tunnel run --token "${CLOUDFLARE_TUNNEL_TOKEN}"
else
    echo -e "${YELLOW}[INFO] Running Quick Tunnel (trycloudflare.com)...${RESET}"
    cloudflared tunnel --url "http://localhost:${BACKEND_PORT}" 2>&1 | tee "${TUNNEL_LOG}"
fi
