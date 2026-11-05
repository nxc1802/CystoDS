#!/usr/bin/env bash
# ==============================================================================
# CystoDS Clinical AI Research Platform — Local Fallback & Demo Runner
# Conforms to Docs/CystoDS_Website_Detailed_Implementation_Plan.md §13
# ==============================================================================

set -euo pipefail

# ── Styling & Color Palette ───────────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
MAGENTA="\033[0;35m"
RESET="\033[0m"

log_info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${RESET} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
log_error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
log_banner()  {
    echo -e "${MAGENTA}${BOLD}======================================================================${RESET}"
    echo -e "${MAGENTA}${BOLD}   CystoDS Local Fallback Runner & Verification System (Tuần 4)       ${RESET}"
    echo -e "${MAGENTA}${BOLD}======================================================================${RESET}"
}

# ── Paths & Directory Resolution ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

CANONICAL_CHECKPOINT_REL="result/30_proposed/three_stage_split0_research_20260819-211511/phase3/split_0/best_model.pt"
CANONICAL_CHECKPOINT="${REPO_ROOT}/${CANONICAL_CHECKPOINT_REL}"
EXPECTED_HASH="6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c"

BACKEND_PORT="${PORT:-8000}"
FRONTEND_PORT="${PORT_FRONTEND:-3000}"
BACKEND_HOST="0.0.0.0"

CHECK_ONLY=false
DRY_RUN=false
OFFLINE_VERIFY=false

# ── CLI Parameter Parsing ─────────────────────────────────────────────────────
print_usage() {
    echo "Usage: ./deployment/run_local_demo.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --check-only         Verify dependencies, checkpoints and assets without starting servers."
    echo "  --dry-run            Simulate execution and print commands without launching processes."
    echo "  --offline-verify     Run full offline asset verification script and gate check."
    echo "  --port-backend PORT  Set FastAPI backend port (default: 8000)."
    echo "  --port-frontend PORT Set Next.js frontend port (default: 3000)."
    echo "  -h, --help           Show this help message and exit."
    echo ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --offline-verify)
            OFFLINE_VERIFY=true
            shift
            ;;
        --port-backend)
            BACKEND_PORT="$2"
            shift 2
            ;;
        --port-frontend)
            FRONTEND_PORT="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ── Signal Trapping & Process Cleanup ──────────────────────────────────────────
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    log_info "Initiating graceful shutdown of background services..."
    if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
        log_info "Stopping FastAPI backend (PID: ${BACKEND_PID})..."
        kill -SIGTERM "${BACKEND_PID}" 2>/dev/null || true
    fi
    if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
        log_info "Stopping Next.js frontend (PID: ${FRONTEND_PID})..."
        kill -SIGTERM "${FRONTEND_PID}" 2>/dev/null || true
    fi
    log_success "Shutdown complete. Resources freed."
}
trap cleanup INT TERM EXIT

log_banner

# ── Step 1: Python Environment Check ──────────────────────────────────────────
log_info "Step 1: Checking Python environment..."
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    log_error "Python 3 is required but not found on PATH."
    exit 1
fi

PY_VERSION=$("${PYTHON_BIN}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
log_success "Found Python ${PY_VERSION} (${PYTHON_BIN})"

# Verify essential Python packages
"${PYTHON_BIN}" -c "
import torch, torchvision, fastapi, uvicorn, PIL, pydantic
" 2>/dev/null || {
    log_error "Missing required Python dependencies (torch, torchvision, fastapi, uvicorn, PIL, pydantic)."
    log_info "Please install requirements: pip install -e . (or pip install fastapi uvicorn torch torchvision pillow pydantic)"
    exit 1
}
log_success "Python dependencies verified (torch, torchvision, fastapi, uvicorn, PIL, pydantic)."

# ── Step 2: Node.js & npm Check ───────────────────────────────────────────────
log_info "Step 2: Checking Node.js and npm environment..."
if ! command -v node >/dev/null 2>&1; then
    log_error "Node.js is not installed or not on PATH."
    exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
    log_error "npm is not installed or not on PATH."
    exit 1
fi
NODE_VERSION=$(node -v)
NPM_VERSION=$(npm -v)
log_success "Found Node.js ${NODE_VERSION} and npm ${NPM_VERSION}."

# ── Step 3: Model Checkpoint Verification ─────────────────────────────────────
log_info "Step 3: Verifying canonical deep learning checkpoint..."
if [[ ! -f "${CANONICAL_CHECKPOINT}" ]]; then
    log_error "Canonical checkpoint not found at: ${CANONICAL_CHECKPOINT}"
    log_error "Please ensure checkpoint exists before running demo."
    exit 1
fi

CHECKPOINT_SIZE_MB=$("${PYTHON_BIN}" -c "import os; print(round(os.path.getsize('${CANONICAL_CHECKPOINT}') / (1024*1024), 1))")
log_info "Checkpoint file size: ${CHECKPOINT_SIZE_MB} MB"

CALCULATED_HASH=$("${PYTHON_BIN}" -c "
import hashlib
sha = hashlib.sha256()
with open('${CANONICAL_CHECKPOINT}', 'rb') as f:
    while chunk := f.read(1024 * 1024):
        sha.update(chunk)
print(sha.hexdigest())
")

if [[ "${CALCULATED_HASH}" != "${EXPECTED_HASH}" ]]; then
    log_error "Checkpoint SHA-256 mismatch!"
    log_error "Expected: ${EXPECTED_HASH}"
    log_error "Actual:   ${CALCULATED_HASH}"
    exit 1
fi
log_success "Checkpoint integrity cryptographically verified: ${CALCULATED_HASH:0:16}..."

# ── Step 4: Curated Clinical Reference Samples Check ──────────────────────────
log_info "Step 4: Checking curated clinical endoscopy reference cases..."
SAMPLES_DIR="${REPO_ROOT}/xvdhy-osfstorage-archive/images"
REQUIRED_SAMPLES=("01c5cce3.png" "0e1d3230.png" "d326d133.png" "004d25e9.png" "23a59431.png" "489f6a5e.png")

for sample in "${REQUIRED_SAMPLES[@]}"; do
    if [[ ! -f "${SAMPLES_DIR}/${sample}" ]]; then
        log_error "Missing curated sample file: ${SAMPLES_DIR}/${sample}"
        exit 1
    fi
done
log_success "All 6 curated clinical endoscopy reference samples verified."

# Execute offline verification utility if requested
if [[ "${OFFLINE_VERIFY}" == true ]]; then
    log_info "Running dedicated offline bundle verification utility..."
    "${PYTHON_BIN}" "${REPO_ROOT}/deployment/verify_offline_bundle.py"
    log_success "Offline bundle verification passed."
fi

# Exit early if check-only flag was passed
if [[ "${CHECK_ONLY}" == true ]]; then
    log_success "All pre-flight checks passed successfully. System is demo-ready."
    exit 0
fi

if [[ "${DRY_RUN}" == true ]]; then
    log_info "[DRY-RUN] Startup commands that would be executed:"
    echo "  Backend:  cd ${REPO_ROOT} && ${PYTHON_BIN} -m uvicorn backend.app.main:app --host ${BACKEND_HOST} --port ${BACKEND_PORT}"
    echo "  Frontend: cd ${REPO_ROOT}/frontend && npm run start -- -p ${FRONTEND_PORT}"
    log_success "[DRY-RUN] Simulation finished cleanly."
    exit 0
fi

# ── Step 5: Check Port Availability ───────────────────────────────────────────
log_info "Step 5: Verifying network ports (${BACKEND_PORT}, ${FRONTEND_PORT})..."
check_port() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
            log_error "Port ${port} is already in use by another process!"
            return 1
        fi
    fi
    return 0
}

check_port "${BACKEND_PORT}" || exit 1
check_port "${FRONTEND_PORT}" || exit 1
log_success "Ports ${BACKEND_PORT} and ${FRONTEND_PORT} are available."

# ── Step 6: Launch FastAPI Backend Service ─────────────────────────────────────
log_info "Step 6: Starting FastAPI Backend on ${BACKEND_HOST}:${BACKEND_PORT}..."
cd "${REPO_ROOT}"

# Set default operational environment variables
export APP_ENV="${APP_ENV:-production}"
export MODEL_PATH="${CANONICAL_CHECKPOINT}"
export MODEL_HASH="${EXPECTED_HASH}"
export PORT="${BACKEND_PORT}"
export HOST="${BACKEND_HOST}"
export MAX_UPLOAD_BYTES="${MAX_UPLOAD_BYTES:-10485760}"

"${PYTHON_BIN}" -m uvicorn backend.app.main:app \
    --host "${BACKEND_HOST}" \
    --port "${BACKEND_PORT}" \
    > "${LOG_DIR}/backend.log" 2>&1 &
BACKEND_PID=$!

log_info "Backend process spawned with PID: ${BACKEND_PID} (Logs: logs/backend.log)"

# ── Step 7: Health Readiness Polling ──────────────────────────────────────────
log_info "Step 7: Polling backend health endpoint (http://localhost:${BACKEND_PORT}/api/v1/health)..."
HEALTH_URL="http://localhost:${BACKEND_PORT}/api/v1/health"
MAX_ATTEMPTS=30
ATTEMPT=1
IS_HEALTHY=false

while [[ ${ATTEMPT} -le ${MAX_ATTEMPTS} ]]; do
    if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
        log_error "Backend process died unexpectedly. Check logs/backend.log for details."
        exit 1
    fi

    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${HEALTH_URL}" 2>/dev/null || echo "000")
    if [[ "${HTTP_STATUS}" == "200" ]]; then
        HEALTH_BODY=$(curl -s "${HEALTH_URL}")
        if echo "${HEALTH_BODY}" | grep -q '"model_loaded":true'; then
            IS_HEALTHY=true
            break
        fi
    fi

    echo -ne "."
    sleep 1
    ATTEMPT=$((ATTEMPT + 1))
done
echo ""

if [[ "${IS_HEALTHY}" != true ]]; then
    log_error "Backend failed to report healthy status within ${MAX_ATTEMPTS} seconds."
    log_error "Last 30 lines of backend.log:"
    tail -n 30 "${LOG_DIR}/backend.log" >&2
    exit 1
fi
log_success "FastAPI Backend is ONLINE and MODEL IS LOADED!"

# ── Step 8: Launch Next.js Frontend Application ───────────────────────────────
log_info "Step 8: Starting Next.js Frontend on port ${FRONTEND_PORT}..."
cd "${REPO_ROOT}/frontend"

# Build if .next directory does not exist
if [[ ! -d ".next" ]]; then
    log_info "Building production Next.js frontend bundle..."
    npm run build
fi

export NEXT_PUBLIC_API_URL="http://localhost:${BACKEND_PORT}"

# Use npm run start for production or fallback to dev if standalone start is not configured
if grep -q '"start"' package.json; then
    npm run start -- -p "${FRONTEND_PORT}" > "${LOG_DIR}/frontend.log" 2>&1 &
    FRONTEND_PID=$!
else
    npm run dev -- -p "${FRONTEND_PORT}" > "${LOG_DIR}/frontend.log" 2>&1 &
    FRONTEND_PID=$!
fi

log_info "Frontend process spawned with PID: ${FRONTEND_PID} (Logs: logs/frontend.log)"

# ── Step 9: Ready Banner & Service URLs ───────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}======================================================================${RESET}"
echo -e "${GREEN}${BOLD}   CystoDS Research Platform is Running Successfully!                ${RESET}"
echo -e "${GREEN}${BOLD}======================================================================${RESET}"
echo -e "  ${BOLD}Interactive Web UI:${RESET}       ${CYAN}http://localhost:${FRONTEND_PORT}${RESET}"
echo -e "  ${BOLD}Interactive Live Demo:${RESET}    ${CYAN}http://localhost:${FRONTEND_PORT}/demo${RESET}"
echo -e "  ${BOLD}Research Storytelling:${RESET}    ${CYAN}http://localhost:${FRONTEND_PORT}/research${RESET}"
echo -e "  ${BOLD}Quantitative Results:${RESET}     ${CYAN}http://localhost:${FRONTEND_PORT}/results${RESET}"
echo -e "  ${BOLD}Backend OpenAPI Docs:${RESET}     ${CYAN}http://localhost:${BACKEND_PORT}/docs${RESET}"
echo -e "  ${BOLD}Backend Health Endpoint:${RESET}  ${CYAN}http://localhost:${BACKEND_PORT}/api/v1/health${RESET}"
echo -e "  ${BOLD}Backend Service Logs:${RESET}     ${CYAN}${LOG_DIR}/backend.log${RESET}"
echo -e "  ${BOLD}Frontend Service Logs:${RESET}    ${CYAN}${LOG_DIR}/frontend.log${RESET}"
echo -e "${GREEN}${BOLD}======================================================================${RESET}"
echo -e "Press ${BOLD}Ctrl+C${RESET} at any time to gracefully terminate both services."
echo ""

# Wait for children to run until user stops with Ctrl+C
wait "${BACKEND_PID}" "${FRONTEND_PID}"
