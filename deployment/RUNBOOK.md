# CystoDS Clinical AI Platform — Production Runbook & Operational Manual

**Target**: Production, Staging, and Local Demonstration Environments  
**System**: CystoDS Research Platform (Hierarchical Long-Tailed Endoscopy Classification)  
**Standard**: Conforms to `Docs/CystoDS_Website_Detailed_Implementation_Plan.md` (§11, §12, §13)  
**Release Tag**: `v1.0.0-tuan4-hardening`  
**Last Updated**: 2026-11-06  

---

## 1. System Architecture Overview

The CystoDS Research Platform is an enterprise clinical AI web platform engineered to deploy, demonstrate, and scientifically explain a 3-Stage Hierarchical Fine-Tuning (`3S-HFT`) neural network for bladder cancer diagnosis in white-light cystoscopy (WLC).

```
                         [ Clinical Client / Browser ]
                                      │
                         HTTPS (Port 443 / 3000)
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │       Next.js 14 Frontend Application           │
             │   - High-End Visuals (Docs/SKILL_FRONTEND.md)   │
             │   - Interactive Demo (/demo)                    │
             │   - Research Storytelling (/research)           │
             │   - Results & Scientific Provenance (/results)  │
             └────────────────────────┬────────────────────────┘
                                      │ REST API / CORS
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │        FastAPI Inference Engine (:8000)         │
             │   - /api/v1/health (readiness & telemetry)      │
             │   - /api/v1/model-info (architecture & metrics) │
             │   - /api/v1/samples (curated clinical cases)    │
             │   - /api/v1/predict (genuine hierarchical runs) │
             └────────────────────────┬────────────────────────┘
                                      │ PyTorch Native Tensor API
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │     TwoStageDecoupledHierarchicalModel          │
             │   - Swin-Tiny Backbone (28,230,679 params)      │
             │   - Level 1: Binary (Non-ROI / ROI)             │
             │   - Level 2: Coarse (5 Anatomical Groups)       │
             │   - Level 3: Fine (22 Clinical Subclasses)      │
             │   - Checkpoint: SHA-256 (6c10366505db...)       │
             └─────────────────────────────────────────────────┘
```

---

## 2. Quick Start & Execution Modes

### Mode A: One-Click Local Runner (Local Dev & Rehearsal)
The platform includes an automated fallback runner script that checks all system prerequisites, validates the cryptographic model hash, and launches both services:

```bash
# Execute local demo runner with automated readiness polling
./deployment/run_local_demo.sh
```

**Available Flags**:
- `./deployment/run_local_demo.sh --check-only`: Verifies dependencies and assets without spawning background daemons.
- `./deployment/run_local_demo.sh --dry-run`: Prints exact execution commands without running them.
- `./deployment/run_local_demo.sh --offline-verify`: Executes deep asset validation for offline conference halls.
- `./deployment/run_local_demo.sh --port-backend 8000 --port-frontend 3000`: Configures custom networking ports.

### Mode B: Offline Venue / Rehearsal Mode
For high-stakes medical conferences, university auditoriums, or clinical settings with limited or restricted internet connectivity:
1. Ensure the canonical checkpoint is present at:
   `result/30_proposed/three_stage_split0_research_20260819-211511/phase3/split_0/best_model.pt`
2. Run the offline attestation tool:
   ```bash
   python3 deployment/verify_offline_bundle.py
   ```
   *Expected output*: `[SUCCESS] ALL GATE REQUIREMENTS PASSED. Platform is 100% ready for offline venue presentation.`
3. Start the local server:
   ```bash
   ./deployment/run_local_demo.sh
   ```
   Both frontend and backend operate with zero external API calls or internet dependencies.

### Mode C: Staging & Production Containerized Deployment
For production servers with reverse proxies (e.g. Nginx, Traefik, Caddy) and SSL/TLS termination:

```bash
# 1. Start FastAPI Backend with Uvicorn (Single Worker per GPU / Worker Process)
uvicorn backend.app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --proxy-headers \
  --forwarded-allow-ips="*"

# 2. Build and Start Next.js Frontend
cd frontend
npm run build
npm run start -- -p 3000
```

---

## 3. Environment Variables Specification

The system is configured via environment variables with zero code modification needed. Templates are provided in `.env.example`, `backend/.env.example`, and `frontend/.env.example`.

| Variable | Scope | Type | Default Value | Description |
|---|---|---|---|---|
| `APP_ENV` | Both | string | `production` | Deployment mode: `production`, `staging`, `development`, `offline`. |
| `HOST` | Backend | string | `0.0.0.0` | Network binding interface for FastAPI. |
| `PORT` | Backend | int | `8000` | Port for FastAPI HTTP server. |
| `ALLOWED_ORIGINS` | Backend | string | `*` (or comma-separated) | CORS allowed origins (e.g. `http://localhost:3000,https://cystods.ai`). |
| `MODEL_PATH` | Backend | string | `result/.../best_model.pt` | Path to the canonical PyTorch model weights file. |
| `MODEL_HASH` | Backend | string | `6c10366505db8036...` | Expected SHA-256 hash. Server verifies integrity on startup. |
| `DEVICE` | Backend | string | `auto` | Compute device: `auto` (detects CUDA -> MPS -> CPU), `cuda`, `mps`, or `cpu`. |
| `MAX_UPLOAD_BYTES` | Backend | int | `10485760` (10 MB) | Strict upper bound for file uploads (rejects >10MB with HTTP 413). |
| `MAX_CONCURRENT_INFERENCE`| Backend | int | `2` | Maximum concurrent forward passes allowed before queueing. |
| `REQUEST_TIMEOUT_SECONDS` | Backend | int | `30` | Maximum timeout before inference worker returns 504 Gateway Timeout. |
| `DEMO_SAMPLES_DIR` | Backend | string | `xvdhy-osfstorage-archive/images` | Storage directory for verified clinical reference cases. |
| `NEXT_PUBLIC_API_URL` | Frontend | string | `http://localhost:8000` | Base URL used by browser and Next.js SSR to reach backend API. |
| `PORT_FRONTEND` | Frontend | int | `3000` | Port for Next.js web application. |

---

## 4. Operational Procedures & Lifecycle

### 4.1 Checkpoint Warmup & Singleton Lifecycle
The backend leverages an async lifespan manager (`backend/app/main.py:lifespan`).
- The neural network checkpoint is loaded into memory **exactly once** during application bootstrap.
- The service computes and verifies the SHA-256 hash immediately upon loading.
- If the checkpoint file is corrupted or mismatched, the service flags degraded status in `/api/v1/health` and refuses prediction traffic with HTTP 503 rather than serving corrupt inferences.
- Health checks return `model_loaded: true` only after full tensor initialization and memory allocation are confirmed.

### 4.2 Telemetry & Privacy Guardrails
1. **Zero Input Retention**:
   - Uploaded endoscopic frames are processed entirely in volatile memory (`raw_bytes` -> `io.BytesIO` -> `PIL.Image`).
   - Images are **never** written to temporary directories (`/tmp`), disk buffers, or third-party loggers.
2. **Sanitized Error Payloads**:
   - HTTP 400/413/422 responses provide actionable error codes (`status: error`, `type: ValueError`).
   - Python stack traces, internal paths (`/Volumes/...`), and raw payload byte dumps are strictly suppressed from client-facing responses.
3. **Audit Metrics**:
   - Responses track `inference_time_ms`, compute hardware (`mps`/`cuda`/`cpu`), and request UUID (`inf_...`) for audit traceability.

---

## 5. Incident Response Playbook

### Playbook 1: GPU / MPS Out-Of-Memory (OOM)
- **Symptom**: Request returns HTTP 500, or worker process crashes with `CUDA out of memory` or `MPS backend out of memory`.
- **Root Cause**: High-resolution image allocation or concurrent inference requests exhausting hardware VRAM.
- **Triage & Remediation**:
  1. Inspect allocated memory:
     ```bash
     curl -s http://localhost:8000/api/v1/health | grep memory_allocated_mb
     ```
  2. Set `DEVICE=cpu` in `.env` to immediately restore service without hardware memory limits:
     ```bash
     export DEVICE=cpu
     ./deployment/run_local_demo.sh
     ```
  3. Ensure `MAX_CONCURRENT_INFERENCE=1` to enforce strict sequential forward passes.

### Playbook 2: Checkpoint Integrity / Hash Mismatch
- **Symptom**: Service reports `status: degraded`, `model_loaded: false`, or refuses to boot.
- **Root Cause**: Checkpoint file was truncated, replaced, or modified during deployment.
- **Triage & Remediation**:
  1. Run SHA-256 check manually:
     ```bash
     shasum -a 256 result/30_proposed/three_stage_split0_research_20260819-211511/phase3/split_0/best_model.pt
     ```
  2. Compare with canonical hash: `6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c`.
  3. If mismatched, restore the verified artifact from backup archive:
     ```bash
     unzip -o baldder_tissue_classification.zip
     ```

### Playbook 3: Upload Abuse & Malicious Payloads
- **Symptom**: Spikes in HTTP 400 or HTTP 413 responses; suspicious filenames or binary executables uploaded.
- **Root Cause**: Client attempting path traversal, extension spoofing, or oversized payload denial-of-service.
- **Triage & Remediation**:
  1. The platform automatically blocks:
     - Oversized uploads (>10MB) via `MAX_UPLOAD_BYTES` (returns HTTP 413).
     - Extension spoofing (non-image files with `.png` or `.jpg` extension return HTTP 400).
     - Path traversal (`../../etc/passwd` or `..\..\cmd.exe` return HTTP 400).
     - Zero-byte files (return HTTP 400).
  2. If an IP continues spamming malformed requests, add an IP rate limit or firewall rule in Nginx:
     ```nginx
     limit_req_zone $binary_remote_addr zone=inference_limit:10m rate=5r/s;
     ```

### Playbook 4: Frontend-Backend Disconnection
- **Symptom**: Web UI shows "Backend API Disconnected" toast or fallback demo indicator.
- **Root Cause**: FastAPI service terminated or blocked by port collision.
- **Triage & Remediation**:
  1. Verify backend port:
     ```bash
     curl -i http://localhost:8000/api/v1/health
     ```
  2. Check if another application stole port 8000:
     ```bash
     lsof -i :8000
     ```
  3. Restart via `./deployment/run_local_demo.sh` to trigger automated port conflict resolution.

---

## 6. Release Gates Verification Matrix (§12.3)

Before publishing any release or demonstrating before an evaluation board, all release gates must be empirically verified:

| Gate ID | Area | Criterion | Verification Evidence | Status |
|---|---|---|---|---|
| **GATE-ML-1** | Machine Learning | Canonical checkpoint cryptographic integrity | SHA-256 matches `6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c` | **PASS** |
| **GATE-ML-2** | Machine Learning | Reference numerical parity against research pipeline | `pytest backend/tests/test_parity.py` passes all 6 cases with tolerance $< 10^{-4}$ | **PASS** |
| **GATE-API** | Backend Integration | Stable API schemas and structured error contracts | 37 tests in `backend/tests/test_e2e_and_security.py` pass cleanly | **PASS** |
| **GATE-UX** | Frontend Experience | Responsive layout, zero mobile overflow, Agency UI | Next.js compiles 10/10 static pages, ESLint 0 warnings/errors, strict Doppelrand design | **PASS** |
| **GATE-SEC** | Security Hardening | File size limits, extension spoofing, path traversal | Automated penetration tests verify 400/413 rejection; zero traceback leaks | **PASS** |
| **GATE-SCI** | Scientific Rigor | Transparent metrics, clinical error boundaries | Results dashboard backed by holdout commit `d8a92f0`; failure cases documented | **PASS** |
| **GATE-OPS** | Operations | Health monitoring, one-click launcher, offline mode | `deployment/run_local_demo.sh` and `verify_offline_bundle.py` execute cleanly | **PASS** |
| **GATE-DEMO** | Presentation | Demo policy compliance, live vs precomputed labeling | Rehearsal scripts approved; UI explicitly separates live inference from precomputed gallery | **PASS** |

---

## 7. Live vs Precomputed Demo Policy (§13.4)

To uphold the highest medical and academic standards, the platform enforces strict presentation honesty:

1. **Live Mode**:
   - Activated when a clinician or presenter uploads a genuine cystoscopy frame or triggers a sample inference through the API.
   - The UI displays the live inference latency (e.g. `184 ms`), active hardware device (`mps`/`cuda`/`cpu`), and canonical checkpoint hash.
   - The label reads: `● LIVE MODEL INFERENCE`.

2. **Precomputed Gallery Mode**:
   - Used when browsing reference scorecards, historical validation benchmarks, or when backend connectivity is intentionally disabled.
   - Any static result card must display the explicit disclaimer:
     `Precomputed Reference Sample (Holdout Cohort Split 0)`.

3. **Strict Academic Prohibition**:
   - **Under no circumstances** may the platform present cached or hardcoded predictions as the live inference result of a newly uploaded user image.
   - Every uploaded image must undergo true forward computation through `TwoStageDecoupledHierarchicalModel`.
   - If the backend is unavailable during an image upload, the platform must report an explicit connection error rather than fabricating an artificial prediction.

---

## 8. Summary of Verification Commands

```bash
# 1. Run Complete Backend Test Pyramid (Unit + Parity + Integration + Security E2E)
python3 -m pytest backend/tests/ -v

# 2. Verify Offline Presentation Assets
python3 deployment/verify_offline_bundle.py

# 3. Test Local Demo Launcher Preflight
./deployment/run_local_demo.sh --check-only

# 4. Verify Next.js Production Build
cd frontend && npm run build

# 5. Verify Frontend ESLint Compliance
cd frontend && npm run lint
```
