# Wildlife Camera Trap Auto-Analyzer (ViumbeLens)

A **FastAPI + React platform** for automated analysis of wildlife camera trap images, designed for the **Gambella Wetland Landscape Baseline Survey** and similar conservation programmes.

The system runs a multi-stage AI pipeline — OCR metadata extraction, animal detection (**MegaDetector v5a**), species identification (**SpeciesNet**), day/night classification, QC flagging, privacy scrubbing, and spatial export — through a modern browser dashboard with real-time per-model output streaming (SSE).

**SpeciesNet-First mode** is the default classifier path. See [SPECIESNET_FIRST_CONFIG.md](SPECIESNET_FIRST_CONFIG.md) for configuration options.

![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)
![React](https://img.shields.io/badge/React-19-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![MegaDetector](https://img.shields.io/badge/MegaDetector-v5a-blue)
![SpeciesNet](https://img.shields.io/badge/SpeciesNet-Google-teal)
![Python](https://img.shields.io/badge/Python-3.11--3.12-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

---

## Table of Contents

1. [Running with Docker (Recommended)](#running-with-docker-recommended)
2. [Local Development](#local-development)
3. [User Sessions & Multi-User](#user-sessions--multi-user)
4. [Storage & Manual Cleanup](#storage--manual-cleanup)
5. [AI Models](#ai-models--detectors--classifiers-explained)
6. [Features](#features)
7. [Architecture](#architecture)
8. [Configuration](#configuration)
9. [Performance & System Requirements](#performance--system-requirements)
10. [Installation Options (Non-Docker)](#installation-options)
11. [Troubleshooting](#troubleshooting)
12. [Recent Changes](#recent-changes)

---

## Running with Docker (Recommended)

Docker runs the **production build** — React frontend and FastAPI API served together on **port 8000**. This is the recommended way to deploy on a field laptop, lab machine, or small server.

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Docker Engine** 24+ | With Compose V2 (`docker compose`, not legacy `docker-compose`) |
| **RAM** | 8 GB minimum; **16 GB recommended** for full model load |
| **Disk** | ~10 GB free for image layers + model caches on first run |
| **Git** | To clone the repository |

Verify Docker works:

```bash
docker compose version
docker info
```

### First-time setup

```bash
git clone <repository-url>
cd camera-traps

# Optional but recommended — copy env template
cp .env.example .env
```

Edit `.env` if needed (see [Environment variables](#environment-variables-docker) below). At minimum, add Kaggle credentials for SpeciesNet:

```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
SESSION_SECRET=change-me-to-a-long-random-string
```

### Build and start

Always use **`./compose.sh`** instead of plain `docker compose`. It enables BuildKit, pip/npm cache mounts, and faster rebuilds automatically:

```bash
./compose.sh up --build
```

| Command | When to use |
|---------|-------------|
| `./compose.sh up --build` | First run, or after code/dependency changes |
| `./compose.sh up` | Normal start (uses existing image) |
| `./compose.sh up -d` | Start detached (background) |
| `./compose.sh logs -f` | Follow container logs |
| `./compose.sh down` | Stop and remove container |
| `./compose.sh build` | Rebuild image only (no start) |

**Open the app:** [http://localhost:8000](http://localhost:8000)

**API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

On first visit you will be prompted for your **name** (username-only session — no password). This name is attached to review actions and retrain runs.

### Build times (what to expect)

| Build type | Typical duration | Why |
|------------|------------------|-----|
| **First full build** | 15–25 minutes | Downloads PyTorch, EasyOCR, MegaDetector, SpeciesNet deps; exports ~2 GB+ layers |
| **Rebuild after code change** | 1–3 minutes | Source layers rebuild; pip/npm layers cached |
| **Rebuild after `requirements.txt` change** | 8–15 minutes | pip layer invalidates |

The `.dockerignore` file keeps build context small (~10 KB instead of hundreds of MB). BuildKit cache mounts in the `Dockerfile` speed up repeat pip/npm installs.

> **Tip:** Use `bash dev.sh` for active development to avoid Docker rebuilds. Use Docker when you want a production-like single-server deployment.

### What Docker mounts (your data persists)

| Host path / volume | Container path | Purpose |
|--------------------|----------------|---------|
| `./wildlife_data.db` | `/app/data/wildlife_data.db` | SQLite database (detections, review log, projects) |
| `./uploads/` | `/app/data/uploads/` | Uploaded camera trap images |
| `/tmp/megadetector_models/` | `/tmp/megadetector_models/` | MegaDetector weights (host cache) |
| `easyocr_cache` (named volume) | `/root/.EasyOCR` | EasyOCR model cache |
| `huggingface_cache` (named volume) | `/root/.cache/huggingface` | Hugging Face / related caches |
| `kaggle_cache` (named volume) | `/root/.cache/kagglehub` | SpeciesNet weights (Kaggle) — avoids re-download on every restart |

The database and uploads live on your **host filesystem** — stopping or rebuilding the container does **not** delete your data.

**First run without an existing database:** Ensure `wildlife_data.db` exists as a **file** on the host before starting (Docker can create a directory instead of a file if the path is missing). Copy a blank DB or let the app create one via local dev first, then mount it.

### Environment variables (Docker)

Set these in `.env` (loaded automatically by `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `/app/data/wildlife_data.db` | Set by compose — usually leave as-is |
| `KAGGLE_USERNAME` | — | Required for SpeciesNet model download |
| `KAGGLE_KEY` | — | Kaggle API key |
| `SESSION_SECRET` | dev default | **Change in production** — signs session cookies |
| `MAX_UPLOAD_MB` | `50` | Max single-file upload size |
| `JOB_TTL_HOURS` | `2` | Hours before finished in-memory jobs are evicted |
| `CORS_ORIGINS` | localhost Vite URLs | Only needed if accessing API from another origin |

### Container health

Docker waits up to **300 seconds** on startup for core models. SpeciesNet may continue downloading in the background — the API is reachable while it loads. Poll `/api/config/status` until `speciesnet.loaded` is true.

Check manually:

```bash
curl http://localhost:8000/api/config/status
# {"models_loaded": true, "error": null}
```

### Docker troubleshooting

**Container exits immediately — `ModuleNotFoundError: itsdangerous`**

Session middleware requires `itsdangerous`. Rebuild after pulling latest code:

```bash
./compose.sh build --no-cache
./compose.sh up
```

**Port 8000 already in use**

```bash
lsof -ti :8000 | xargs kill
./compose.sh up
```

Or change the port in `docker-compose.yml`: `"8080:8000"` → open http://localhost:8080

**Permission denied on Docker socket**

```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Out of memory (OOM) during model load**

The compose file limits memory to **4 GB**; full model load can need **~4.5 GB**. Either:

- Increase `deploy.resources.limits.memory` in `docker-compose.yml` to `6G` or `8G`, or
- Enable **Low-Spec Mode** in the sidebar (disables SpeciesNet, reduces RAM)

**SpeciesNet not loading**

Add valid `KAGGLE_USERNAME` and `KAGGLE_KEY` to `.env`, then restart:

```bash
./compose.sh down && ./compose.sh up
```

Detection still works without SpeciesNet; species names will be missing.

**Slow rebuild every time**

Use `./compose.sh` (not bare `docker compose`) so BuildKit caches apply. Only run `--build` when code or dependencies changed.

---

## Local Development

For day-to-day coding with hot reload (no Docker rebuilds):

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r backend/requirements.txt
cd frontend && npm install && cd ..
cp .env.example .env   # optional

bash dev.sh
```

| Service | URL |
|---------|-----|
| Frontend (Vite + HMR) | http://localhost:5173 |
| API + Swagger | http://localhost:8000/docs |

Vite proxies `/api/*` to the backend. Same username prompt on first visit.

---

## User Sessions & Multi-User

There is **no password authentication**. The app uses **username-only sessions** for attribution on a trusted local/LAN network.

| Behaviour | Detail |
|-----------|--------|
| **First visit** | Modal asks for your name |
| **Session storage** | Signed HttpOnly cookie (`SESSION_SECRET`) |
| **Same browser, multiple tabs** | One shared session — same username everywhere |
| **Review locking** | Each `detection_id` can only be reviewed once (409 if duplicate) |
| **Switch user** | Sidebar → "Switch user" clears session and prompts again |
| **Review / retrain / community** | Actions automatically use session username |

**What sessions do not provide:** API access control. Anyone on the network can still call endpoints. Config thresholds remain shared server-wide (last sidebar change wins).

**Multiple computers:** Each machine gets its own session cookie. Two people can use the same username on different machines — attribution still works, but review races are possible until one completes an item.

---

## Storage & Manual Cleanup

Images are classified into tiers after processing:

| Tier | Meaning |
|------|---------|
| `empty` | No animal, or person/vehicle only, or zero confidence |
| `low_conf` | Animal detected, confidence ≤ 0.4 |
| `valid` | Animal detected, confidence > 0.4 |

**Cleanup is manual only** — there is no background scheduler. Run cleanup when you need to free disk space:

1. Open the **Storage Management** tab in the UI, or
2. Call the API:

```bash
# Preview deletions (dry run)
curl -X POST "http://localhost:8000/api/storage/cleanup?action=delete_empty&dry_run=true&days_old=30"

# Delete empty-tier files older than 30 days
curl -X POST "http://localhost:8000/api/storage/cleanup?action=delete_empty&dry_run=false&days_old=30"

# Delete files marked for deletion past grace period
curl -X POST "http://localhost:8000/api/storage/cleanup?action=delete_marked&dry_run=false&days_old=7"
```

**Database reset** (destructive): requires `confirm=true` and typing your session username if signed in:

```bash
curl -X POST "http://localhost:8000/api/storage/reset-db?confirm=true&confirm_username=YourName"
```

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) and [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) for more storage operations.

---

## AI Models — Detectors & Classifiers

> **Production stack (2026):** The running application uses **MegaDetector v5a** for detection and **SpeciesNet** for species classification, plus **EasyOCR** and a **Day/Night** classifier. BioCLIP and MegaDetector v1000 are **not loaded**.

### Detectors vs Classifiers

| | Detector | Classifier |
|---|---|---|
| **Question** | *Is there an animal here, and where?* | *What species is this animal?* |
| **Output** | Bounding box + coarse label (Animal / Person / Vehicle) | Species name + confidence |
| **Works on** | Full image | Cropped region of the detected animal |

The pipeline order is: **Detect → Classify crop → OCR / Day-Night → optional privacy scrub**.

---

### MegaDetector v5a (Detector)

**Source:** [agentmorris/MegaDetector](https://github.com/agentmorris/MegaDetector)  
**Install:** `pip install megadetector` (weights auto-download on first run, ~600 MB)

Scans each image and returns bounding boxes for **Animal**, **Person**, and **Vehicle**. Does not identify species.

**Confidence threshold:** Default 0.35 in the sidebar. Lower (0.15–0.25) if distant or dark animals are missed.

---

### SpeciesNet (Classifier)

**Source:** [google/cameratrapai](https://github.com/google/cameratrapai)  
**Install:** `pip install speciesnet` — requires **Kaggle credentials** for model download (~220 MB).

Classifies each animal crop from MegaDetector. Trained on 65M+ camera trap images; strong on blanks, night/IR, and common field species.

**Kaggle setup:**

```bash
# 1. Create a free account at https://www.kaggle.com/
# 2. Account → Settings → API → Create New Token
# 3. Add to .env:
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

If credentials are missing, SpeciesNet is skipped at startup — detection still works, but species names will be empty.

---

### Pipeline flow

```
Full image
  → MegaDetector v5a     (Animal / Person / Vehicle boxes)
  → SpeciesNet           (species ID on each animal crop)
  → EasyOCR              (timestamp / camera metadata strip)
  → Day/Night classifier
  → Privacy scrub        (optional — blur Person/Vehicle boxes)
  → SQLite + file store
```

**Output per detection:** species label, SpeciesNet confidence, agreement tier (High ≥ 0.7, Medium ≥ 0.4, Low otherwise), and `model_breakdown` with MDv5a + SpeciesNet scores only.

**Low-Spec mode:** Disables SpeciesNet to reduce RAM. Requires a **server restart** after toggling in the sidebar.

---

## Features

| Tab | Feature |
|-----|---------|
| Upload & Process | Auto-upload on file selection; real-time per-model output panel (OCR · MegaDetector · SpeciesNet · Day/Night); SSE progress stream |
| Review Results | Table and gallery views; bounding box overlays; multi-animal grouping; sortable columns; inline per-detection species editing; confidence bars; Excel/CSV export |
| Statistics | Per-species bar charts, day/night pie chart, confidence distribution |
| History | Long-term trends from SQLite database, CSV export |
| Diagnostics | OCR strip debugger, raw model output viewer |
| Ecological Analytics | IDE computation, RAI, species richness, accumulation curve, group size, visitation rate |
| QC Dashboard | Automated quality-control checks with colour-coded severity flags |
| Stations & Deployments | Camera registry, GPS coordinates, deployment history, trap-night calculator |
| Review Queue | Responsive card grid with bounding boxes; confirm / correct / flag per detection; session-based reviewer attribution |
| Community Observer | Field observer sighting entry, cross-verification against camera data |
| Spatial & Map | Interactive Leaflet map, GeoJSON / Shapefile / KML / CSV export |
| Species Library | 159 African wildlife species with full scientific names, IUCN status, synonym resolver |
| Corridor Analysis | Directional flow detection, passage frequency, bottleneck identification |
| Project Config | Multi-project support, indicator thresholds, baseline locking, JSON export |
| ArcGIS Sync | Offline file exports (GeoJSON, Shapefile, KML) + live push to ArcGIS Online / Enterprise |
| Storage Management | Tiered storage (empty/low_conf/valid); manual cleanup; hash deduplication; batch ZIP download |

---

## Architecture

```
camera-traps/
├── backend/                      # FastAPI application
│   ├── main.py                   # App factory, CORS, SessionMiddleware, lifespan model loading
│   ├── routers/                  # One router per feature tab
│   │   ├── session.py            # Username-only sessions (GET/POST/DELETE /api/session)
│   │   ├── config.py             # GET/PATCH /api/config
│   │   ├── images.py             # Upload, processing, SSE stream; MIME validation
│   │   ├── results.py            # Review, edit, export
│   │   ├── review.py             # Review queue — confirm/correct/flag by detection_id
│   │   ├── storage.py            # Storage metrics, manual cleanup, reset-db
│   │   └── …                     # statistics, history, ecological, spatial, etc.
│   ├── models/
│   │   ├── state.py              # AppState + AppConfig (runtime settings)
│   │   └── schemas.py            # Pydantic request/response models
│   └── services/
│       ├── file_manager.py       # Tier classification, ZIP downloads, manual cleanup
│       └── job_manager.py        # In-memory job tracker with model_events queue
│
├── frontend/                     # React + TypeScript + Vite application
│   ├── vite.config.ts            # Vite dev proxy: /api → localhost:8000
│   └── src/
│       ├── App.tsx               # Router + username modal on first visit
│       ├── components/UsernameModal.tsx
│       ├── store/sessionStore.ts # Session username state
│       ├── store/configStore.ts  # Zustand global config store
│       └── pages/                # 16 dashboard pages
│
├── core/                         # AI/ML business logic
│   ├── animal_detector.py        # MegaDetectorWrapper + AnimalDetector
│   ├── speciesnet_classifier.py  # Google SpeciesNet wrapper
│   ├── image_processor.py        # Unified OCR → detect → classify pipeline
│   ├── review_engine.py          # HITL review store (detection_id locking)
│   └── db_manager.py             # SQLite schema, WAL mode, one image row per file
│
├── compose.sh                      # Docker wrapper (BuildKit + caches) — use instead of docker compose
├── docker-compose.yml            # Container orchestration with named volumes
├── Dockerfile                    # Multi-stage build (Node → React → Python)
├── .dockerignore                 # Keeps build context small (~10 KB)
└── wildlife_data.db              # SQLite database (bind-mounted in Docker)
```

### Runtime data flow (Docker)

```
Browser → http://localhost:8000
              ↓
         FastAPI (serves React dist + /api/*)
              ↓
    core/image_processor.py  (OCR ∥ Day/Night → MegaDetector → SpeciesNet)
              ↓
    /app/data/wildlife_data.db  +  /app/data/uploads/
```

Only **one ML processing job** runs inference at a time (semaphore). Multiple users can browse and review concurrently; SQLite WAL mode supports parallel reads.

---

## Quick Start (Local Install — Non-Docker)

> **Prefer Docker?** See [Running with Docker (Recommended)](#running-with-docker-recommended) above.

### Step 1 — Install Python dependencies

> **Python version:** Python 3.12 is strongly recommended. Python 3.14+ is **not supported** — key packages (`megadetector`, `yolov5`) have no pre-built wheels for 3.14.

**Option A — Use the installer (recommended)**

```bash
# macOS / Linux
chmod +x install.sh && ./install.sh

# Windows
install.bat
```

**Option B — Manual setup**

```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -r backend/requirements.txt
python force_download.py          # pre-downloads MDv5a + SpeciesNet weights
```

### Step 2 — Set up SpeciesNet credentials (optional but recommended)

SpeciesNet downloads its weights from Kaggle on first run. Without credentials, SpeciesNet is skipped — MegaDetector detection still works, but species classification will be unavailable.

```bash
# 1. Create a free account at https://www.kaggle.com/
# 2. Go to: Account → Settings → API → Create New Token
# 3. Copy your username and key into .env:
cp .env.example .env
# Edit .env and add:
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_api_key
```

### Step 3 — Install frontend dependencies

```bash
cd frontend && npm install && cd ..
```

### Step 4 — Run

```bash
bash dev.sh
```

Open **http://localhost:5173** in your browser.

- Frontend (React) → `http://localhost:5173`
- API docs (Swagger) → `http://localhost:8000/docs`

---

## Installation Options

> **Recommended:** [Running with Docker](#running-with-docker-recommended) via `./compose.sh up --build`

### Option A — Docker (production)

See the full [Running with Docker](#running-with-docker-recommended) section for prerequisites, volumes, env vars, build times, and troubleshooting.

Quick reference:

```bash
cp .env.example .env
./compose.sh up --build    # first time
./compose.sh up            # subsequent runs
```

Open **http://localhost:8000**

---

### Option B — macOS / Linux (local dev, one-click)

```bash
git clone <repository-url>
cd camera-traps
chmod +x install.sh && ./install.sh
cd frontend && npm install && cd ..
bash dev.sh
```

For a clean reinstall: `./install.sh --fresh`

The Linux installer automatically:
- Installs `python3.12-venv` via apt (needed on Ubuntu 25.04+ where system Python is 3.14)
- Detects NVIDIA GPU via `nvidia-smi` — installs CPU-only PyTorch (~300 MB) if no GPU found
- Downloads packages in parallel with live progress, speed, ETA, and auto-retry

---

### Option C — Windows (local dev, one-click)

**Requirements:** Python 3.12 from [python.org](https://www.python.org/downloads/). Node.js from [nodejs.org](https://nodejs.org/).

```bat
git clone <repository-url>
cd camera-traps
install.bat
cd frontend && npm install && cd ..
bash dev.sh
```

> Python 3.14 does not need to be uninstalled — `install.bat` uses the Windows Python Launcher (`py -3.12`) to pick the right version automatically.

For a clean reinstall: `install.bat --fresh`

---

### Option D — Conda / Miniconda

```bash
git clone <repository-url>
cd camera-traps
conda env create -f environment.yml
conda activate wildlife-analyzer
pip install -r backend/requirements.txt
python force_download.py
cd frontend && npm install && cd ..
bash dev.sh
```

---

### Option E — VS Code Dev Container

1. Install the **Dev Containers** extension.
2. Open the repository and click **"Reopen in Container"**.
3. The container installs all Python and Node dependencies automatically.
4. Run `bash dev.sh` in the integrated terminal.

---

## How the Dev Setup Works

```
Browser → http://localhost:5173  →  Vite dev server (React, HMR)
                                         ↓  proxy /api/*
                                    FastAPI  (http://localhost:8000)
                                         ↓
                                    core/ (OCR + MegaDetector v5a + SpeciesNet + Day/Night)
                                         ↓
                                    wildlife_data.db (SQLite, WAL mode)
```

For production/Docker, the browser connects directly to **http://localhost:8000** (no separate Vite server).

---

## Configuration

All runtime settings are in the left sidebar (Config tab). Changes `PATCH /api/config` and take effect on the next processing run — no restart needed.

**User identity** is **not** in config — it comes from your [username session](#user-sessions--multi-user) (sidebar shows "Signed in as …").

| Setting | Description |
|---------|-------------|
| Detection Confidence | Score cutoff (default 0.35). Lower for dark/distant shots. |
| Brightness Threshold | Day/Night classification sensitivity (0–255). |
| Metadata Strip (%) | % of image bottom scanned for date/time OCR text. |
| Auto-Scrub Person/Vehicle | Apply Gaussian blur to privacy-sensitive bounding boxes. |
| Blur Strength | Gaussian kernel size (11–101, odd numbers). |
| Review Queue Threshold | Detections below this confidence appear in the Review Queue. |
| Independence Window (min) | Same species + station within this window = one IDE (default 30 min). |
| Default Station ID | Fallback when filename doesn't encode a station. |
| Default Trap Nights | Used for RAI when no deployment records exist. |
| Low-Spec Mode | Disables SpeciesNet; reduces RAM (~2 GB). MegaDetector + OCR still run. |
| CPU Threads | PyTorch intra-op threads (default ¼ of core count). |
| SpeciesNet geo prior | Lat/lng/country for SpeciesNet geographic filtering (default: Kenya). |

---

## Performance & System Requirements

### Model memory footprint

All models load once at startup and stay resident.

| Component | RAM | Disk | Notes |
|-----------|-----|------|-------|
| PyTorch runtime | ~1.2 GB | ~2 GB | Shared across all models |
| MegaDetector v5a | ~600 MB | ~600 MB | Always loaded |
| SpeciesNet | ~1.0 GB | ~220 MB | Skipped without Kaggle credentials or in Low-Spec Mode |
| EasyOCR | ~200 MB | ~200 MB | Always loaded |
| Day/Night classifier | negligible | — | CPU-only |
| **Full stack total** | **~3–4.5 GB** | **~4 GB** | Docker compose default limit is 4 GB |
| **Low-Spec total** | **~2 GB** | **~3 GB** | SpeciesNet disabled |

### Minimum recommended specs

| Resource | Low-Spec Mode | Full Stack |
|----------|--------------|------------|
| RAM | 8 GB | 16 GB (8 GB minimum with Docker memory limit raised) |
| CPU | 2-core | 4+ cores |
| GPU | Not required | CUDA optional; CPU inference works |
| Disk | 5 GB free | 10 GB free |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04+ | — |

> **Docker:** Increase `deploy.resources.limits.memory` in `docker-compose.yml` to `6G` or `8G` if the container OOMs during model load.

---

## Spatial File Exports

| Format | File | Best used with |
|--------|------|----------------|
| **GeoJSON** | `.geojson` | ArcGIS Online, QGIS, Mapbox |
| **Shapefile** | `.zip` (`.shp`, `.dbf`, `.shx`, `.prj`) | ArcGIS Pro / Desktop, QGIS |
| **KML** | `.kml` | Google Earth, ArcGIS Earth |
| **CSV** | `.csv` | Excel, R, Python |

---

## ArcGIS Live Sync Setup

1. Create a hosted **Feature Layer** in ArcGIS Online (Point geometry) with fields: `station_id`, `species`, `detection_confidence`, `capture_date`, `day_night`.
2. Copy the REST endpoint URL (ends in `/FeatureServer/0`).
3. In the **ArcGIS Sync** tab, enter the URL and your token, then click **Push to ArcGIS**.

---

## Troubleshooting

### SpeciesNet not loading

**Symptom:** Startup log shows `"SpeciesNet failed to load"` and the live panel shows `"not loaded"` for SpeciesNet predictions.

**Cause:** Kaggle credentials are missing or incorrect.

**Fix:**
```bash
# 1. Verify credentials exist in .env
cat .env | grep KAGGLE

# 2. If missing, add them:
echo "KAGGLE_USERNAME=your_username" >> .env
echo "KAGGLE_KEY=your_api_key" >> .env

# 3. Restart
./compose.sh down && ./compose.sh up
```

The pipeline continues without SpeciesNet — MegaDetector animal detection still works, but species names will be missing from results.

---

### MegaDetector v1000 (not used)

MegaDetector v1000 is **not part of the current pipeline**. Only MegaDetector v5a is loaded for detection.

---

### Models not loading

If `GET /api/config/status` returns `"models_loaded": false`, the backend started but a model import failed.

**Most common cause:** uvicorn is using the system Python, not the venv.

```bash
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

`bash dev.sh` handles this automatically.

**Missing packages:**
```bash
source venv/bin/activate
pip install megadetector speciesnet
python force_download.py
```

---

### No animals detected

### Step 1 — Check model status

```
GET http://localhost:8000/api/config/status
```

Should return `{"models_loaded": true}`.

### Step 2 — Watch the live panel

The **Upload & Process** page streams per-model output cards as each image is analysed. If **MDv5a** shows `—` (no detections), the animal was not detected regardless of threshold.

### Step 3 — Lower confidence threshold

The sidebar defaults to **0.35**. Night/IR shots and distant animals often score 0.15–0.25. Try **0.10–0.15**.

### Step 4 — Check image quality

| Condition | Action |
|-----------|--------|
| Very dark / underexposed | Check camera flash settings |
| Animal < 2% of frame | Move camera closer to the path |
| Heavy motion blur | Increase camera shutter speed |
| Corrupted file | Re-export from SD card |

### Step 5 — Disable Low-Spec Mode

INT8 quantization can drop borderline detections (scores 0.20–0.35) below the threshold.

---

### Linux — venv creation fails

**Symptom:** `ensurepip` error on Ubuntu 25.04+ where system Python is 3.14.

```bash
sudo apt-get install python3.12-venv
rm -rf venv
bash install.sh
```

---

### Docker — permission denied

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
sudo chown root:docker /var/run/docker.sock
newgrp docker
```

---

### Docker — port 8000 already in use

```bash
lsof -ti :8000 | xargs kill
./compose.sh up
```

Or change the Docker port mapping in `docker-compose.yml` to `"8080:8000"`.

---

### Windows — machine restarts during processing

This is a GPU driver TDR failure. The app probes for NVIDIA GPUs via `nvidia-smi` before loading any model, and sets `CUDA_VISIBLE_DEVICES=-1` if no healthy GPU is found. Update to the latest version if you see this on an older install:

```bat
git pull && install.bat
```

---

## Technical Notes

### Pipeline

- **MegaDetector v5a** detects Animal / Person / Vehicle bounding boxes on each full image.
- **SpeciesNet** classifies each animal crop (temp JPEG → classify → delete). Returns JSON labels with `common_name`, `scientific_name`, and `hierarchy`; all candidates are ranked by confidence.
- **Day/night classification** runs before species classification; `DayNightClassifier` detects grayscale/low-saturation images as night-vision regardless of pixel brightness.
- **Agreement** tiers (High / Medium / Low) are derived from SpeciesNet confidence (≥ 0.7 / ≥ 0.4 / otherwise).
- Images upload automatically in the browser (400 ms debounce) on file selection — the server receives files before the user clicks "Start".
- SSE stream (`GET /api/images/job/{id}/stream`) emits both `model_event` messages (per model per image) and `progress` heartbeats.
- **Threading model:**
  - *Startup* — `asyncio.to_thread` offloads model loading so the FastAPI event loop stays responsive during boot.
  - *Per-job* — FastAPI `BackgroundTasks` runs `_run_processing` in a starlette threadpool thread. A `threading.Semaphore(1)` limits concurrent jobs to one.
  - *Per-image* — a `ThreadPoolExecutor(_PARALLEL_IMAGES)` processes multiple images concurrently within the job.
  - *Within-image* — OCR and day/night classification run in parallel before detection; EasyOCR calls are serialised via `_ocr_lock`.
- **Privacy scrubbing** runs per image when enabled (Person/Vehicle boxes blurred to `uploads/scrubbed/`).
- `JobManager` uses a `threading.Lock` to protect the `_jobs` dict. Completed job metadata is persisted to SQLite; in-memory results are evicted after TTL but detections remain in the database.
- File uploads are validated by magic bytes (JPEG, PNG, TIFF, BMP, WebP) — not just MIME type headers — before being saved.

### Data

- **EasyOCR** reads date, time, and temperature from camera metadata strips (bottom 10% of image) via regex.
- **Independence rule** — same species + same station + detections within the window = one IDE. RAI = IDEs / trap nights.
- **Privacy scrubbing** — Gaussian blur applied to Person/Vehicle bounding boxes. Original files are never modified.
- **SQLite** (`wildlife_data.db`) in WAL mode — all stations, IDEs, review actions, community observations, project config, job metadata, and ArcGIS sync log stored in one local file.
- **Species library** — 159 African wildlife species with full scientific names (e.g. *Panthera leo*), family, order, IUCN status, and 59 synonym mappings (e.g. "painted wolf" → "African Wild Dog").
- **`SPECIES_TAXONOMY`** — compile-time taxonomy table covering African wildlife species, mapping common name → order / family / genus / scientific name. Used for species library and display normalization.

---

## Recent Changes

### Sessions, review locking & storage (Aug 2026)

- **Username-only sessions** — no password; signed cookie via `SESSION_SECRET`; modal on first visit
- **Review locking** by `detection_id` — duplicate review returns HTTP 409
- **One `images` row per file**, multiple `detections` rows for multi-animal frames
- **Fixed file tier classification** (`empty` / `low_conf` / `valid`)
- **Manual cleanup only** — no background scheduler
- **Docker:** `compose.sh` wrapper, `.dockerignore`, BuildKit cache mounts; `itsdangerous` dependency

### File Lifecycle & Storage Management System (June 2026)

**Tiered File Storage & Classification:**
- Detections and images are automatically sorted into `valid` (confidence > 0.4), `low_conf` (confidence 0.2 - 0.4), and `empty` (no wildlife detected or priority Person/Vehicle detections) tiers to optimize host system storage.
- Deleted results are soft-deleted via SQLite audit records (`deleted_at`), initiating a 7-day grace period during which files can be recovered or downloaded before permanent erasure.
- Bulk downloading files is optimized using asynchronous, chunked ZIP streaming of images alongside a custom manifest and structured `metadata.json` layout.

**Hash-based Deduplication & Clearing Strategies:**
- Computes SHA256 hashes for all uploaded images, preventing duplicate database entries and warning the user in real-time.
- Implements 5 hash-clearing strategies (Empty, Deduplicate, Old, All, and Custom Tiers) directly controllable from the frontend to optimize database performance.

**Storage Dashboard UI:**
- Created [frontend/src/pages/Storage.tsx](file:///home/mutuma/Downloads/camera-traps/frontend/src/pages/Storage.tsx) displaying overall storage usage metrics, interactive progress gauges, deletion warning countdowns, and quick access buttons for batch download and optimization strategies.
- Fully integrated loading skeletons and error boundaries with recovery buttons.

### Concurrency & Threading Improvements (May 2026)

**Parallel image processing within a job (`backend/routers/images.py`):**
- The per-job image loop was fully sequential; images now process in parallel using a `ThreadPoolExecutor` sized to `max(1, min(4, cpu_count // 2))`. On a 4-core machine: 2 images in parallel; on 8 cores: 4 images. Results are collected by index and re-ordered before DB insert so upload order is preserved regardless of completion order.
- Per-image failures (bad file, corrupted read) are caught and logged individually. The rest of the batch continues and `job.error` is set to a summary of which images failed, rather than killing the entire job.

**Job concurrency limit (`backend/routers/images.py`):**
- Added `_job_semaphore = threading.Semaphore(1)`. A second `POST /process/{id}` call now blocks inside the background thread (job stays "queued") until the first job finishes. Prevents concurrent jobs from doubling ML memory usage (~9 GB) and OOM-killing the process.

**Thread-safe `JobManager` (`backend/services/job_manager.py`):**
- Added `self._lock = threading.Lock()` protecting all `self._jobs` dict mutations (`create`, `get`, `delete`, `_evict_expired`). Previously relied on CPython's GIL for dict safety — correct in practice but fragile across Python implementations.
- Filesystem cleanup (`shutil.rmtree`) moved outside the lock so temp-dir deletion doesn't block concurrent readers.

**Dynamic `_executor` sizing (`core/animal_detector.py`):**
- Changed `ThreadPoolExecutor(max_workers=4)` to `ThreadPoolExecutor(max_workers=max(4, cpu_count // 2))`. On an 8-core machine the pool grows to 4; on a 16-core machine to 8, letting more parallel model calls run simultaneously.

**Parallel OCR + day/night classification (`core/image_processor.py`):**
- Added module-level `_pipeline_executor` and `_ocr_lock`. Inside `process_single_image`, OCR and day/night are now submitted as futures simultaneously rather than run sequentially. OCR (~500 ms) and day/night (~100 ms) now overlap, saving ~100 ms per image. `_ocr_lock` serialises EasyOCR calls across threads because `Reader.readtext()` has shared internal buffers.

---

### Pipeline simplified to MDv5a + SpeciesNet (Aug 2026)

- Removed BioCLIP, MegaDetector v1000, and classifier fusion — SpeciesNet is the sole classifier.
- SpeciesNet loads in a **background thread** at startup so the API responds while Kaggle weights download.
- Docker: persistent **`kaggle_cache`** volume; set `KAGGLE_USERNAME` / `KAGGLE_KEY` in `.env`.
- Per-image privacy scrub, independence event persistence, JSON export, hourly activity chart.
- Low-Spec toggle reloads/unloads SpeciesNet without a full server restart.

---

### Reliability & Security (May 2026)

**Backend (`backend/routers/images.py`):**
- Added `_is_allowed_image()` — validates uploaded files by magic bytes (JPEG `FF D8 FF`, PNG `89 50 4E 47`, TIFF, BMP, WebP). Returns HTTP 415 if the file is not a recognised image format, regardless of filename extension.
- DB save failure changed from `except Exception: pass` → `logger.error(...)`. Failures now appear in logs instead of being silently discarded.
- Job metadata (`status`, `total`, `completed`, `error`, `created_at`, `finished_at`) is saved to SQLite `jobs` table on job completion via `db_manager.save_job()`.

**Database (`core/db_manager.py`):**
- Added `jobs` table (created alongside existing tables on startup).
- Added `save_job()` — upsert by `job_id`.
- Added `load_recent_jobs()` — returns the 50 most recent completed/errored jobs ordered by `finished_at`.

**Frontend (`frontend/src/`):**
- Added `ErrorBoundary.tsx` — class-based React Error Boundary that renders a red error card with a "Try again" button instead of a blank crash.
- Wrapped all 15 routes in `App.tsx` with `<ErrorBoundary label="...">` — a render error in any single page no longer crashes the entire app.

---

### Review Results & Review Queue Overhaul (May 2026)

- Table ↔ Gallery view toggle with bounding box SVG overlays (multi-colour per detection).
- Multi-animal grouping in gallery — single card per image with all boxes overlaid.
- Per-detection inline editing in lightbox — edits route to correct `detections` table row via `detection_id`.
- Sortable columns, confidence colour bars, day/night badges.
- Redesigned Review Queue as responsive card grid with bounding boxes, inline confirm/correct/flag panels, and agreement badges.
- Pagination on both Results (50 per page) and Review Queue (20 per page) with keyboard navigation (J/K or arrow keys, A/C/F shortcuts on focused card).

---

### FastAPI + React Migration (May 2026)

- Replaced the Streamlit monolith with a FastAPI REST backend and React + TypeScript frontend.
- AI models load once at startup via FastAPI lifespan (off the event loop via `asyncio.to_thread`).
- Real-time SSE progress stream replaced 1.5 s polling.
- Pipeline Ready badge on Upload page polls `/api/config/status` until models are loaded.
- SQLite WAL mode for concurrent reads during processing.
- Job TTL eviction cleans up temp dirs after 2 hours.
- File upload size limit (50 MB, configurable via `MAX_UPLOAD_MB` env var), path-traversal sanitization.
- `.env` support with fallbacks for `DB_PATH`, `CORS_ORIGINS`, `MAX_UPLOAD_MB`, `JOB_TTL_HOURS`.

---

## License

Open-source — intended for wildlife research and conservation.
