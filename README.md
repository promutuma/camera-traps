# Wildlife Camera Trap Auto-Analyzer

A **FastAPI + React platform** for automated analysis of wildlife camera trap images, designed for the **Gambella Wetland Landscape Baseline Survey** and similar conservation programmes.

The system handles the full pipeline from raw images to publication-ready outputs: OCR metadata extraction, AI animal detection (**MegaDetector V5a**), species identification (**BioClip**), independent detection event (IDE) computation, QC flagging, privacy scrubbing, and spatial export — all through a modern browser-based dashboard.

![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)
![React](https://img.shields.io/badge/React-19-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![MegaDetector](https://img.shields.io/badge/MegaDetector-V5a-blue)
![BioClip](https://img.shields.io/badge/BioClip-Enabled-green)
![Python](https://img.shields.io/badge/Python-3.11--3.13-blue)
![Recommended Python](https://img.shields.io/badge/Recommended-Python%203.12-brightgreen)

---

## Features

| Tab | Feature |
|-----|---------|
| Upload & Process | Batch upload with real-time progress bar, OCR, MegaDetector + BioClip pipeline |
| Review Results | Editable table with inline species/notes editing, Excel/CSV export |
| Statistics | Per-species bar charts, day/night pie chart, confidence distribution |
| History | Long-term trends from the SQLite database, CSV export |
| Diagnostics | OCR strip debugger, raw model output viewer (MegaDetector + BioClip top-20) |
| Ecological Analytics | IDE computation, RAI, species richness, accumulation curve, group size, visitation rate |
| QC Dashboard | Automated quality-control checks with colour-coded severity flags |
| Stations & Deployments | Camera registry, GPS coordinates, deployment history, trap-night calculator |
| Review Queue | HITL expert review — confirm / correct / flag with reviewer logging and privacy audit |
| Community Observer | Field observer sighting entry, cross-verification against camera data |
| Spatial & Map | Interactive Leaflet map, GeoJSON / Shapefile / KML / CSV export |
| Species Library | Pre-loaded mammal reference library, synonym resolver, quick lookup |
| Corridor Analysis | Directional flow detection, passage frequency, bottleneck identification |
| Project Config | Multi-project support, indicator thresholds, baseline locking, JSON export |
| ArcGIS Sync | Offline file exports (GeoJSON, Shapefile, KML) + live push to ArcGIS Online / Enterprise |

---

## Architecture

```
camera-traps/
├── backend/                      # FastAPI application
│   ├── main.py                   # App factory, CORS, lifespan model loading
│   ├── routers/                  # One router per feature tab (16 total)
│   │   ├── config.py             # GET/PATCH /api/config
│   │   ├── images.py             # Upload, background processing, job polling
│   │   ├── results.py            # Review, edit, export
│   │   ├── statistics.py         # Stats summary
│   │   ├── history.py            # History, CSV export
│   │   ├── diagnostics.py        # Deep inspection endpoint
│   │   ├── ecological.py         # IDE, RAI, richness, accumulation, visitation
│   │   ├── qc.py                 # QC flags and summary
│   │   ├── stations.py           # Station registry and deployments
│   │   ├── review.py             # Review queue, confirm/correct/flag
│   │   ├── community.py          # Community observer data
│   │   ├── spatial.py            # GeoJSON, Shapefile, KML, CSV
│   │   ├── species.py            # Species library, synonym resolver
│   │   ├── corridor.py           # Corridor movement analysis
│   │   ├── project.py            # Project configuration
│   │   └── arcgis.py             # ArcGIS push and exports
│   ├── models/
│   │   ├── state.py              # AppState + AppConfig dataclasses
│   │   └── schemas.py            # Pydantic request/response models
│   └── services/
│       └── job_manager.py        # In-memory background job tracker
│
├── frontend/                     # React + TypeScript + Vite application
│   ├── vite.config.ts            # Vite dev proxy: /api → localhost:8000
│   ├── src/
│   │   ├── App.tsx               # React Router with 15 routes
│   │   ├── api/client.ts         # Typed API client (axios)
│   │   ├── store/configStore.ts  # Zustand global config store
│   │   ├── components/Layout/
│   │   │   ├── Sidebar.tsx       # Live config sidebar (replaces st.sidebar)
│   │   │   └── TabNav.tsx        # Top tab navigation
│   │   └── pages/                # One page component per tab (15 total)
│
├── core/                         # UNCHANGED — all AI/ML business logic
│   ├── animal_detector.py        # MegaDetector + BioClip ensemble
│   ├── bioclip_classifier.py     # OpenCLIP species classifier
│   ├── day_night_classifier.py   # Brightness-based day/night
│   ├── ocr_processor.py          # EasyOCR metadata extraction
│   ├── image_processor.py        # Unified processing pipeline
│   ├── db_manager.py             # SQLite schema and persistence
│   ├── independence_engine.py    # 30-min IDE grouping + RAI
│   ├── qc_engine.py              # QC flag system
│   ├── station_manager.py        # Station registry + deployments
│   ├── privacy_scrubber.py       # Gaussian blur for Person/Vehicle
│   ├── review_engine.py          # HITL accept/correct/reject queue
│   ├── community_observer.py     # Field observer sighting store
│   ├── spatial_exporter.py       # GeoJSON, Shapefile, KML export
│   ├── species_library.py        # Species reference library
│   ├── corridor_analyzer.py      # Directional corridor flow analysis
│   ├── project_config.py         # Multi-project config + baselines
│   └── arcgis_sync.py            # ArcGIS REST API sync
│
├── dev.sh                        # One command: starts both servers
├── wildlife_data.db              # SQLite database (auto-created on first run)
├── requirements.txt              # Python ML dependencies
├── Dockerfile                    # Multi-stage build (Node → React → Python)
├── docker-compose.yml            # Container orchestration
├── install.sh                    # Mac / Linux one-click installer (creates venv)
├── install.bat                   # Windows one-click installer
└── force_download.py             # Pre-downloads AI models (~1.5 GB)
```

---

## Quick Start

### Step 1 — Install Python dependencies

> **Python version:** Python 3.12 is strongly recommended. Python 3.14+ is **not supported** — key packages (`megadetector`, `yolov5`) have no pre-built wheels for 3.14 and will fail to compile.

**Option A — Use the existing installer (recommended)**

The installer creates a `venv/` with all Python dependencies and pre-downloads the AI models:

```bash
# macOS / Linux
chmod +x install.sh
./install.sh

# Windows
install.bat
```

**Option B — Manual setup**

```bash
# Create venv with Python 3.12
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate.bat

# Install Python dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Pre-download AI models (one-time, ~1.5 GB)
python force_download.py
```

### Step 2 — Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 3 — Run

```bash
bash dev.sh
```

Then open **http://localhost:5173** in your browser.

- Frontend (React) → `http://localhost:5173`
- API docs (Swagger) → `http://localhost:8000/docs`

`dev.sh` automatically activates the `venv/` if it exists, then starts both the FastAPI backend (port 8000) and the Vite dev server (port 5173) together. Press `Ctrl+C` to stop both.

---

## Installation Options

### Option A — macOS / Linux (one-click)

```bash
git clone <repository-url>
cd camera-traps
chmod +x install.sh
./install.sh
cd frontend && npm install && cd ..
bash dev.sh
```

For a clean reinstall: `./install.sh --fresh`

**What the Linux installer does automatically:**

- Installs `python3.12-venv` via apt so the venv works even when the system default Python is newer (e.g. 3.14 on Ubuntu 25.04+).
- Probes for an NVIDIA GPU via `nvidia-smi`. If none is found, installs the **CPU-only** PyTorch wheel (~300 MB) instead of the full CUDA build (~3 GB), saving both download time and disk space.
- Downloads all other packages in parallel with live progress, speed, and ETA — with automatic retry (up to 3 attempts) on transient failures.

---

### Option B — Windows (one-click)

**Requirements:** Python 3.12 from [python.org](https://www.python.org/downloads/) (check "Add Python to PATH"). Node.js from [nodejs.org](https://nodejs.org/).

```bat
git clone <repository-url>
cd camera-traps
install.bat
cd frontend && npm install && cd ..
bash dev.sh
```

> **Tip:** If you have Python 3.14 installed, you do **not** need to uninstall it. Install Python 3.12 alongside it — `install.bat` uses the Windows Python Launcher (`py -3.12`) to pick the right version automatically.

For a clean reinstall: `install.bat --fresh`

---

### Option C — Conda / Miniconda

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

To update after `git pull`:
```bash
conda env update -f environment.yml --prune
```

---

### Option D — Docker (production, single server)

In production the FastAPI backend serves the built React app as static files — one server, one port.

```bash
git clone <repository-url>
cd camera-traps

# Build and run
docker-compose up --build
```

Access the app at `http://localhost:8000`.

To persist the database between runs:
```bash
docker run -p 8000:8000 \
  -v $(pwd)/wildlife_data.db:/app/wildlife_data.db \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  wildlife-analyzer
```

---

### Option E — VS Code Dev Container

The repository includes `.devcontainer/devcontainer.json` for a pre-configured container.

1. Install the **Dev Containers** extension in VS Code.
2. Open the repository folder and click **"Reopen in Container"** when prompted.
3. The container installs all Python and Node dependencies automatically.
4. Run `bash dev.sh` in the integrated terminal.

---

## How the Dev Setup Works

In development, two processes run simultaneously:

```
Browser → http://localhost:5173  →  Vite dev server (React, HMR)
                                         ↓ proxy /api/*
                                    FastAPI  (http://localhost:8000)
                                         ↓
                                    core/ (AI models, SQLite)
```

Vite's proxy (`/api → localhost:8000`) means you only ever open one URL. All API calls from the frontend transparently reach FastAPI. In production (Docker), FastAPI serves the built React `dist/` directly — no Vite needed.

---

## Configuration

The left sidebar in the app exposes all runtime settings. Changes are sent to `PATCH /api/config` instantly and take effect on the next image processing run — no restart required.

| Setting | Description |
|---------|-------------|
| Detection Confidence | Score cutoff (default 0.35). Lower = more detections, higher = fewer false positives. |
| Brightness Threshold | Day/Night classification sensitivity (0–255). |
| Metadata Strip (%) | % of image bottom scanned for date/time text. |
| Auto-Scrub Person/Vehicle | Apply Gaussian blur to privacy-sensitive detections. |
| Blur Strength | Gaussian kernel size (11–101, odd numbers). |
| Independence Window (min) | Same species + same station within this window = one IDE (default 30 min). |
| Default Station ID | Fallback when filename doesn't encode a station. |
| Default Trap Nights | Used for RAI when no deployment records exist. |
| Review Queue Threshold | Images below this confidence appear in the Review Queue for expert review. |
| Reviewer ID | Name/ID logged against review actions. |
| Low-Spec Mode | INT8 dynamic quantization — halves model memory on machines with < 8 GB RAM. |
| CPU Threads (Windows only) | PyTorch intra-op threads (default ¼ of core count — the stable sweet spot on Windows). |

---

## Spatial File Exports

The **Spatial & Map** and **ArcGIS Sync** tabs provide offline GIS formats — no ArcGIS account required:

| Format | File | Best used with |
|--------|------|----------------|
| **GeoJSON** | `.geojson` | ArcGIS Online, QGIS, Mapbox, web apps |
| **Shapefile** | `.zip` (`.shp`, `.dbf`, `.shx`, `.prj`) | ArcGIS Pro / Desktop, QGIS, any desktop GIS |
| **KML** | `.kml` | Google Earth, ArcGIS Earth, ArcGIS Pro |
| **CSV** | `.csv` | Excel, R, Python — georeferenced detections |

---

## ArcGIS Live Sync Setup

1. In ArcGIS Online, create a hosted **Feature Layer** (Point geometry) with fields: `station_id`, `species`, `detection_confidence`, `capture_date`, `day_night`.
2. Copy the layer's REST endpoint URL from **Item Details → URL** (ends in `/FeatureServer/0`).
3. In the **ArcGIS Sync** tab, enter the URL and your ArcGIS token, then click **Push to ArcGIS**.

For ArcGIS Enterprise, use your portal root URL (e.g. `https://gis.yourorg.com/portal`).

---

## Performance & System Requirements

### Model memory footprint

AI models are loaded once at startup (FastAPI lifespan) and held in memory for the lifetime of the server — not reloaded per request.

| Component | RAM | Disk |
|-----------|-----|------|
| PyTorch runtime | ~1.2 GB | ~2 GB |
| MegaDetector V5a | ~600 MB | ~600 MB |
| BioClip (OpenCLIP) | ~850 MB | ~850 MB |
| **Total** | **~2.6 GB** | **~3.5 GB** |

### Minimum recommended specs

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | Any modern dual-core | 4+ cores |
| GPU | Not required | CUDA GPU speeds up BioClip on Linux/Mac |
| Disk | 5 GB free | 10 GB free |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | — |

> **GPU detection:** On all platforms the installer probes for an NVIDIA GPU via `nvidia-smi`. If found, the full CUDA PyTorch build is used; if not, the CPU-only build is installed automatically. AMD and Intel GPUs are not CUDA-capable and always fall back to CPU.

### Running on a low-RAM machine

Enable **Low-Spec / Low-Memory Mode** in the sidebar. This applies INT8 dynamic quantization to all models, roughly halving memory use. Detection accuracy may decrease slightly on borderline images.

---

## VS Code Interpreter Setup

If VS Code shows "Cannot find module" for `cv2`, `torch`, etc., it is checking the wrong Python:

1. Press `Ctrl+Shift+P` → **Python: Select Interpreter**
2. Choose `./venv/bin/python` (venv) or the `wildlife-analyzer` conda env

---

## Troubleshooting: Linux — venv Creation Fails (`ensurepip` error)

On Ubuntu 25.04+ the system default Python is 3.14, so `python3-venv` installs the venv module for 3.14 — not for Python 3.12 that the installer picks. The symptom is:

```
Error: Command '['.../python3.12', '-m', 'ensurepip', '--upgrade']' returned non-zero exit status 1.
```

The installer now handles this automatically by also installing `python3.12-venv`. If you hit this on an older installer version, fix it manually:

```bash
sudo apt-get install python3.12-venv
rm -rf venv
bash install.sh
```

---

## Troubleshooting: Linux — Disk Quota Exceeded During Install

By default, `pip download` fetches every wheel regardless of what is already installed in the venv. Without GPU auto-detection, this includes the full CUDA PyTorch build (~2.5 GB for torch alone) plus NVIDIA libraries (`nvidia_cublas`, `nvidia_cufft`, `triton`, etc.) even on CPU-only machines — easily exceeding per-user disk quotas.

The current installer avoids this by:
1. Detecting the GPU with `nvidia-smi` before the parallel download phase.
2. Pre-installing the CPU-only PyTorch wheel if no GPU is found.
3. Excluding `torch` and `torchvision` from the parallel download queue so the CUDA builds are never fetched.

If you hit the quota error on an older version, upgrade and reinstall:

```bash
git pull
rm -rf venv temp_packages_download
bash install.sh
```

---

## Troubleshooting: Models Not Loading

If the API returns `"models_loaded": false` at `GET /api/config/status`, the backend started but failed to import one or more packages. Check the uvicorn terminal output for the specific error.

**Most common cause:** uvicorn is using the system Python, not the venv.

```bash
# Always activate the venv before running manually
source venv/bin/activate          # Windows: venv\Scripts\activate.bat
uvicorn backend.main:app --reload --port 8000
```

`bash dev.sh` handles this automatically — it activates `venv/` if present.

**Missing packages:**
```bash
source venv/bin/activate
pip install megadetector open_clip_torch
python force_download.py
```

Then restart the backend. Models reload automatically on next startup.

---

## Troubleshooting: No Animals Detected

### Step 1 — Check model status

```
GET http://localhost:8000/api/config/status
```

Should return `{"models_loaded": true, "error": null}`. If `models_loaded` is `false`, fix the model loading issue first (see above).

### Step 2 — Use the Diagnostics tab

Upload a failing image to the **Diagnostics** tab and click **Run Deep Inspection**. This shows:
- Raw OCR extraction output
- MegaDetector candidates at all confidence levels
- BioClip top-20 species scores

If MegaDetector returns no candidates, the animal was not detected regardless of threshold.

### Step 3 — Lower the confidence threshold

The sidebar **Detection Confidence** defaults to **0.35**. Night/IR shots and distant animals often score 0.15–0.25. Try **0.10–0.15** and reprocess.

### Step 4 — Check image quality

| Condition | Action |
|-----------|--------|
| Very dark / underexposed | Check camera flash settings |
| Animal < 2% of frame | Reposition the camera closer to the path |
| Heavy motion blur | Increase camera shutter speed |
| Corrupted or zero-byte file | Re-export from SD card |

### Step 5 — Disable Low-Spec mode

INT8 quantization can drop borderline detections (scores 0.20–0.35) below the threshold. Uncheck **Low-Spec Mode** in the sidebar and restart the backend.

---

## Troubleshooting: Windows — Machine Restarts During Processing

This is a GPU driver TDR failure, not a RAM crash.

The app sets `CUDA_VISIBLE_DEVICES=-1` and probes for NVIDIA GPUs via `nvidia-smi` (a lightweight CLI that does not open a CUDA context). On machines without a healthy NVIDIA driver, CUDA is disabled entirely before any model loads.

If you are on an older version and seeing restarts, update and restart:

```bat
git pull
install.bat
```

### Windows stability fixes included in the current version

| Fix | What it prevents |
|-----|-----------------|
| `nvidia-smi` probe + conditional `CUDA_VISIBLE_DEVICES=-1` | GPU TDR crash on machines without a healthy NVIDIA GPU |
| `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE` | OpenMP deadlock from PyTorch + OpenCV + EasyOCR loading duplicate DLLs |
| `torch.set_num_threads(¼ cores)` on Windows | Thread-stack exhaustion when all three models compete for threads |
| `multiprocessing.freeze_support()` | Crash when EasyOCR or YOLO spawn worker processes |

---

## Troubleshooting: Windows — SSL Certificate Error During Install

If `install.bat` fails with `ssl.SSLCertVerificationError`, your network performs SSL inspection with a corporate root CA that Python does not trust.

**Option 1 — Re-run the updated installer** (pulls the latest version which handles this):
```bat
git pull
install.bat
```

**Option 2 — Manual override:**
```bat
set PYTHONHTTPSVERIFY=0
venv\Scripts\activate.bat
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir -r requirements.txt
```

**Option 3 — Ask IT** to export the corporate root CA certificate and:
```bat
set REQUESTS_CA_BUNDLE=C:\path\to\corporate-ca.pem
set SSL_CERT_FILE=C:\path\to\corporate-ca.pem
pip install --no-cache-dir -r requirements.txt
```

---

## Technical Notes

- **MegaDetector V5a** — Microsoft AI for Earth model detecting Animal / Person / Vehicle classes.
- **BioClip (OpenCLIP)** — Imageomics foundation model for fine-grained species classification from cropped detections.
- **EasyOCR** — Reads date, time, and temperature from camera metadata strips via regex.
- **Independence rule** — Same species + same station + detections within the window → one IDE. RAI = IDEs / trap nights.
- **Privacy scrubbing** — Gaussian blur applied to Person/Vehicle bounding boxes. Originals are never modified.
- **Background jobs** — Image processing runs in a `ThreadPoolExecutor` background task. The frontend polls `GET /api/images/job/{id}` every 1.5 s and updates the progress bar in real time.
- **SQLite** (`wildlife_data.db`) — All persistent data (stations, IDEs, review actions, community observations, project config, ArcGIS sync log) stored in a single local database.

---

## Recent Changes

### FastAPI + React Migration (May 2026)

- Replaced the Streamlit monolith (`app.py`) with a **FastAPI REST backend** (`backend/`) and a **React + TypeScript frontend** (`frontend/`).
- All `core/` modules are unchanged — they are called directly by FastAPI route handlers.
- AI models now load once at server startup via FastAPI's lifespan context (replaces `@st.cache_resource`).
- Image processing moved to background tasks with a polling API (`POST /process → GET /job/{id}`) giving real-time progress in the browser.
- Config sidebar changes now call `PATCH /api/config` and take effect immediately without a page reload.
- Added `dev.sh` — one command to start both servers with automatic venv activation.
- Updated `Dockerfile` to a multi-stage build: Node stage builds the React app, Python stage serves everything via FastAPI on port 8000.
- Interactive API documentation available at `http://localhost:8000/docs`.

### Linux Installer & Download Engine (May 2026)

- `install.sh` now installs `python3.12-venv` explicitly on Debian/Ubuntu, fixing venv creation failures on systems where the default Python is 3.14+ (e.g. Ubuntu 25.04).
- Added NVIDIA GPU auto-detection via `nvidia-smi`: CPU-only PyTorch is installed when no GPU is present, reducing the download from ~3 GB to ~300 MB and preventing disk quota errors.
- Rewrote the parallel download engine (`install_dependencies.py`):
  - Live dashboard showing every package with phase badges: `[Queued]` → `[Resolving]` → `[Downloading]` → `[Done]`
  - Real-time bytes downloaded / total, speed (MB/s), and ETA parsed from pip's output
  - Automatic retry with exponential back-off — up to 3 attempts at 5 s → 15 s → 30 s delays
  - Dynamic worker count scaled to `min(8, cpu_count ÷ 2)` instead of a hardcoded 5
  - Install phase now streams pip output live (package-by-package) instead of running silently

### Windows Installer & Dependency Fixes (May 2026)

- `install.bat` now uses the Windows Python Launcher to prefer Python 3.12 → 3.11 → 3.13, avoiding the Python 3.14 wheel-compilation failure for `megadetector` and `yolov5`.
- Replaced blanket `CUDA_VISIBLE_DEVICES=-1` with `nvidia-smi` GPU probe — CUDA is now automatically enabled on machines with a working NVIDIA GPU.
- Pinned `megadetector>=5.0.0,<6.0.0` to avoid the breaking API changes in version 10.x.
- Added `--fresh` flag to both `install.sh` and `install.bat` for clean-slate reinstalls.

### Earlier Changes (May 2026)

- Removed `numpy<2.0.0` upper-bound constraint to allow NumPy 2.x wheels on Python 3.14.
- Fixed `TypeError` in Excel report generation on newer Pandas/Arrow string backends.
- Added **Low-Spec / Low-Memory Mode** for INT8 quantization on machines with < 8 GB RAM.
- Added `OMP_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE` and `torch.set_num_threads(1)` on Windows to fix OpenMP deadlocks and thread-stack exhaustion.

---

## License

Open-source — intended for wildlife research and conservation.
