# Wildlife Camera Trap Auto-Analyzer

A **FastAPI + React platform** for automated analysis of wildlife camera trap images, designed for the **Gambella Wetland Landscape Baseline Survey** and similar conservation programmes.

The system runs a full multi-model AI ensemble pipeline — OCR metadata extraction, parallel animal detection (**MegaDetector V5a + V1000**), species identification (**BioClip + SpeciesNet**), detection fusion, independent detection event (IDE) computation, QC flagging, privacy scrubbing, and spatial export — all through a modern browser-based dashboard with real-time per-model output streaming.

![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)
![React](https://img.shields.io/badge/React-19-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![MegaDetector](https://img.shields.io/badge/MegaDetector-V5a%20%2B%20V1000-blue)
![BioClip](https://img.shields.io/badge/BioClip-Enabled-violet)
![SpeciesNet](https://img.shields.io/badge/SpeciesNet-Google-teal)
![Python](https://img.shields.io/badge/Python-3.11--3.13-blue)
![Recommended Python](https://img.shields.io/badge/Recommended-Python%203.12-brightgreen)

---

## AI Models — Detectors & Classifiers Explained

Understanding what each model does is important for configuring the pipeline correctly and diagnosing issues.

### The distinction: Detectors vs Classifiers

| | Detector | Classifier |
|---|---|---|
| **Question answered** | *Is there an animal here, and where?* | *What species is this animal?* |
| **Output** | Bounding box + coarse label (Animal / Person / Vehicle) | Species name + confidence score |
| **Works on** | The full image | A cropped region of the detected animal |
| **Speed** | Fast (object detection) | Slower (fine-grained recognition) |
| **Example** | "There is an Animal at [0.2, 0.1, 0.4, 0.3] with 94% confidence" | "Lion 89%, Leopard 5%, Cheetah 3%…" |

The pipeline always runs in order: **Detect first → then Classify the crop**. A classifier cannot run without a detector first finding the animal.

---

### Model 1 — MegaDetector V5a (Detector)

**Source:** [agentmorris/MegaDetector](https://github.com/agentmorris/MegaDetector) (formerly Microsoft AI for Earth)  
**Architecture:** YOLOv5  
**Install:** `pip install megadetector` (weights auto-download on first run, ~600 MB)

MegaDetector is the primary detector. It scans every full image and returns bounding boxes for anything it identifies as an **animal**, **person**, or **vehicle**. It does not attempt to identify species — that is the classifier's job.

**What it needs to run:**
- The `megadetector` Python package
- ~600 MB disk for the model weights (downloaded automatically to `~/megadetector_models/`)
- ~600 MB RAM while loaded
- No GPU required (CPU inference works well)

**Output format:**
```
detections: [
  { category: "1", conf: 0.94, bbox: [x, y, w, h] }   ← normalised 0–1, top-left origin
]
category "1" = Animal,  "2" = Person,  "3" = Vehicle
```

**Confidence threshold:** 0.15–0.25 is the recommended range for V5a. The app defaults to 0.35 — lower it in the sidebar if animals are being missed on dark or distant shots.

---

### Model 2 — MegaDetector V1000 / Redwood (Detector)

**Source:** [agentmorris/MegaDetector](https://github.com/agentmorris/MegaDetector), July 2024 release  
**Architecture:** YOLOv5 (newer training data than V5a)  
**Install:** Same `megadetector` package. Weights downloaded automatically on first run.  
**Model string used:** `"redwood"` (one of five V1000 variants: redwood, spruce, cedar, larch, sorrel)

MDv1000 is the second detector running **in parallel** with MDv5a. Because each was trained on a different dataset partition, they have complementary blind spots — an animal missed by one is often caught by the other.

After both detectors run, the pipeline applies **Non-Maximum Suppression (NMS)** to merge their bounding boxes:
- Boxes from both detectors that overlap (IoU ≥ 0.5) → merge into one, keep the higher-confidence geometry
- Boxes unique to one detector → kept if confidence ≥ threshold
- Result: better recall without duplicate detections

**What it needs to run:**
- Same `megadetector` package as V5a
- Additional ~600 MB disk + ~600 MB RAM (both models loaded simultaneously)
- Automatically skipped when **Low-Spec Mode** is enabled

---

### Model 3 — BioClip (Classifier)

**Source:** [Imageomics/bioclip](https://github.com/Imageomics/BioCLIP) — OpenCLIP foundation model  
**Architecture:** CLIP (Contrastive Language–Image Pre-Training), vision transformer  
**Install:** `pip install open_clip_torch` (weights auto-download, ~850 MB)

BioClip is a **zero-shot** classifier. Unlike trained classifiers, it does not have a fixed set of species it was trained to recognise. Instead, it compares the animal crop against a text description of every species in a custom label list (`WILDLIFE_CLASSES`, ~100 entries for African wildlife). The species whose text description best matches the image wins.

This makes BioClip flexible — you can add new species to the label list without retraining. The trade-off is that it is less accurate than trained classifiers for common species.

**What it needs to run:**
- `open_clip_torch` package
- ~850 MB disk + ~850 MB RAM
- GPU recommended but not required (CPU works, just slower)

**How it receives input:** The pipeline extracts a padded crop (10% padding around the bounding box) and passes it as a PIL image.

---

### Model 4 — SpeciesNet (Classifier)

**Source:** [google/cameratrapai](https://github.com/google/cameratrapai)  
**Architecture:** EfficientNetV2-M trained on **65 million** camera trap images  
**Install:** `pip install speciesnet`  
**Model download:** Requires **Kaggle credentials** (free account) — see setup below.

SpeciesNet is Google's purpose-built camera trap classifier. Unlike BioClip (zero-shot), SpeciesNet was directly trained on camera trap images across diverse global ecosystems. It classifies into **2 000+ labels** covering individual species, higher taxonomic ranks (Felidae, Mammalia), and non-animal classes (blank, vehicle, human).

Because it was trained specifically on camera trap imagery, it outperforms general-purpose CLIP models on common species and gives the ensemble its strongest species-level signal.

**What it needs to run:**
- `speciesnet` Python package
- A free [Kaggle account](https://www.kaggle.com/) with API credentials
- Model weights downloaded automatically from Kaggle on first run
- ~1 GB disk + ~1 GB RAM

**Kaggle credentials setup:**

```bash
# 1. Create a free account at https://www.kaggle.com/
# 2. Go to: Account → Settings → API → Create New Token
#    This downloads a kaggle.json file.
# 3. Add to your .env file (in the project root):
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

If credentials are absent, SpeciesNet logs a warning at startup and the pipeline continues with BioClip only — no crash, no manual intervention needed.

---

### How the ensemble combines all four models

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: DETECTION                        │
│                                                             │
│  Full image → MDv5a   ─────────┐                           │
│                                 ├──► NMS Fusion             │
│  Full image → MDv1000 ─────────┘    (IoU ≥ 0.5)            │
│                                        │                    │
│                              merged bounding boxes          │
└─────────────────────────────────────────────────────────────┘
                                         │  Animal boxes only
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│               STAGE 2: CROP + CLASSIFY (parallel)           │
│                                                             │
│  Crop + 10% padding → BioClip    → [(Lion, 0.89), ...]     │
│                     → SpeciesNet → [(Panthera leo, 0.82)]  │
│                                                             │
│  Fusion (weighted avg):                                     │
│    score = 0.45 × BioClip + 0.55 × SpeciesNet              │
│    + 0.08 agreement bonus when both pick the same species   │
└─────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   FINAL OUTPUT per detection                 │
│                                                             │
│  species     = "Lion"                                       │
│  confidence  = 0.91                                         │
│  agreement   = "High"   ← both classifiers agreed          │
│  breakdown   = { MDv5a: …, MDv1000: …,                     │
│                  BioClip: …, SpeciesNet: … }                │
└─────────────────────────────────────────────────────────────┘
```

**Agreement levels:**
- **High** — both classifiers pick the same top species (or genus match). Confidence bonus applied.
- **Medium** — classifiers share a keyword (e.g. "leopard" vs "African leopard"). Partial bonus.
- **Low** — classifiers disagree. Result is the weighted average; image sent to Review Queue.

**Low-Spec Mode** (sidebar toggle) disables MDv1000 and SpeciesNet, reverting to the original single-detector + BioClip pipeline to stay within ~2 GB RAM.

---

## Features

| Tab | Feature |
|-----|---------|
| Upload & Process | Auto-upload on file selection; real-time per-model output panel (MDv5a · MDv1000 · BioClip · SpeciesNet · Fusion); SSE progress stream |
| Review Results | Table and gallery views; bounding box overlays; multi-animal grouping; sortable columns; inline per-detection species editing; confidence bars; Excel/CSV export |
| Statistics | Per-species bar charts, day/night pie chart, confidence distribution |
| History | Long-term trends from SQLite database, CSV export |
| Diagnostics | OCR strip debugger, raw model output viewer (all four models) |
| Ecological Analytics | IDE computation, RAI, species richness, accumulation curve, group size, visitation rate |
| QC Dashboard | Automated quality-control checks with colour-coded severity flags |
| Stations & Deployments | Camera registry, GPS coordinates, deployment history, trap-night calculator |
| Review Queue | Responsive card grid with bounding boxes; confirm / correct / flag inline; agreement badge; reviewer logging |
| Community Observer | Field observer sighting entry, cross-verification against camera data |
| Spatial & Map | Interactive Leaflet map, GeoJSON / Shapefile / KML / CSV export |
| Species Library | 159 African wildlife species with full scientific names, IUCN status, synonym resolver |
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
│   │   ├── images.py             # Upload (auto), processing, SSE stream
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
│   │   ├── state.py              # AppState (md_model, md_v1000_model, bio_model,
│   │   │                         #           speciesnet_model, …) + AppConfig
│   │   └── schemas.py            # Pydantic request/response models
│   └── services/
│       └── job_manager.py        # In-memory job tracker with model_events queue
│
├── frontend/                     # React + TypeScript + Vite application
│   ├── vite.config.ts            # Vite dev proxy: /api → localhost:8000
│   └── src/
│       ├── App.tsx               # React Router with 15 routes
│       ├── api/client.ts         # Typed axios client
│       ├── store/configStore.ts  # Zustand global config store
│       └── pages/
│           ├── Upload.tsx        # Auto-upload + live model output panel
│           └── …                 # 14 other page components
│
├── core/                         # AI/ML business logic
│   ├── animal_detector.py        # MegaDetectorWrapper (v5a + v1000) + AnimalDetector
│   │                             # orchestrator with parallel inference
│   ├── ensemble_engine.py        # NMS detection fusion + weighted species fusion
│   ├── speciesnet_classifier.py  # Google SpeciesNet wrapper (EfficientNetV2-M)
│   ├── bioclip_classifier.py     # OpenCLIP zero-shot species classifier
│   ├── species_library.py        # 159-species African wildlife DB + synonyms
│   ├── day_night_classifier.py   # Brightness-based day/night
│   ├── ocr_processor.py          # EasyOCR metadata extraction
│   ├── image_processor.py        # Unified processing pipeline
│   ├── db_manager.py             # SQLite schema and persistence (WAL mode)
│   ├── independence_engine.py    # 30-min IDE grouping + RAI
│   ├── qc_engine.py              # QC flag system
│   ├── station_manager.py        # Station registry + deployments
│   ├── privacy_scrubber.py       # Gaussian blur for Person/Vehicle
│   ├── review_engine.py          # HITL accept/correct/reject queue
│   ├── community_observer.py     # Field observer sighting store
│   ├── spatial_exporter.py       # GeoJSON, Shapefile, KML export
│   ├── corridor_analyzer.py      # Directional corridor flow analysis
│   ├── project_config.py         # Multi-project config + baselines
│   └── arcgis_sync.py            # ArcGIS REST API sync
│
├── dev.sh                        # One command: starts both servers
├── wildlife_data.db              # SQLite database (auto-created on first run)
├── requirements.txt              # Python ML dependencies
├── backend/requirements.txt      # FastAPI/server dependencies
├── Dockerfile                    # Multi-stage build (Node → React → Python)
├── docker-compose.yml            # Container orchestration with named volumes
├── .env.example                  # Environment variable template
├── install.sh                    # Mac / Linux one-click installer
├── install.bat                   # Windows one-click installer
└── force_download.py             # Pre-downloads AI models
```

---

## Quick Start

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
python force_download.py          # pre-downloads MDv5a + BioClip (~1.5 GB)
```

### Step 2 — Set up SpeciesNet credentials (optional but recommended)

SpeciesNet downloads its weights from Kaggle on first run. Without credentials the pipeline falls back to BioClip-only — still functional, just less accurate on common species.

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

### Option A — macOS / Linux (one-click)

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

### Option B — Windows (one-click)

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

---

### Option D — Docker (production, single server)

```bash
git clone <repository-url>
cd camera-traps
cp .env.example .env          # add KAGGLE_USERNAME and KAGGLE_KEY for SpeciesNet
docker compose up --build
```

Access the app at **http://localhost:8000**.

> Use `docker compose` (space, not hyphen). If you get `command not found`:
> ```bash
> echo "alias docker-compose='docker compose'" >> ~/.bashrc && source ~/.bashrc
> ```

Named volumes prevent model re-downloads on rebuild:
```
easyocr_cache       → /root/.EasyOCR
huggingface_cache   → /root/.cache/huggingface
```
MegaDetector and SpeciesNet weights are stored inside the container layer on first build.

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
                                    core/ (MDv5a + MDv1000 + BioClip + SpeciesNet)
                                         ↓
                                    wildlife_data.db (SQLite, WAL mode)
```

---

## Configuration

All runtime settings are in the left sidebar. Changes POST to `PATCH /api/config` and take effect on the next processing run — no restart needed.

| Setting | Description |
|---------|-------------|
| Detection Confidence | Score cutoff (default 0.35). Lower for dark/distant shots. |
| Brightness Threshold | Day/Night classification sensitivity (0–255). |
| Metadata Strip (%) | % of image bottom scanned for date/time OCR text. |
| Auto-Scrub Person/Vehicle | Apply Gaussian blur to privacy-sensitive bounding boxes. |
| Blur Strength | Gaussian kernel size (11–101, odd numbers). |
| Independence Window (min) | Same species + station within this window = one IDE (default 30 min). |
| Default Station ID | Fallback when filename doesn't encode a station. |
| Default Trap Nights | Used for RAI when no deployment records exist. |
| Review Queue Threshold | Images below this confidence go to the Review Queue. |
| Reviewer ID | Name/ID logged against review actions. |
| Low-Spec Mode | Disables MDv1000 + SpeciesNet; INT8-quantizes MDv5a + BioClip. Keeps RAM under ~2 GB. |
| CPU Threads (Windows only) | PyTorch intra-op threads (default ¼ of core count). |

---

## Performance & System Requirements

### Model memory footprint

All models load once at startup and stay resident. The ensemble runs four models simultaneously.

| Component | RAM | Disk | Notes |
|-----------|-----|------|-------|
| PyTorch runtime | ~1.2 GB | ~2 GB | Shared across all models |
| MegaDetector V5a | ~600 MB | ~600 MB | Always loaded |
| MegaDetector V1000 | ~600 MB | ~600 MB | Skipped in Low-Spec Mode |
| BioClip (OpenCLIP) | ~850 MB | ~850 MB | Always loaded |
| SpeciesNet (EfficientNetV2-M) | ~1.0 GB | ~1.0 GB | Skipped without Kaggle credentials or in Low-Spec Mode |
| EasyOCR | ~200 MB | ~200 MB | Always loaded |
| **Full ensemble total** | **~4.5 GB** | **~5.5 GB** | |
| **Low-Spec total** | **~2.3 GB** | **~3.5 GB** | MDv1000 + SpeciesNet disabled |

### Minimum recommended specs

| Resource | Low-Spec Mode | Full Ensemble |
|----------|--------------|---------------|
| RAM | 8 GB | 16 GB |
| CPU | 2-core | 4+ cores (parallel inference benefits significantly) |
| GPU | Not required | CUDA GPU speeds up BioClip + SpeciesNet |
| Disk | 5 GB free | 10 GB free |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | — |

> GPU detection is automatic on all platforms. AMD and Intel GPUs are not CUDA-capable and always fall back to CPU.

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

## Troubleshooting: SpeciesNet Not Loading

**Symptom:** Startup log shows `"SpeciesNet failed to load"` and the live panel shows `"not loaded"` for SpeciesNet predictions.

**Cause:** Kaggle credentials are missing or incorrect.

**Fix:**
```bash
# 1. Verify credentials exist in .env
cat .env | grep KAGGLE

# 2. If missing, add them:
echo "KAGGLE_USERNAME=your_username" >> .env
echo "KAGGLE_KEY=your_api_key" >> .env

# 3. Restart the backend
```

The pipeline continues to work without SpeciesNet — it falls back to BioClip only. Agreement will show "Medium" instead of "High" for most detections.

---

## Troubleshooting: Models Not Loading

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
pip install megadetector open_clip_torch speciesnet
python force_download.py
```

---

## Troubleshooting: No Animals Detected

### Step 1 — Check model status

```
GET http://localhost:8000/api/config/status
```

Should return `{"models_loaded": true}`.

### Step 2 — Watch the live panel

The **Upload & Process** page streams per-model output cards as each image is analysed. If both MDv5a and MDv1000 show `—` (no detections), the animal was not detected by either detector regardless of threshold.

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

## Troubleshooting: Linux — venv Creation Fails

**Symptom:** `ensurepip` error on Ubuntu 25.04+ where system Python is 3.14.

```bash
sudo apt-get install python3.12-venv
rm -rf venv
bash install.sh
```

---

## Troubleshooting: Docker — Permission Denied

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
sudo chown root:docker /var/run/docker.sock
newgrp docker
```

---

## Troubleshooting: Docker — Port 8000 Already in Use

```bash
lsof -ti :8000 | xargs kill
docker compose up
```

Or change the Docker port mapping in `docker-compose.yml` to `"8080:8000"`.

---

## Troubleshooting: Windows — Machine Restarts During Processing

This is a GPU driver TDR failure. The app probes for NVIDIA GPUs via `nvidia-smi` before loading any model, and sets `CUDA_VISIBLE_DEVICES=-1` if no healthy GPU is found. Update to the latest version if you see this on an older install:

```bat
git pull && install.bat
```

---

## Technical Notes

### Pipeline

- **MegaDetector V5a + V1000** run in parallel threads per image. Bounding boxes are NMS-merged (IoU ≥ 0.5).
- **BioClip + SpeciesNet** run in parallel threads per animal crop. Scores are fused with weights 0.45/0.55 (SpeciesNet weighted higher due to camera-trap training data). Agreement bonus of +0.08 applied when both classifiers pick the same top species.
- Images upload automatically in the browser (400 ms debounce) on file selection — the server receives files before the user clicks "Start".
- SSE stream (`GET /api/images/job/{id}/stream`) emits both `model_event` messages (per model per image) and `progress` heartbeats. The frontend renders each image's results as a live card as they arrive.
- All processing runs in a `ThreadPoolExecutor` background task to keep the FastAPI event loop free.

### Data

- **EasyOCR** reads date, time, and temperature from camera metadata strips (bottom 10% of image) via regex.
- **Independence rule** — same species + same station + detections within the window = one IDE. RAI = IDEs / trap nights.
- **Privacy scrubbing** — Gaussian blur applied to Person/Vehicle bounding boxes. Original files are never modified.
- **SQLite** (`wildlife_data.db`) in WAL mode — all stations, IDEs, review actions, community observations, project config, and ArcGIS sync log stored in one local file.
- **Species library** — 159 African wildlife species with full scientific names (e.g. *Panthera leo*), family, order, IUCN status, and 59 synonym mappings (e.g. "painted wolf" → "African Wild Dog").

---

## Recent Changes

### Multi-Model Ensemble Pipeline (May 2026)

**Core:**
- `core/ensemble_engine.py` (new) — IoU-based NMS to merge MDv5a + MDv1000 detections; weighted score fusion (0.45/0.55) for BioClip + SpeciesNet species predictions; agreement scoring (High / Medium / Low) with +0.08 confidence bonus on agreement.
- `core/speciesnet_classifier.py` (new) — thin wrapper around Google's `SpeciesNetClassifier` (EfficientNetV2-M); saves crops to temp JPEG for the filepath-based API, classifies, cleans up.
- `core/animal_detector.py` — `MegaDetectorWrapper` now accepts `model_version` param, supporting any MD model string. `AnimalDetector` accepts `megadetector_v1000` and `speciesnet` as optional second detector/classifier. Both detector pairs and classifier pairs run in parallel via `ThreadPoolExecutor`. Result dicts carry a `_model_events` list for SSE streaming.
- `core/species_library.py` — expanded from 33 Gambella-specific entries to **159 pan-African species** with full scientific names, families, IUCN status, and **59 synonym mappings**.

**Backend:**
- `backend/services/job_manager.py` — `Job` dataclass gains `model_events: List[Dict]` queue.
- `backend/routers/images.py` — SSE stream tracks an event cursor and yields `model_event` messages (per-model per-image) before each `progress` heartbeat. `_run_processing` extracts `_model_events` from each result and appends them to `job.model_events`.
- `backend/models/state.py` — adds `md_v1000_model` and `speciesnet_model` fields.
- `backend/main.py` — loads MDv1000 (redwood) and SpeciesNet in `_load_all_models()`. Both are skipped when `enable_low_spec=True`. `pin_memory` PyTorch warning suppressed.
- `backend/requirements.txt` — adds `speciesnet>=4.0.2`.

**Frontend:**
- `frontend/src/pages/Upload.tsx` — files auto-upload with 400 ms debounce on select/drop. "Start AI Analysis" button only calls `startProcessing()` (files already on server). Live model output panel replaces the plain text log with per-image cards showing MDv5a → MDv1000 → Detection fusion → BioClip → SpeciesNet → final result with agreement badge.

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
