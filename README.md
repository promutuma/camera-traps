# Wildlife Camera Trap Auto-Analyzer

A **FastAPI + React platform** for automated analysis of wildlife camera trap images, designed for the **Gambella Wetland Landscape Baseline Survey** and similar conservation programmes.

The system runs a full multi-model AI ensemble pipeline — OCR metadata extraction, parallel animal detection (**MegaDetector V5a + V1000**), species identification (**BioCLIP + SpeciesNet**), taxonomy-aware detection fusion, independent detection event (IDE) computation, QC flagging, privacy scrubbing, and spatial export — all through a modern browser-based dashboard with real-time per-model output streaming.

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

**Source:** [agentmorris/MegaDetector](https://github.com/agentmorris/MegaDetector), v1000.0 release  
**Architecture:** YOLOv5x6 (larger and newer than V5a)  
**Install:** Same `megadetector` package. Weights (~600 MB) downloaded automatically from GitHub releases on first run.  
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

### Model 3 — BioCLIP (Classifier)

**Source:** [Imageomics/bioclip](https://github.com/Imageomics/BioCLIP) — OpenCLIP foundation model trained on the tree of life  
**Architecture:** Vision Transformer (ViT), CLIP-based  
**Install:** `pip install open_clip_torch` (weights auto-download, ~850 MB)

BioCLIP is a **zero-shot foundation model** trained on the entire tree of life. Unlike traditional classifiers with fixed categories, it links visual features to biological taxonomy — meaning it understands hierarchical relationships between species (lion → Felidae → Carnivora → Mammalia) rather than treating them as isolated labels.

**What sets BioCLIP apart:**

The pipeline uses BioCLIP's full taxonomic output via `predict_taxonomy()`, which runs a single image forward pass and produces:
- **Species-level predictions** — top-k species with confidence scores from the 129-species African wildlife list
- **Full taxonomic path** — for each prediction: `Mammalia > Carnivora > Felidae > Panthera > leo`
- **Family-level classification** — an independent classification against 31 family-level text prompts (e.g. `"a photo of a wild cat such as a lion, leopard, cheetah…"`) that cross-checks the species prediction at a coarser level

This dual output — species prediction + independent family prediction — is what enables reliable **taxonomy-aware agreement** with SpeciesNet (see ensemble section below).

**What it needs to run:**
- `open_clip_torch` package
- ~850 MB disk + ~850 MB RAM
- GPU recommended but not required (CPU works, just slower)

**How it receives input:** The pipeline extracts a padded crop (10% padding around the bounding box) and passes it as a PIL image.

---

### Model 4 — SpeciesNet (Classifier)

**Source:** [google/cameratrapai](https://github.com/google/cameratrapai)  
**Architecture:** EfficientNetV2-M (CNN) trained on **65 million** human-verified camera trap images  
**Install:** `pip install speciesnet`  
**Model download:** Requires **Kaggle credentials** (free account) — see setup below.

SpeciesNet is Google's purpose-built camera trap classifier, developed in partnership with Wildlife Insights. It is the specialist of the ensemble: while BioCLIP understands broad biological relationships across 450,000+ taxa, SpeciesNet was trained specifically on camera trap conditions — poor lighting, awkward angles, nocturnal infrared shots, partial visibility, and motion blur — making it the most accurate classifier for common field species.

**SpeciesNet's key advantages for camera trap data:**

| Capability | Detail |
|---|---|
| **Blank detection** | 98.7% accuracy at identifying empty frames (wind, shadows, vegetation). Classifies these as `"blank"` before any species ID begins — allowing bulk-deletion of thousands of false triggers |
| **Camera trap optimised** | Trained on 65M field images across diverse global ecosystems; outperforms general-purpose models on nocturnal and low-quality shots |
| **Rich taxonomy output** | Each prediction includes scientific name, common name, and full hierarchical path (e.g. `Animalia > Chordata > Mammalia > Carnivora > Felidae > Panthera > leo`) |
| **Night specialisation** | Explicitly trained on IR/night-vision imagery — the ensemble gives SpeciesNet higher weight at night |

**SpeciesNet label format** (raw output, one record per prediction):
```
taxon_id ; Animalia ; Chordata ; Mammalia ; Carnivora ; Felidae ; Panthera ; leo ; lion
```

The pipeline parses this into a structured dict with `common_name`, `scientific_name`, and `hierarchy` fields.

**What it needs to run:**
- `speciesnet` Python package
- A free [Kaggle account](https://www.kaggle.com/) with API credentials
- Model weights downloaded automatically from Kaggle on first run (~220 MB)
- ~1 GB RAM

**Kaggle credentials setup:**

```bash
# 1. Create a free account at https://www.kaggle.com/
# 2. Go to: Account → Settings → API → Create New Token
#    This downloads a kaggle.json file.
# 3. Add to your .env file (in the project root):
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

If credentials are absent, SpeciesNet logs a warning at startup and the pipeline continues with BioCLIP only — no crash, no manual intervention needed.

---

### BioCLIP vs SpeciesNet — When Each Model Wins

| | BioCLIP | SpeciesNet |
|---|---|---|
| **Architecture** | Vision Transformer (CLIP-based) | CNN (EfficientNetV2-M) |
| **Organism scope** | 450,000+ taxa (animals, plants, fungi, insects) | ~2,500 categories (mammals, birds, reptiles) |
| **Training data** | 10M–200M diverse biological images | 65M+ human-labelled camera trap images |
| **Zero-shot** | Yes — can identify species never seen in training | No — fixed category ceiling |
| **Blank detection** | Poor (classifies every frame) | Excellent (98.7% accuracy) |
| **Night/IR images** | Not optimised | Explicitly trained on nocturnal field imagery |
| **Taxonomy output** | Full path with genus/family/order | Full path via label hierarchy |
| **Best for** | Unusual or rare species, taxonomic fallback, cross-checking | High-volume common species ID, blank filtering |

**Design implication:** SpeciesNet is the dominant species signal (95% default weight). BioCLIP contributes a 5% minority vote as a zero-shot cross-check and provides the taxonomic path used for agreement scoring — particularly valuable when SpeciesNet encounters a rare or out-of-distribution species. Both weights are configurable in the sidebar.

---

### How the ensemble combines all four models

The pipeline runs a two-stage ensemble architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: DETECTION                        │
│                                                             │
│  Full image → MDv5a    ────────┐                            │
│                                ├──► NMS Fusion              │
│  Full image → MDv1000  ────────┘    (IoU ≥ 0.5)             │
│                                        │                    │
│                              merged bounding boxes          │
└─────────────────────────────────────────────────────────────┘
                                         │  Animal boxes only
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│               STAGE 2: CROP + CLASSIFY (parallel)           │
│                                                             │
│  Crop + 10% padding                                         │
│    → BioCLIP  predict_taxonomy() → species + family + path  │
│    → SpeciesNet classify_crop()  → label JSON with          │
│                                    common_name + hierarchy  │
│                                                             │
│  Fusion (day):   0.05 × BioCLIP + 0.95 × SpeciesNet        │
│  Fusion (night): 0.02 × BioCLIP + 0.98 × SpeciesNet        │
│  Bypass: if SpeciesNet top ≥ threshold → skip fusion        │
│                                                             │
│  Agreement bonus (taxonomy-aware):                          │
│    +0.08 if BioCLIP genus appears in SpeciesNet hierarchy   │
│    +0.04 if BioCLIP family matches SpeciesNet hierarchy     │
└─────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   FINAL OUTPUT per detection                 │
│                                                             │
│  species     = "Lion"                                       │
│  confidence  = 0.93                                         │
│  agreement   = "High"   ← genus Panthera in SN hierarchy   │
│  breakdown   = { MDv5a: …, MDv1000: …,                     │
│                  BioCLIP: …, SpeciesNet: … }                │
└─────────────────────────────────────────────────────────────┘
```

#### Stage 1: Bounding Box Fusion (NMS)

1. **Detections:** Bounding boxes from **MegaDetector V5a** and **MegaDetector V1000** are collected concurrently.
2. **NMS Merge:** Non-Maximum Suppression merges overlapping boxes (IoU ≥ 0.5):
   $$\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$
3. **Geometry Selection:** If both models detect the same animal, the higher-confidence box geometry is kept.

#### Stage 2: Taxonomy-Aware Classifier Fusion

For every merged box classified as `"Animal"`, the padded crop runs through **BioCLIP** and **SpeciesNet** in parallel:

**1. Dynamic weights based on time of day:**

| Condition | BioCLIP weight | SpeciesNet weight | Reason |
|---|---|---|---|
| Day (colour image) | 0.05 | 0.95 | SpeciesNet dominant; BioCLIP as minority cross-check |
| Night / IR image | 0.02 | 0.98 | SpeciesNet explicitly trained on nocturnal camera trap imagery; BioCLIP not optimised for IR |

Day/night is determined before classification via `DayNightClassifier`, which detects grayscale/low-saturation images as night-vision regardless of brightness.

**1b. SpeciesNet bypass:**

When SpeciesNet's top confidence exceeds the bypass threshold (default 0.60), the fusion step is skipped entirely and SpeciesNet's prediction is used directly (with an agreement bonus added if BioCLIP concurs at genus/family level). This avoids the rare case where weighted averaging with a low-confidence BioCLIP prediction pulls the final score down on images where SpeciesNet is already highly confident.

Set the bypass threshold to 0 in the **Classifier Fusion** sidebar section to always fuse.

**2. Taxonomy-aware agreement detection:**

Rather than word-matching, agreement is computed by comparing BioCLIP's full taxonomic output against SpeciesNet's hierarchy:

| Level | Agreement | Bonus | Example |
|---|---|---|---|
| Genus matches SN hierarchy | High | +0.08 | BioCLIP: *Panthera* leo → SN hierarchy contains *Panthera* |
| Species common name matches | High | +0.08 | BioCLIP: "lion" matches SN display "lion" |
| Family matches SN hierarchy | Medium | +0.04 | Both predict Felidae even if species differ |
| Independent family prediction matches | Medium | +0.04 | BioCLIP family features confirm same family |
| Order matches SN hierarchy | Medium | +0.04 | Both predict Carnivora |
| No shared taxon | Low | +0.00 | Flagged for human review |

BioCLIP's `predict_taxonomy()` computes two independent signals in a single forward pass:
- Species-level scoring against the 129-species African wildlife list
- Family-level scoring against 31 family text prompts (e.g. `"a photo of a wild cat such as a lion, leopard, cheetah…"`)

The family-level prediction serves as an independent cross-check — if both the species prediction and the independent family prediction map to the same family as SpeciesNet's hierarchy, agreement is elevated to Medium even if species labels differ.

---

## Features

| Tab | Feature |
|-----|---------|
| Upload & Process | Auto-upload on file selection; real-time per-model output panel (MDv5a · MDv1000 · BioCLIP · SpeciesNet · Fusion); SSE progress stream |
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
│   │   ├── images.py             # Upload, processing, SSE stream; MIME validation
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
│       ├── App.tsx               # React Router with 15 routes; Error Boundary on each
│       ├── components/
│       │   └── ErrorBoundary.tsx # Class-based Error Boundary — catches render errors
│       ├── api/client.ts         # Typed axios client
│       ├── store/configStore.ts  # Zustand global config store
│       └── pages/
│           ├── Upload.tsx        # Auto-upload + live model output panel
│           └── …                 # 14 other page components
│
├── core/                         # AI/ML business logic
│   ├── animal_detector.py        # MegaDetectorWrapper (v5a + v1000) + AnimalDetector
│   │                             # orchestrator; parallel inference; is_night → fuse_species
│   ├── ensemble_engine.py        # NMS detection fusion + taxonomy-aware species fusion
│   │                             # Dynamic weights (day 0.05/0.95, night 0.02/0.98) + bypass
│   ├── speciesnet_classifier.py  # Google SpeciesNet wrapper; parses full taxonomy JSON
│   ├── bioclip_classifier.py     # BioCLIP zero-shot classifier with predict_taxonomy():
│   │                             # species path + independent family-level cross-check
│   ├── species_library.py        # 159-species African wildlife DB + synonyms
│   ├── day_night_classifier.py   # Brightness + saturation day/night/IR detection
│   ├── ocr_processor.py          # EasyOCR metadata extraction
│   ├── image_processor.py        # Unified pipeline; passes is_night to detector
│   ├── db_manager.py             # SQLite schema, WAL mode, job persistence table
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
python force_download.py          # pre-downloads MDv5a + BioCLIP (~1.5 GB)
```

### Step 2 — Set up SpeciesNet credentials (optional but recommended)

SpeciesNet downloads its weights from Kaggle on first run. Without credentials the pipeline falls back to BioCLIP-only — still functional, just less accurate on common species and unable to detect blanks with high confidence.

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
wildlife_data       → /app/data  (SQLite DB + uploaded images)
```
MegaDetector and SpeciesNet weights are cached at `/tmp/megadetector_models/` and `~/.cache/kagglehub/` respectively.

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
                                    core/ (MDv5a + MDv1000 + BioCLIP + SpeciesNet)
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
| **BioClip Weight** | BioCLIP contribution in species fusion (default 0.05). |
| **SpeciesNet Weight** | SpeciesNet contribution in species fusion (default 0.95). |
| **SpeciesNet Bypass Threshold** | Skip fusion when SpeciesNet top confidence ≥ this value (default 0.60). Set to 0 to always fuse. |
| Auto-Scrub Person/Vehicle | Apply Gaussian blur to privacy-sensitive bounding boxes. |
| Blur Strength | Gaussian kernel size (11–101, odd numbers). |
| Independence Window (min) | Same species + station within this window = one IDE (default 30 min). |
| Default Station ID | Fallback when filename doesn't encode a station. |
| Default Trap Nights | Used for RAI when no deployment records exist. |
| Review Queue Threshold | Images below this confidence go to the Review Queue. |
| Reviewer ID | Name/ID logged against review actions. |
| Low-Spec Mode | Disables MDv1000 + SpeciesNet; INT8-quantizes MDv5a + BioCLIP. Keeps RAM under ~2 GB. |
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
| BioCLIP (OpenCLIP) | ~850 MB | ~850 MB | Always loaded; family features add ~1 MB |
| SpeciesNet (EfficientNetV2-M) | ~1.0 GB | ~220 MB | Skipped without Kaggle credentials or in Low-Spec Mode |
| EasyOCR | ~200 MB | ~200 MB | Always loaded |
| **Full ensemble total** | **~4.5 GB** | **~5.5 GB** | |
| **Low-Spec total** | **~2.3 GB** | **~3.5 GB** | MDv1000 + SpeciesNet disabled |

### Minimum recommended specs

| Resource | Low-Spec Mode | Full Ensemble |
|----------|--------------|---------------|
| RAM | 8 GB | 16 GB |
| CPU | 2-core | 4+ cores (parallel inference benefits significantly) |
| GPU | Not required | CUDA GPU speeds up BioCLIP + SpeciesNet |
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

The pipeline continues to work without SpeciesNet — it falls back to BioCLIP only. Agreement will show "Medium" for most detections and blank detection accuracy will be reduced.

---

## Troubleshooting: MDv1000 Not Loading

**Symptom:** Startup log shows `Error loading redwood: <urlopen error [Errno 111] Connection refused>`

**Cause:** An older version of the `megadetector` pip package ships with a placeholder URL (`http://localhost:8181/`) for v1000 model weights instead of the real GitHub releases URL. The app patches this automatically at startup, but if you installed from a very old package, the patch may not apply cleanly.

**Fix:**
```bash
pip install --upgrade megadetector
```

The real download URL is `https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-redwood.pt`.

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
- **Day/night classification** runs before species classification; `DayNightClassifier` detects grayscale/low-saturation images as night-vision regardless of pixel brightness.
- **BioCLIP `predict_taxonomy()`** performs a single image forward pass scoring against both the 129-species list and 31 family-level prompts. Returning: top species, full taxonomic path (`Mammalia > order > family > genus > epithet`), and an independent family prediction.
- **SpeciesNet** classifies the same crop via its filepath-based API (crop saved to temp JPEG, classified, file deleted). Returns JSON labels with `common_name`, `scientific_name`, and `hierarchy`.
- **Fusion weights** are dynamic and configurable: defaults are `(0.05 BioCLIP, 0.95 SpeciesNet)` for day images; `(0.02 BioCLIP, 0.98 SpeciesNet)` at night. When SpeciesNet's top score exceeds the bypass threshold (default 0.60), fusion is skipped entirely and SpeciesNet's result is used directly. All three parameters are adjustable live from the sidebar without restart.
- **Agreement** is computed by comparing BioCLIP's genus/family against SpeciesNet's taxonomy hierarchy — not word matching. High (+0.08) for genus match; Medium (+0.04) for family match.
- Images upload automatically in the browser (400 ms debounce) on file selection — the server receives files before the user clicks "Start".
- SSE stream (`GET /api/images/job/{id}/stream`) emits both `model_event` messages (per model per image) and `progress` heartbeats.
- **Three-layer threading model:**
  - *Startup* — `asyncio.to_thread` offloads model loading (~4.5 GB) so the FastAPI event loop stays responsive during the multi-minute boot.
  - *Per-job* — FastAPI `BackgroundTasks` runs `_run_processing` in a starlette threadpool thread. A `threading.Semaphore(1)` limits concurrent jobs to one, preventing OOM from overlapping ML workloads.
  - *Per-image* — a `ThreadPoolExecutor(_PARALLEL_IMAGES)` processes multiple images concurrently within the job. `_PARALLEL_IMAGES = max(1, min(4, cpu_count // 2))` so it scales with hardware.
  - *Per-model* — a module-level `ThreadPoolExecutor(max(4, cpu_count // 2))` in `animal_detector.py` runs MDv5a ∥ MDv1000 (Stage 1) and BioCLIP ∥ SpeciesNet (Stage 2) in parallel.
  - *Within-image* — OCR and day/night classification are submitted to `_pipeline_executor` simultaneously; both complete before detection starts (detection needs the `is_night` flag). EasyOCR calls are serialised via `_ocr_lock` because `Reader.readtext()` has shared internal buffers.
- `JobManager` uses a `threading.Lock` to protect the `_jobs` dict from concurrent reads and writes across HTTP handler threads and background processing threads.
- Completed jobs are persisted to a `jobs` table in SQLite so job metadata survives server restarts.
- File uploads are validated by magic bytes (JPEG, PNG, TIFF, BMP, WebP) — not just MIME type headers — before being saved.

### Data

- **EasyOCR** reads date, time, and temperature from camera metadata strips (bottom 10% of image) via regex.
- **Independence rule** — same species + same station + detections within the window = one IDE. RAI = IDEs / trap nights.
- **Privacy scrubbing** — Gaussian blur applied to Person/Vehicle bounding boxes. Original files are never modified.
- **SQLite** (`wildlife_data.db`) in WAL mode — all stations, IDEs, review actions, community observations, project config, job metadata, and ArcGIS sync log stored in one local file.
- **Species library** — 159 African wildlife species with full scientific names (e.g. *Panthera leo*), family, order, IUCN status, and 59 synonym mappings (e.g. "painted wolf" → "African Wild Dog").
- **`SPECIES_TAXONOMY`** — compile-time taxonomy table covering all 129 `WILDLIFE_CLASSES` entries, mapping common name → order / family / genus / scientific name. Used by BioCLIP to build taxonomic paths and by the ensemble for family/genus-level agreement scoring.

---

## Recent Changes

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

### Classifier Fusion Tuning + Model Breakdown Display (May 2026)

**Ensemble weight rebalancing (`core/ensemble_engine.py`):**
- `_DEFAULT_WEIGHTS` revised to `(0.05, 0.95)` (BioCLIP / SpeciesNet) and `_NIGHT_WEIGHTS` to `(0.02, 0.98)`. Field observation confirmed SpeciesNet is consistently more accurate on camera trap images; BioCLIP now acts as a minority cross-check rather than an equal partner.
- Added `speciesnet_bypass_threshold` parameter to `fuse_species()`. When SpeciesNet's top-1 score ≥ threshold, the weighted average step is skipped and SpeciesNet's prediction is returned directly (plus agreement bonus when BioCLIP concurs at genus/family level). Avoids artificially dragging down a high-confidence SpeciesNet result.

**Configurable fusion weights (`core/animal_detector.py`, `backend/`):**
- `AnimalDetector.__init__` accepts `bioclip_weight`, `speciesnet_weight`, and `speciesnet_bypass_threshold`.
- `AppConfig` and `ConfigResponse` / `ConfigUpdate` schemas expose all three fields.
- `_run_processing` in `backend/routers/images.py` passes live config values to `AnimalDetector` so weight changes take effect on the next processing run without restart.

**Settings UI — Classifier Fusion section (`frontend/src/components/Layout/Sidebar.tsx`, `store/configStore.ts`):**
- Added "Classifier Fusion" settings section to the sidebar with three sliders: BioClip Weight (0–1), SpeciesNet Weight (0–1), and SpeciesNet Bypass Threshold (0 = always fuse, >0 = skip fusion above that confidence).
- `AppConfig` TypeScript interface updated with all three new fields plus previously missing `speciesnet_lat`, `speciesnet_lng`, `speciesnet_country`.

**Full model breakdown display (`frontend/src/pages/ReviewQueue.tsx`, `Results.tsx`):**
- Added `FullModelBreakdown` component in Review Queue showing ranked top-3 predictions for Object Detector (MDv5a / MDv1000), BioCLIP, SpeciesNet, and Fusion with confidence bars.
- For records without `model_breakdown` data (processed before this fix), the component falls back to displaying the stored scalar `bioclip_confidence` / `speciesnet_confidence` as a top-1 entry.
- Results gallery card now passes `model_breakdown` to the expanded "Show Details" view.

**model_breakdown data-loss fix (`core/image_processor.py`):**
- `process_single_image()` was computing `model_breakdown` inside the detector but never copying it to the result row before DB insert — it was always stored as `NULL`. Added `row['model_breakdown'] = det.get('model_breakdown')` to close this gap. Newly processed images will have full per-model ranked predictions in the UI.

---

### Taxonomy-Aware Ensemble + ML Quality (May 2026)

**BioCLIP (`core/bioclip_classifier.py`):**
- Added `SPECIES_TAXONOMY` — a compile-time table mapping all 129 `WILDLIFE_CLASSES` entries to their order, family, genus, and scientific name.
- Added `FAMILY_PROMPTS` — 31 family-level natural language prompts (e.g. `"a photo of a wild cat such as a lion, leopard, cheetah…"`) for independent family classification.
- Added `_compute_family_features()` — pre-computes family text embeddings alongside species embeddings at startup.
- Added `_encode_image()` — single image forward pass whose result is shared by both scoring passes.
- Added `predict_taxonomy()` — replaces `predict_list()` for ensemble use. Single forward pass; scores against both species and family embeddings; returns `top`, `candidates`, `family_prediction`, `family_confidence`.

**Ensemble (`core/ensemble_engine.py`):**
- Replaced flat word-matching agreement with `_taxonomy_agreement_structured()` — compares BioCLIP's genus/family/order against SpeciesNet's parsed hierarchy. High agreement fires correctly when BioCLIP predicts "lion" and SpeciesNet returns the genus *Panthera*.
- `_DEFAULT_WEIGHTS` set to `(0.40, 0.60)` — SpeciesNet carries more weight overall (later revised, see below).
- Added `_NIGHT_WEIGHTS = (0.25, 0.75)` — SpeciesNet dominant at night (trained on IR/nocturnal imagery; BioCLIP is not).
- `fuse_species()` gains `is_night: bool` and `bioclip_taxonomy: Optional[Dict]` parameters.

**Pipeline (`core/image_processor.py`, `core/animal_detector.py`):**
- `ImageProcessor.process_single_image()` passes `is_night` (from day/night result) into `detector.detect()`.
- `AnimalDetector._classify_parallel()` now calls `predict_taxonomy()` and returns `(bc_pairs, sn_pairs, bc_taxonomy)`.
- `fuse_species()` receives the full BioCLIP taxonomy dict for structured comparison.

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

### MDv1000 / Redwood Model Fix (May 2026)

- The `megadetector` pip package (≤ 5.0.29) ships with `localhost:8181` as the download URL for all v1000 model variants — a test-only placeholder. The app now patches `megadetector.detection.run_detector.known_models` at startup to use the real GitHub releases URL (`https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/`).
- Added proper metadata (`typical_detection_threshold: 0.3`, `image_size`, `model_type`) to all v1000 entries.

---

### Multi-Model Ensemble Pipeline (May 2026)

**Core:**
- `core/ensemble_engine.py` (new) — IoU-based NMS to merge MDv5a + MDv1000 detections; weighted score fusion for BioCLIP + SpeciesNet species predictions; agreement scoring (High / Medium / Low) with confidence bonus.
- `core/speciesnet_classifier.py` (new) — thin wrapper around Google's `SpeciesNetClassifier`; saves crops to temp JPEG for the filepath-based API, classifies, cleans up.
- `core/animal_detector.py` — `MegaDetectorWrapper` now accepts `model_version` param. `AnimalDetector` accepts `megadetector_v1000` and `speciesnet` as optional second detector/classifier. Both detector pairs and classifier pairs run in parallel via `ThreadPoolExecutor`. Result dicts carry a `_model_events` list for SSE streaming.
- `core/species_library.py` — expanded from 33 Gambella-specific entries to **159 pan-African species** with full scientific names, families, IUCN status, and **59 synonym mappings**.

**Backend:**
- `backend/services/job_manager.py` — `Job` dataclass gains `model_events: List[Dict]` queue.
- `backend/routers/images.py` — SSE stream tracks an event cursor and yields `model_event` messages (per-model per-image) before each `progress` heartbeat.
- `backend/models/state.py` — adds `md_v1000_model` and `speciesnet_model` fields.
- `backend/main.py` — loads MDv1000 (redwood) and SpeciesNet in `_load_all_models()`. Both are skipped when `enable_low_spec=True`.
- `backend/requirements.txt` — adds `speciesnet>=4.0.2`.

**Frontend:**
- `frontend/src/pages/Upload.tsx` — files auto-upload with 400 ms debounce on select/drop. Live model output panel with per-image cards showing MDv5a → MDv1000 → Detection fusion → BioCLIP → SpeciesNet → final result with agreement badge.

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
