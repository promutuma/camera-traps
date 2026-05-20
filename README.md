# Wildlife Camera Trap Auto-Analyzer

A **Streamlit-based platform** for automated analysis of wildlife camera trap images, designed for the **Gambella Wetland Landscape Baseline Survey** and similar conservation programmes.

The system handles the full pipeline from raw images to publication-ready outputs: OCR metadata extraction, AI animal detection (**MegaDetector V5a**), species identification (**BioClip**), independent detection event (IDE) computation, QC flagging, privacy scrubbing, and spatial export — all through a browser-based dashboard.

![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![MegaDetector](https://img.shields.io/badge/MegaDetector-V5a-blue)
![BioClip](https://img.shields.io/badge/BioClip-Enabled-green)
![Python](https://img.shields.io/badge/Python-3.11--3.13-blue)

---

## Features

| Tab | Feature |
|-----|---------|
| Upload & Process | Batch upload, OCR, MegaDetector + BioClip pipeline |
| Review Results | Gallery / Inspector view with bounding boxes and editable species labels |
| Statistics | Per-species counts, confidence distributions, day/night breakdown |
| History & Analytics | Long-term trends from the SQLite database |
| Diagnostics | OCR strip debugger, raw model output viewer |
| Ecological Analytics | IDE computation, RAI, species richness, accumulation curve, visitation rate |
| QC Dashboard | 7 automated quality-control checks with colour-coded flags |
| Stations & Deployments | Camera registry, GPS coordinates, deployment history, trap-night calculator |
| Review Queue | HITL expert review — accept / correct / reject with reviewer logging |
| Community Observer | Field observer sighting entry, cross-verification against camera data |
| Spatial & Map | Interactive pydeck map, GeoJSON export, georeferenced CSV |
| Species Library | 34 pre-loaded Gambella mammals, synonym resolver, IUCN status |
| Corridor Analysis | Directional flow detection, passage frequency, bottleneck identification |
| Project Config | Multi-project support, indicator thresholds, baseline locking, JSON export |
| ArcGIS / Spatial Exports | Offline file exports (GeoJSON, Shapefile, KML) + live push to ArcGIS Online / Enterprise |

---

## Project Structure

```
camera-traps/
├── app.py                        # Main Streamlit application (15 tabs)
├── requirements.txt              # pip dependencies
├── environment.yml               # Conda environment spec
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Container orchestration
├── install.bat                   # Windows one-click installer
├── install.sh                    # Mac / Linux one-click installer
├── run.bat                       # Windows launcher (after install)
├── run.sh                        # Mac / Linux launcher (after install)
├── force_download.py             # Pre-downloads AI models (~1.5 GB)
├── wildlife_data.db              # SQLite database (auto-created on first run)
├── .devcontainer/
│   └── devcontainer.json         # VS Code Dev Container / GitHub Codespaces
└── core/
    ├── animal_detector.py        # MegaDetector + BioClip ensemble
    ├── bioclip_classifier.py     # OpenCLIP species classifier
    ├── day_night_classifier.py   # Brightness-based day/night
    ├── ocr_processor.py          # EasyOCR metadata extraction
    ├── image_processor.py        # Unified processing pipeline
    ├── db_manager.py             # SQLite schema and persistence
    ├── independence_engine.py    # 30-min IDE grouping + RAI
    ├── qc_engine.py              # 7-check QC flag system
    ├── station_manager.py        # Station registry + deployments
    ├── privacy_scrubber.py       # Gaussian blur for Person/Vehicle
    ├── review_engine.py          # HITL accept/correct/reject queue
    ├── community_observer.py     # Field observer sighting store
    ├── spatial_exporter.py       # GeoJSON, Shapefile, KML + pydeck layer data
    ├── species_library.py        # 34-species reference library
    ├── corridor_analyzer.py      # Directional corridor flow analysis
    ├── project_config.py         # Multi-project config + baselines
    └── arcgis_sync.py            # ArcGIS REST API sync
```

---

## Installation

> **AI model note:** On first run the app downloads MegaDetector V5a and BioClip (~1.5 GB total). Run `python force_download.py` (or `python3 force_download.py`) **before** launching to do this once in the foreground. The script resumes automatically if interrupted.

Choose the environment that matches your setup:

---

### Option A — Windows (one-click installer)

**Requirements:** Python 3.11 - 3.13 from [python.org](https://www.python.org/downloads/) — check **"Add Python to PATH"** during installation. Optionally, [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for Conda support.

```bat
REM 1. Clone the repository
git clone <repository-url>
cd camera-traps

REM 2. Run the installer (creates venv, installs deps, downloads models)
install.bat

REM 3. Start the app (every subsequent run)
run.bat
```

The installer detects Conda if installed and asks which environment manager you prefer.

**If an existing installation is found**, `install.bat` will ask:
```
An existing installation was found.

  1) Update / repair  - keep the current environment, install missing packages
  2) Fresh install    - wipe everything and start clean
```

To force a fresh install from the command line:
```bat
install.bat --fresh
```

---

### Option B — macOS (one-click installer)

**Requirements:** Python 3.11 - 3.13 (`brew install python@3.12` or `python@3.11`) or [python.org](https://www.python.org/downloads/). Optionally, [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
# 1. Clone the repository
git clone <repository-url>
cd camera-traps

# 2. Run the installer
chmod +x install.sh
./install.sh

# 3. Start the app (every subsequent run)
./run.sh
```

To force a fresh install (wipes the existing venv and re-downloads everything):
```bash
./install.sh --fresh
```

---

### Option C — Linux (one-click installer)

**Requirements:** Python 3.11 - 3.13, `python3-venv`. On Debian/Ubuntu the installer handles system dependencies automatically via `sudo apt-get`.

```bash
# 1. Clone the repository
git clone <repository-url>
cd camera-traps

# 2. Run the installer (installs libgl1, libglib2.0-0 etc. automatically)
chmod +x install.sh
./install.sh

# 3. Start the app (every subsequent run)
./run.sh
```

To force a fresh install:
```bash
./install.sh --fresh
```

---

### Option D — Conda / Miniconda (any platform)

Use this if you prefer Conda for environment management. `environment.yml` is provided.

```bash
# 1. Clone the repository
git clone <repository-url>
cd camera-traps

# 2. Create and activate the Conda environment
conda env create -f environment.yml
conda activate wildlife-analyzer

# 3. Download AI models (one-time)
python force_download.py

# 4. Launch the app
python -m streamlit run app.py
```

To update after a `git pull`:
```bash
conda env update -f environment.yml --prune
```

---

### Option E — Docker (any platform)

Best for reproducible environments and teams. Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
# 1. Clone the repository
git clone <repository-url>
cd camera-traps

# 2. Build and run with Docker Compose (recommended)
docker-compose up --build

# OR using plain Docker
docker build -t wildlife-analyzer .
docker run -p 8501:8501 wildlife-analyzer
```

Access the app at `http://localhost:8501`.

**Persist the database and model cache between runs:**
```bash
docker run -p 8501:8501 \
  -v $(pwd)/wildlife_data.db:/app/wildlife_data.db \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  wildlife-analyzer
```

---

### Option F — VS Code Dev Container

The repository includes a `.devcontainer/devcontainer.json` that configures a Python 3.11 container with all system dependencies pre-installed.

1. Install the **Dev Containers** extension in VS Code.
2. Open the repository folder in VS Code.
3. When prompted, click **"Reopen in Container"** (or press `F1` → *Dev Containers: Reopen in Container*).
4. VS Code builds the container, installs all dependencies, and auto-launches the app on port 8501.
5. The app opens automatically in a browser preview tab.

No manual `pip install` or model downloading required — the container handles everything.

---

### Option G — GitHub Codespaces

Run the app entirely in the cloud — no local installation needed.

1. On the GitHub repository page, click **Code → Codespaces → Create codespace on main**.
2. Codespaces builds the dev container (same config as Option F).
3. When the terminal appears, run the model pre-download:
   ```bash
   python force_download.py
   ```
4. The app starts automatically. Click the **"Open in Browser"** notification or go to the **Ports** tab and open port `8501`.

> **Note:** Codespaces free tier provides 60 hours/month. Model downloads count against storage (up to 15 GB free).

---

### Option H — Google Colab / Jupyter

> **Note:** Streamlit is not natively supported in Colab's output iframe, but you can tunnel to it using `pyngrok`.

```python
# Cell 1 — Install dependencies
!pip install -q streamlit pyngrok
!pip install -q -r requirements.txt

# Cell 2 — Download models
!python force_download.py

# Cell 3 — Launch with ngrok tunnel
from pyngrok import ngrok
import subprocess, threading

def run_streamlit():
    subprocess.run(["python", "-m", "streamlit", "run", "app.py",
                    "--server.port=8501", "--server.headless=true"])

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

# Expose via ngrok (sign up at https://ngrok.com for a free authtoken)
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
public_url = ngrok.connect(8501)
print(f"App running at: {public_url}")
```

---

### Option I — Streamlit Community Cloud

Best for sharing with collaborators without hosting infrastructure.

1. Push the repository to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select your repository, and set the main file to `app.py`.
4. Click **Deploy** — Streamlit Cloud auto-installs from `requirements.txt`.

> **Limitation:** Streamlit Cloud uses ephemeral storage. The SQLite database (`wildlife_data.db`) resets on each deployment. For persistent data, mount an external database or use Streamlit's `st.secrets` with a cloud database.

---

## Configuration

### Sidebar Settings

| Setting | Description |
|---------|-------------|
| Confidence Threshold | Detection score cutoff (default 0.35). Lower = more detections, higher = fewer false positives. |
| Brightness Threshold | Day/Night classification sensitivity. |
| OCR Strip Height | % of image bottom scanned for date/time text. Adjust to match your camera's metadata bar. |
| Privacy Scrubbing | Enable Gaussian blur on Person/Vehicle detections. |
| Blur Strength | Kernel size for privacy blur (odd numbers, 11–101). |
| Independence Window | Minutes between detections of the same species at the same station to count as separate events (default 30 min). |
| Default Station ID | Fallback station ID when filename doesn't encode one. |
| Trap Nights (fallback) | Used for RAI when no deployment records exist. |
| Review Confidence Threshold | Images below this score enter the HITL review queue. |
| Reviewer ID | Name/ID logged against review actions. |
| Reload AI Models | Clears the cached models and reloads from disk. Use this after installing or updating packages without restarting the app. |

---

## Spatial File Exports

The **ArcGIS / Spatial Exports** tab provides three offline formats — no ArcGIS account required:

| Format | File | Best used with |
|--------|------|----------------|
| **GeoJSON** | `.geojson` | ArcGIS Online, QGIS, Mapbox, web apps |
| **Shapefile** | `.zip` (contains `.shp`, `.dbf`, `.shx`, `.prj`) | ArcGIS Pro / Desktop, QGIS, any desktop GIS |
| **KML** | `.kml` | Google Earth, ArcGIS Earth, ArcGIS Pro |

Both **Station Locations** and **Detection Events (IDEs)** can be exported in all three formats.

> **Shapefile note:** Requires the `pyshp` package, included in `requirements.txt`. If you installed manually without `pyshp`, run `pip install pyshp`.

---

## ArcGIS Live Sync Setup

For pushing data directly to a hosted feature layer (no file download needed):

1. In ArcGIS Online, create two hosted **Feature Layers** (Point geometry):
   - **Stations layer** — fields: `station_id`, `habitat_stratum`, `camera_model`, `trap_nights`, `functionality_pct`, `status`
   - **Detections layer** — fields: `ide_id`, `station_id`, `species`, `image_count`, `max_confidence`, `first_detection`, `last_detection`, `duration_minutes`
2. Copy each layer's REST endpoint URL from **Item Details → URL** (ends in `/FeatureServer/0`).
3. In the app's **ArcGIS / Spatial Exports** tab, expand **Connection Settings**, enter your portal URL and credentials or API token, then click **Connect**.
4. Paste the layer URLs and click **Push Stations** / **Push Detections**.

For ArcGIS Enterprise, replace `https://www.arcgis.com` with your portal root (e.g. `https://gis.yourorg.com/portal`).

---

## Performance & System Requirements

### Why does the app feel heavy?

The **AI models are the bottleneck — not Streamlit itself**. Streamlit's web server uses roughly 100–200 MB of RAM. The heaviness comes from:

| Component | RAM | Disk |
|-----------|-----|------|
| PyTorch runtime | ~1.2 GB | ~2 GB |
| MegaDetector V5a | ~600 MB | ~600 MB |
| BioClip (OpenCLIP) | ~850 MB | ~850 MB |
| **Total (first load)** | **~2.6 GB** | **~3.5 GB** |

Once loaded, models are cached in memory (`@st.cache_resource`) so reruns are fast.

### Minimum recommended specs

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | Any modern dual-core | 4+ cores |
| GPU | Not required (CPU-only on Windows) | CUDA GPU speeds up BioClip on Linux/Mac |
| Disk | 5 GB free | 10 GB free |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | — |

> **Windows GPU note:** GPU acceleration is intentionally disabled on Windows. PyTorch's CUDA initialisation probes the GPU driver on startup, which on Windows (especially laptops with Optimus / hybrid graphics or older drivers) can cause an **instant machine restart with no BSOD** — a GPU driver TDR failure. All three models run on CPU on Windows, which is sufficient for camera trap batch processing.

### Running on a low-RAM machine

If you have less than 8 GB RAM, enable **Low-Spec / Low-Memory Mode** in the sidebar. This applies INT8 dynamic quantization to all three models, roughly halving their memory footprint. Detection accuracy may decrease slightly on borderline images.

**Option 2 — Run models on a server, view on laptop**
Deploy the Docker image on a cloud VM (e.g. AWS EC2 `t3.xlarge`, ~16 GB RAM) and access `http://<server-ip>:8501` from any browser. The laptop only runs the browser — all heavy processing happens on the server.

### About Streamlit vs other frameworks

Streamlit is appropriate for this use case (field biologists, not web developers). If you need:
- **Multi-user access with roles** → migrate to Dash (all `core/` modules are framework-agnostic)
- **REST API access** → wrap `core/` modules with FastAPI
- **Maximum performance** → the bottleneck is PyTorch, not the web framework; switching away from Streamlit won't measurably help

---

## VS Code Interpreter Setup

If VS Code shows "Cannot find module" errors for `streamlit`, `cv2`, `torch` etc., the editor is checking the wrong Python. Fix it once:

1. Press `Ctrl+Shift+P` → **Python: Select Interpreter**
2. Choose `./venv/bin/python` (venv) or the `wildlife-analyzer` conda env

All module-not-found errors across every file will disappear immediately.

---

## Troubleshooting: AI Models Not Loading

If the app shows a red banner — **"AI models could not be loaded — animal detection is disabled"** — one or both of the required packages (`megadetector`, `open_clip_torch`) are not installed in the active Python environment.

### Fix

Install the missing packages in your activated environment and re-download the model weights:

```bash
pip install megadetector open_clip_torch
python force_download.py
```

Then click the **Reload Models** button on the banner (or **Reload AI Models** in the sidebar). This clears the cached broken state and reloads the models without restarting the app.

If the packages install but the banner keeps appearing, the app is likely running from a different Python than the one you installed into. Check which Python Streamlit is using:

```bash
# In the terminal where you run the app
python -m streamlit run app.py
```

Make sure you have activated your `venv` (`source venv/bin/activate` / `venv\Scripts\activate.bat`) or conda environment (`conda activate wildlife-analyzer`) first.

### Why this can happen after a successful-looking install

The `megadetector` package uses calendar-based version numbers (e.g. `1.0.0.20240430`). Older versions of `requirements.txt` in this project specified `megadetector>=5.0.0`, which is an unsatisfiable constraint — pip would silently skip the package rather than error. This has been corrected; `requirements.txt` now pins only `megadetector` with no version constraint. Re-running the installer picks up the fix automatically.

---

## Troubleshooting: Windows Machine Restarts Instantly When Processing Images

If the Windows machine reboots with **no Blue Screen of Death (BSOD)** as soon as image processing begins, this is a **GPU driver TDR (Timeout Detection and Recovery) failure** — not a RAM or software crash.

### Why it happens

Even though all models run on CPU, PyTorch calls `torch.cuda.is_available()` during initialisation. On Windows, this call opens a CUDA context against the GPU driver. On machines with:
- Hybrid/Optimus graphics (integrated + discrete GPU)
- Outdated GPU drivers
- Certain NVIDIA driver versions with known TDR bugs

...this driver probe can fail hard enough that Windows cannot recover the GPU driver and reboots instantly instead of showing a BSOD.

This has been fixed in the current version. The app now sets `CUDA_VISIBLE_DEVICES=-1` at startup on Windows, which tells CUDA there are no GPUs — the driver is never touched.

### If you are still seeing restarts on an older version

Update to the latest version of the repository and re-run the installer:

```bat
git pull
install.bat
```

Or apply the fix manually by adding this line to the top of `app.py`, **before any other imports**:

```python
import os, sys
if sys.platform == 'win32':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Other Windows-specific stability fixes included in the current version

| Fix | What it prevents |
|-----|-----------------|
| `CUDA_VISIBLE_DEVICES=-1` | GPU driver TDR crash / instant machine restart |
| `OMP_NUM_THREADS=1` + `KMP_DUPLICATE_LIB_OK=TRUE` | Deadlock from multiple OpenMP runtimes (PyTorch + OpenCV + EasyOCR each load their own) |
| `torch.set_num_threads(1)` on Windows | Thread stack exhaustion when 3 models each spawn `cpu_count` threads simultaneously |
| `multiprocessing.freeze_support()` | Crash when EasyOCR or YOLO spawn worker processes using Windows' `spawn` start method |

---

## Troubleshooting: Windows Installation — SSL Certificate Error

If `install.bat` fails with an error like:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: unable to get local issuer certificate
ERROR: Failed to build 'ultralytics-yolov5'
```

this is a **corporate network / SSL inspection issue**, not a bug in the app. Managed Windows machines (schools, offices, government) often run a proxy that intercepts HTTPS traffic using a custom root certificate. Python does not trust that certificate by default, so any HTTPS call from inside a package's build script fails.

The updated `install.bat` handles this automatically. If you are running an older version:

### Option 1 — Re-run install.bat (recommended)

Pull the latest version of the repository and re-run `install.bat`. The installer now:
- Installs `pip-system-certs`, which tells pip to use the Windows certificate store (where your organisation's root CA is already trusted).
- Sets `PYTHONHTTPSVERIFY=0` for the duration of the install session so that `setup.py` subprocesses (which bypass pip's cert store and use Python's `urllib` directly) also succeed.

### Option 2 — Run manually if install.bat still fails

Open **Command Prompt** in the project folder and run:

```bat
set PYTHONHTTPSVERIFY=0
venv\Scripts\activate.bat
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir -r requirements.txt
```

The `set PYTHONHTTPSVERIFY=0` line only applies to the current Command Prompt window and does not change any system settings.

### Option 3 — Ask IT to add a certificate exception

If your organisation's IT policy blocks `PYTHONHTTPSVERIFY=0`, ask them to export the corporate root CA certificate (`.cer` or `.pem`) and then run:

```bat
set REQUESTS_CA_BUNDLE=C:\path\to\corporate-ca.pem
set SSL_CERT_FILE=C:\path\to\corporate-ca.pem
pip install --no-cache-dir -r requirements.txt
```

### Why does `ultralytics-yolov5` trigger this?

Unlike standard packages, `ultralytics-yolov5` makes an HTTPS download request **inside its `setup.py`** during the build step (before pip even installs it). This means pip's own certificate configuration doesn't protect it — the request goes directly through Python's `urllib`, which only trusts the certificates bundled with Python itself. On a machine with SSL inspection, that verification fails.

---

## Troubleshooting: No Animals Detected

If the pipeline returns `Empty` or `Unknown` for all images, work through the checks below in order.

### Step 1 — Check the startup banner

If a red banner appears at the top of the app saying **"AI models could not be loaded"**, animal detection is completely disabled. Follow the [AI Models Not Loading](#troubleshooting-ai-models-not-loading) section above before continuing.

### Step 2 — Check the Diagnostics Tab

Open the **Diagnostics** tab, upload one of the failing images, and click **Run Deep Inspection**. This bypasses the confidence threshold and shows raw model output.

- **MegaDetector status** — must show `MegaDetector Loaded Successfully`. If it shows a load error, the model file is missing or corrupt. Re-run `python force_download.py` and restart the app.
- **Raw candidates table** — lists every detection MegaDetector found, including those below the threshold. If the table is empty, MegaDetector found nothing in the image (see Step 4). If rows are present but all have low confidence, see Step 3.
- **BioClip predictions** — lists the top-20 species scores for the image. If this section is blank or shows a warning, BioClip did not initialise correctly (see Step 5).

### Step 3 — Lower the Confidence Threshold

The sidebar **Detection Confidence Threshold** defaults to **0.35**. Camera trap images (especially night/IR shots, distant animals, or dense vegetation) often produce MegaDetector scores of 0.15–0.30.

- Try **0.10–0.15** first, re-process, and check whether detections appear.
- BioClip reuses the same threshold for species scoring. With 20 wildlife classes in the list, softmax probabilities are spread across all classes; the top species may only score 0.15–0.25 even on a clear image.
- If lowering the threshold produces too many false positives, raise the **Review Queue Threshold** instead so genuine detections surface in the queue for manual review.

### Step 4 — Check Image Quality

MegaDetector may return zero detections if:

| Condition | What to do |
|-----------|-----------|
| Very dark / underexposed night image | Increase camera flash range; check if Day/Night classification is correct |
| Animal occupies < 2% of frame | Camera trap is too far from the path; reposition |
| Heavy motion blur | Increase shutter speed on camera settings |
| Image is grayscale or RGBA instead of RGB | Convert to standard JPEG/RGB before uploading |
| Corrupted or zero-byte file | Re-export from the SD card |

The **Diagnostics** raw candidate table shows detections at **all** confidence levels (including below 0.01), so even a poor-quality image should contain some rows if the animal is partially visible.

### Step 5 — Verify BioClip Initialised Correctly

BioClip requires two things to classify species: the model weights and pre-computed text embeddings for the species list. If either is missing, it silently returns nothing.

Check the app startup logs (terminal window where you ran `streamlit run app.py`) for:

```
Loading BioClip on cpu (Low-Spec: False)...
BioClip loaded successfully.
BioClip: Updated text features for 20 species.
```

If you see `Error loading BioClip` or `Error updating species list`, the model download is incomplete. Run `python force_download.py` and restart.

### Step 6 — Disable Low-Spec Mode

The **Low-Spec / Low-Memory Mode** checkbox applies INT8 dynamic quantization to MegaDetector and BioClip. On some hardware this degrades detection confidence enough that borderline detections (scores near 0.20–0.35) drop below the threshold.

If Low-Spec mode is enabled:
1. Uncheck it in the sidebar.
2. Click **Reload AI Models** in the sidebar.
3. Re-run the failing images.

Only use Low-Spec mode if the app crashes due to out-of-memory errors.

### Step 7 — Known Diagnostics Limitations (Current Version)

Be aware of two gaps in the current Diagnostics tab output:

1. **BioClip runs on the full image in the Diagnostics tab**, not on the cropped bounding box that the real pipeline uses. A score of `Zebra 0.03` in the Diagnostics view does not mean the real pipeline scored 0.03 — cropping to the animal region typically produces much higher scores. Use the raw candidate table to confirm MegaDetector found the animal first; if it did, BioClip is likely classifying it correctly in the main pipeline.

2. **The Diagnostics tab always uses a 0.2 threshold internally**, regardless of the sidebar slider. If your slider is set to 0.35, the Diagnostics view may show detections that the main pipeline would filter out. Compare the raw confidence values in the candidate table directly against your slider value.

---

## Technical Notes

- **MegaDetector V5a** — Microsoft AI for Earth model detecting Animal / Person / Vehicle classes.
- **BioClip (OpenCLIP)** — Imageomics foundation model for fine-grained species classification from cropped detections.
- **EasyOCR** — Reads date, time, and temperature from camera metadata strips via regex.
- **Independence rule** — Same species + same station + detections within the window → one IDE. RAI = IDEs / trap nights.
- **Privacy scrubbing** — Gaussian blur applied to Person/Vehicle bounding boxes; originals are never modified.
- **SQLite** (`wildlife_data.db`) — All persistent data (stations, IDEs, review actions, community observations, project config, ArcGIS sync log) is stored in a single local database.

---

## Recent Changes

### Platform & Stability Fixes (May 2026)

**Windows crash fix — instant machine restart during image processing**
- Added `CUDA_VISIBLE_DEVICES=-1` on Windows startup (before any torch import) to prevent PyTorch's CUDA initialisation from probing the GPU driver. On Windows with hybrid/Optimus graphics or certain driver versions, this probe causes a GPU TDR failure that reboots the machine instantly with no BSOD.
- Added `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `KMP_DUPLICATE_LIB_OK=TRUE` to prevent deadlocks from multiple OpenMP runtimes (PyTorch/MKL + OpenCV + EasyOCR each ship their own).
- Added `torch.set_num_threads(1)` unconditionally on Windows (previously only in Low-Spec mode) to prevent thread-stack exhaustion.
- Added `multiprocessing.freeze_support()` on Windows to prevent crashes when EasyOCR or YOLO spawn worker processes.

**MegaDetector & BioClip not loading**
- Fixed `megadetector>=5.0.0` version constraint in `requirements.txt` — the package uses calendar versioning (`1.0.0.YYYYMMDD`) so `>=5.0.0` was unsatisfiable; pip silently skipped the package. Changed to unpinned `megadetector`.
- Added `megadetector` to `environment.yml` — it was missing entirely, so conda users never had it installed.
- Fixed conda install path in both `install.sh` and `install.bat` — `force_download.py` (BioClip model, ~599 MB) was never called for conda users.

**Installer improvements**
- Added `--fresh` flag to `install.sh` and `install.bat` for a clean-slate reinstall (`./install.sh --fresh` / `install.bat --fresh`).
- When `install.bat` is double-clicked and an existing `venv\` is found, an interactive prompt now asks whether to update/repair or do a fresh install — no terminal required.

**In-app recovery**
- Added a dependency health check banner at startup that shows exactly which packages are missing and the commands to install them.
- Added a **Reload Models** button on the error banner and a **Reload AI Models** button in the sidebar — both clear `@st.cache_resource` and reload models without restarting the app or terminal.

### Earlier Changes

- **Dependency & Platform Compatibility (May 2026)**
  - Removed upper-bound limit (`<2.0.0`) on `numpy` to allow installing pre-compiled NumPy 2.x wheels on Python 3.14.
  - Enhanced `install.sh` to detect `python3.13` and `python3.12` before falling back to `python3`.
  - Fixed a `TypeError` in Excel report generation (`create_excel_report`) caused by `apply(len)` on null float values on newer Pandas/Arrow string backends.
  - Added **Low-Spec / Low-Memory Mode** sidebar toggle for INT8 quantization and single-thread CPU mode on machines with < 8 GB RAM.

---

## License

Open-source — intended for wildlife research and conservation.
