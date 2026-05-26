# Migration Plan: Streamlit → FastAPI + React

## Overview

Migrate the Wildlife Camera Trap Auto-Analyzer from a 3,176-line Streamlit monolith (`app.py`) to a
**FastAPI REST backend** + **React (Vite) frontend**. The `core/` modules are untouched — they become
the business logic layer that the FastAPI routes call directly.

---

## Project Structure (Target)

```
camera-traps/
├── backend/
│   ├── main.py                  # FastAPI app entry point, lifespan, CORS, static file mount
│   ├── routers/
│   │   ├── images.py            # Upload & processing endpoints
│   │   ├── results.py           # Review, edit, export results
│   │   ├── statistics.py        # Stats & analytics
│   │   ├── history.py           # Analysis history
│   │   ├── diagnostics.py       # Deep inspection tool
│   │   ├── ecological.py        # Ecological analytics (IDE, RAI, richness)
│   │   ├── qc.py                # QC dashboard
│   │   ├── stations.py          # Stations & deployments
│   │   ├── review.py            # Review queue
│   │   ├── community.py         # Community observer data entry
│   │   ├── spatial.py           # Spatial outputs & map viewer
│   │   ├── species.py           # Species reference library
│   │   ├── corridor.py          # Corridor movement analysis
│   │   ├── project_config.py    # Project configuration
│   │   └── arcgis.py            # ArcGIS sync
│   ├── models/
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── state.py             # App-level shared state (loaded AI models, config)
│   ├── services/
│   │   └── job_manager.py       # Background job tracking (upload → process → done)
│   └── requirements.txt         # Backend deps (fastapi, uvicorn, etc. + existing core deps)
│
├── frontend/
│   ├── index.html
│   ├── vite.config.ts           # Proxy /api → localhost:8000
│   ├── package.json
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx              # Router + tab layout
│   │   ├── api/
│   │   │   └── client.ts        # Axios/fetch wrapper pointing to /api
│   │   ├── pages/
│   │   │   ├── Upload.tsx       # Tab 1 — Upload & Process
│   │   │   ├── Results.tsx      # Tab 2 — Review Results
│   │   │   ├── Statistics.tsx   # Tab 3 — Statistics
│   │   │   ├── History.tsx      # Tab 4 — History & Analytics
│   │   │   ├── Diagnostics.tsx  # Tab 5 — Deep Inspection
│   │   │   ├── Ecological.tsx   # Tab 6 — Ecological Analytics
│   │   │   ├── QC.tsx           # Tab 7 — QC Dashboard
│   │   │   ├── Stations.tsx     # Tab 8 — Stations & Deployments
│   │   │   ├── ReviewQueue.tsx  # Tab 9 — Review Queue
│   │   │   ├── Community.tsx    # Tab 10 — Community Observer
│   │   │   ├── Spatial.tsx      # Tab 11 — Spatial & Map
│   │   │   ├── SpeciesLibrary.tsx # Tab 12 — Species Library
│   │   │   ├── Corridor.tsx     # Tab 13 — Corridor Analysis
│   │   │   ├── ProjectConfig.tsx # Tab 14 — Project Config
│   │   │   └── ArcGIS.tsx       # Tab 15 — ArcGIS Sync
│   │   ├── components/
│   │   │   ├── Layout/
│   │   │   │   ├── Sidebar.tsx  # Config sidebar (replaces st.sidebar)
│   │   │   │   └── TabNav.tsx   # Top tab navigation
│   │   │   ├── ImageGallery.tsx
│   │   │   ├── DetectionCard.tsx
│   │   │   ├── DataTable.tsx    # Editable results table
│   │   │   ├── ProgressBar.tsx  # Job progress polling
│   │   │   ├── ChartWrapper.tsx # Recharts wrapper
│   │   │   └── MapViewer.tsx    # Pydeck/Leaflet map
│   │   └── store/
│   │       └── configStore.ts   # Zustand store for sidebar config state
│
├── core/                        # UNCHANGED — all existing Python modules stay here
├── app.py                       # KEPT as-is until migration is complete
├── wildlife_data.db
└── dev.sh                       # Starts both servers with one command
```

---

## Phase 1 — FastAPI Backend Scaffold

### 1.1 Dependencies

Add to a new `backend/requirements.txt`:

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9      # file uploads
aiofiles>=23.2.1             # async file writes
```

Keep all existing `requirements.txt` entries — the `core/` modules still need them.

### 1.2 App Entry Point (`backend/main.py`)

- Create FastAPI app with a **lifespan context manager** that loads all AI models once at startup
  (replacing Streamlit's `@st.cache_resource`).
- Store loaded models in a module-level `AppState` dataclass in `models/state.py`.
- Mount all routers under the `/api` prefix.
- In production: mount the React `dist/` build as static files at `/`.
- Enable CORS for `localhost:5173` during development.

### 1.3 Model Loading

The current `load_models_v2()` in `app.py` loads:
- `OCRProcessor`
- `MegaDetectorWrapper`
- `BioClipClassifier`
- `DayNightClassifier`
- `AnimalDetector`
- `IndependenceEngine`, `QCEngine`, `StationManager`, `PrivacyScrubber`,
  `ReviewEngine`, `CommunityObserver`, `SpatialExporter`, `SpeciesLibrary`,
  `CorridorAnalyzer`, `ProjectConfig`, `ArcGISSync`

All of these move into the FastAPI lifespan and are stored in `AppState`. Each router
receives the relevant instances via FastAPI dependency injection.

### 1.4 Configuration State

The Streamlit sidebar sliders/checkboxes become a config object that:
- Has defaults defined in `AppState`.
- Is updatable via a `PATCH /api/config` endpoint.
- Persists in memory for the session (can be extended to a config file later).

Config fields:
```python
enable_ocr: bool = True
enable_detection: bool = True
enable_day_night: bool = True
enable_scrubbing: bool = True
enable_low_spec: bool = False
cpu_threads: int          # auto-detected
detection_confidence: float = 0.35
brightness_threshold: int = 100
ocr_strip_height: float = 0.10
blur_strength: int = 51
review_confidence_threshold: float = 0.90
reviewer_id: str = "anonymous"
default_station_id: str = "Station-1"
independence_window: int = 30
trap_nights_default: int = 30
```

---

## Phase 2 — Background Job System

Image processing is long-running (seconds to minutes per batch). In Streamlit this
blocks the UI via a progress bar. In FastAPI we use background jobs.

### Flow

```
POST /api/images/upload        → saves files, creates job_id
POST /api/images/process/{job_id} → starts BackgroundTask, returns immediately
GET  /api/images/job/{job_id}  → polls status {queued|running|done|error} + progress %
GET  /api/images/results/{job_id} → returns processed results when done
```

### Job Manager (`services/job_manager.py`)

- In-memory dict keyed by `job_id` (UUID).
- Each job stores: `status`, `total`, `completed`, `results`, `error`.
- Processing runs in a `BackgroundTask` (FastAPI built-in) or a `ThreadPoolExecutor`
  (since the core ML code is CPU-bound and not async).
- The frontend polls `GET /api/images/job/{job_id}` every 1–2 seconds and updates
  the progress bar.

---

## Phase 3 — API Endpoints (per tab)

### Tab 1 — Upload & Process
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/images/upload` | Accept multipart files, save to temp dir, return `job_id` |
| POST | `/api/images/process/{job_id}` | Start background processing with current config |
| GET | `/api/images/job/{job_id}` | Poll progress `{status, completed, total}` |
| GET | `/api/images/results/{job_id}` | Return processed results list |

### Tab 2 — Review Results
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/results` | All results from DB with optional filters |
| PATCH | `/api/results/{image_id}` | Edit species, notes, etc. |
| GET | `/api/results/export/excel` | Download Excel report |
| GET | `/api/results/export/json` | Download JSON export |
| GET | `/api/results/image/{filename}` | Serve image file |

### Tab 3 — Statistics
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/stats/summary` | Species distribution, day/night counts, confidence distribution |

### Tab 4 — History & Analytics
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/history` | All historical processing sessions |
| DELETE | `/api/history/{session_id}` | Delete a session |
| GET | `/api/history/export/csv` | Download history CSV |

### Tab 5 — Diagnostics
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/diagnostics/inspect` | Run deep inspection (OCR + MegaDetector raw + BioClip top-20) on a single image |

### Tab 6 — Ecological Analytics
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ecological/ide` | Independent Detection Events |
| GET | `/api/ecological/rai` | Relative Abundance Index |
| GET | `/api/ecological/timeline` | Detection timeline |
| GET | `/api/ecological/richness` | Species richness over time |
| GET | `/api/ecological/accumulation` | Species accumulation curve |
| GET | `/api/ecological/activity` | Time-of-day activity heatmap data |
| GET | `/api/ecological/export` | Export ecological data |

### Tab 7 — QC Dashboard
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/qc/flags` | All QC flags with severity |
| GET | `/api/qc/summary` | Flag counts by type |

### Tab 8 — Stations & Deployments
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/stations` | List all stations |
| POST | `/api/stations` | Add station |
| PATCH | `/api/stations/{id}` | Update station |
| DELETE | `/api/stations/{id}` | Delete station |
| GET | `/api/stations/deployments` | List deployments |
| POST | `/api/stations/deployments` | Add deployment |
| GET | `/api/stations/summary` | Summary with trap nights |
| GET | `/api/stations/map` | GeoJSON for map rendering |

### Tab 9 — Review Queue
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/review/queue` | Images below confidence threshold |
| POST | `/api/review/confirm/{id}` | Confirm detection |
| POST | `/api/review/correct/{id}` | Correct species label |
| POST | `/api/review/flag/{id}` | Flag for removal |
| GET | `/api/review/log` | Correction log |
| GET | `/api/review/privacy-audit` | Privacy scrub audit log |

### Tab 10 — Community Observer
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/community/observations` | List observations |
| POST | `/api/community/observations` | Add observation |
| DELETE | `/api/community/observations/{id}` | Delete observation |
| GET | `/api/community/crosscheck` | Cross-verify with camera detections |

### Tab 11 — Spatial & Map
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/spatial/geojson` | GeoJSON of all detections |
| GET | `/api/spatial/export/geojson` | Download GeoJSON file |
| GET | `/api/spatial/export/csv` | Download georeferenced CSV |
| GET | `/api/spatial/export/shapefile` | Download Shapefile ZIP |
| GET | `/api/spatial/export/kml` | Download KML file |

### Tab 12 — Species Library
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/species` | List all species |
| GET | `/api/species/lookup/{name}` | Lookup species info |
| GET | `/api/species/synonyms/{name}` | Resolve synonym |
| POST | `/api/species/add` | Add custom species |

### Tab 13 — Corridor Analysis
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/corridor/pairs` | Station pairs within corridor distance |
| GET | `/api/corridor/movements` | Directional movement events |
| GET | `/api/corridor/bottlenecks` | Bottleneck pairs |
| GET | `/api/corridor/utilisation` | Corridor utilisation by species |

### Tab 14 — Project Config
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/project` | Current project config |
| PATCH | `/api/project` | Update project settings |
| POST | `/api/project/baseline/lock` | Lock reference baseline |
| GET | `/api/project/export` | Export project config |

### Tab 15 — ArcGIS Sync
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/arcgis/push` | Push data to ArcGIS Online/Enterprise |
| GET | `/api/arcgis/status` | Last sync status |
| GET | `/api/arcgis/export/shapefile` | Download Shapefile |
| GET | `/api/arcgis/export/geojson` | Download GeoJSON |
| GET | `/api/arcgis/export/kml` | Download KML |

---

## Phase 4 — React Frontend Scaffold

### 4.1 Setup

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm install axios react-router-dom zustand recharts leaflet react-leaflet
npm install -D @types/leaflet
```

### 4.2 Tech Stack

| Concern | Library |
|---------|---------|
| Routing | `react-router-dom` v6 |
| HTTP client | `axios` with a base `/api` client |
| Global config state | `zustand` |
| Charts | `recharts` |
| Map | `react-leaflet` (replaces `pydeck`) |
| Tables | `@tanstack/react-table` |
| UI components | `shadcn/ui` (Tailwind-based, no vendor lock-in) |
| File upload | native `<input type="file">` + `axios` multipart |

### 4.3 Vite Dev Proxy (`vite.config.ts`)

```ts
server: {
  proxy: {
    '/api': 'http://localhost:8000'
  }
}
```

This makes `localhost:5173` the single URL in development. All `/api/*` calls
are transparently forwarded to FastAPI.

### 4.4 Global Config Store (`store/configStore.ts`)

Zustand store mirrors the backend config schema. On app load it fetches
`GET /api/config` and hydrates the store. The Sidebar component reads/writes
from this store and calls `PATCH /api/config` on change.

### 4.5 Upload & Progress Pattern (Tab 1)

1. User selects files → `POST /api/images/upload` (multipart) → receive `job_id`
2. `POST /api/images/process/{job_id}` to start processing
3. Poll `GET /api/images/job/{job_id}` every 1.5s → update `<ProgressBar />`
4. When `status === "done"` → fetch `GET /api/images/results/{job_id}` → populate results store
5. Auto-navigate to Results tab

---

## Phase 5 — Production Build & Single Server

Once the React app is built, FastAPI serves everything:

```bash
# Build React
cd frontend && npm run build   # outputs to frontend/dist/

# FastAPI serves dist/ as static files
# In backend/main.py:
app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")
```

Run the entire app with one command:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## Phase 6 — Dev Workflow (Two Commands, One Script)

`dev.sh` (root of project):

```bash
#!/bin/bash
# Start FastAPI backend
uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start React dev server
cd frontend && npm run dev &
FRONTEND_PID=$!

# Kill both on Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
```

Run: `bash dev.sh` → open `http://localhost:5173`

---

## Build Order (Step by Step)

| Step | What | Output |
|------|------|--------|
| 1 | Create `backend/` folder structure | Empty routers, `main.py` scaffold |
| 2 | Implement `AppState` + lifespan model loading | Models load at startup |
| 3 | Implement `PATCH /api/config` + `GET /api/config` | Config works |
| 4 | Implement upload + job manager | `POST /api/images/upload` + polling |
| 5 | Implement Tab 1 processing pipeline | End-to-end image processing via API |
| 6 | Scaffold React app + Vite proxy | `localhost:5173` proxies to FastAPI |
| 7 | Build Sidebar config UI + Zustand store | Config syncs frontend ↔ backend |
| 8 | Build Upload page with progress polling | Full Tab 1 working |
| 9 | Build Results page (Tab 2) | Editable table, image viewer, export |
| 10 | Build remaining tabs one by one (3–15) | Feature parity with Streamlit |
| 11 | Production build + static file mount | Single server deployment |
| 12 | Update Dockerfile + docker-compose | Containerised deployment |

---

## What Stays the Same

- All files in `core/` — zero changes
- `wildlife_data.db` — same SQLite database
- `install.sh` / `install.bat` — still install Python deps
- Windows thread/GPU env-var fixes — move into `backend/main.py` startup

## What Gets Deleted

- `app.py` — replaced entirely
- Streamlit from `requirements.txt`
- `pydeck` (replaced by `react-leaflet` in the frontend)
