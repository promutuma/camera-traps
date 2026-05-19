# WildlifeID Pro — Overhaul Feature Specification
### Gambella Wetland Landscape Baseline Survey, Kenya
**Prepared by:** CETRAD  
**Version:** 2.0 — Full Platform Overhaul  
**Status:** Proposed  
**Purpose:** Define the complete feature set for the overhauled WildlifeID Pro platform, building on the existing Streamlit prototype and redesigning it as a cross-platform, scientifically rigorous, field-deployable application.

---

## 1. Overhaul Philosophy

The existing application (Streamlit-based, single-machine) is a functional prototype. The overhaul transforms it into a **production-grade, cross-platform tool** that can run:

- On a **laptop or desktop** in the field without internet (offline-first)
- As a **local web server** on a shared field network (multi-user, same site)
- As a **cloud-hosted instance** for remote teams and stakeholders
- On **Windows, macOS, and Linux** without modification

The overhaul retains what works — MegaDetector V5a detection, BioClip species classification, EasyOCR metadata extraction, day/night classification, gallery/inspector views, and Excel export — and addresses every current limitation while adding the ecological analytics required by the survey methodology.

---

## 2. Cross-Platform Architecture (New)

### 2.1 Packaging Strategy

The app is repackaged from a raw Streamlit script into a **self-contained executable** that any user can run without installing Python, dependencies, or configuring environments.

| Deployment Mode | How it Works | Best For |
|---|---|---|
| **Desktop app (offline)** | PyInstaller or Nuitka bundles Python + all dependencies into a single `.exe` (Windows), `.app` (macOS), or binary (Linux) | Field teams with no internet |
| **Local network server** | App runs on one machine; others on the same Wi-Fi connect via browser at `http://[host-ip]:8501` | Field station with shared laptop |
| **Docker container** | Single `docker-compose up` launches the full stack | IT-managed deployments, cloud servers |
| **Cloud hosted** | Deploy container to any cloud (AWS, Azure, GCP) with persistent storage | Remote review teams, headquarters |

### 2.2 One-Command Installer

- A single installer script (`install.sh` / `install.bat`) handles everything: dependency installation, model downloads, database initialization, and launching the app.
- First-run wizard guides the user through: selecting deployment mode, setting storage paths, and creating the admin account.
- **No Python knowledge required** to install or run the application.

### 2.3 Hardware Requirements (Minimum)

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4-core, 2.0 GHz | 8-core, 3.0 GHz |
| RAM | 8 GB | 16 GB |
| Storage | 50 GB free | 500 GB SSD |
| GPU | Not required | NVIDIA GPU (CUDA) for faster inference |
| OS | Windows 10, macOS 12, Ubuntu 20.04 | Any of the above |

- **GPU auto-detection:** If a CUDA-compatible GPU is found, inference is automatically routed to it. If not, CPU inference runs with a progress indicator.
- **Low-spec mode:** A toggle reduces model precision (INT8 quantization) to run on machines with < 8 GB RAM.

### 2.4 Offline-First Data Handling

- All AI models are bundled or downloaded once on first run and cached locally — no internet needed during field operations.
- The database is local SQLite by default; upgrades to PostgreSQL for multi-user deployments via a config switch.
- Sync module: when internet is available, the app can push validated records to a central server or ArcGIS Portal in the background.

---

## 3. Retained and Improved Core Features

### 3.1 Batch Image Upload and Processing (Improved)

**What exists:** Multi-image upload, unified pipeline for detection, classification, OCR, day/night, and database storage.

**Improvements:**
- **Folder-based ingestion** (new): Drag and drop an entire station folder rather than selecting individual files. The app reads the folder structure and assigns Station ID and Deployment ID automatically from the folder name (`YYYYMMDD_StationID_DeploymentID`).
- **Folder naming validator** (new): On ingestion, the app checks whether the folder name conforms to the naming convention. Non-conforming folders are flagged before processing begins.
- **Resumable batch processing** (new): If the app is closed mid-batch, processing resumes from the last completed image on relaunch.
- **Hash-based deduplication** (new): Images already in the database are detected by checksum and skipped, preventing double-counting.
- **Video support** (new): MP4 and AVI files are accepted; frames are extracted at configurable intervals (e.g., 1 frame per 5 seconds) and processed through the same pipeline.
- **Progress dashboard** (improved): Real-time progress bar per image and per batch, with estimated time remaining and per-image status (queued / processing / complete / failed).

### 3.2 AI Detection and Species Classification (Improved)

**What exists:** MegaDetector V5a for object detection; BioClip/OpenCLIP for species classification; confidence threshold slider.

**Improvements:**
- **Regional species fine-tuning** (new): BioClip is fine-tuned on a Gambella-specific species list. The classifier prioritizes species known to occur in the ecosystem before falling back to global predictions.
- **Top-3 species candidates** (improved): The interface shows the top 3 predicted species with confidence scores, not just the top-1 prediction, giving reviewers more context.
- **Domestic vs. wild classifier** (new): A dedicated binary classifier distinguishes livestock (cattle, goats, camels, donkeys) from wildlife before species classification runs. Livestock detections are tagged separately and excluded from wildlife indicators.
- **Multi-animal frame handling** (improved): When multiple animals appear in one frame, each detected bounding box is classified independently and all detections are stored as separate records linked to the same image.
- **Confidence threshold per stratum** (new): Different confidence thresholds can be set for wetland habitat, watering points, and corridors independently, rather than a single global slider.
- **Model version tracking** (new): The model version used for each detection is stored in the database, so results are reproducible and traceable if models are updated.

### 3.3 OCR Metadata Extraction (Improved)

**What exists:** EasyOCR reads date, time, and temperature from the image metadata strip; adjustable strip margin slider.

**Improvements:**
- **GPS coordinate parsing** (new): If coordinates appear in the metadata strip or EXIF data, they are extracted and stored automatically.
- **Camera model parsing** (new): Camera make and model extracted from EXIF and stored per image.
- **OCR failure fallback** (improved): If OCR fails to parse a date/time, the app falls back to the image file's filesystem timestamp and flags the record for review rather than leaving it blank.
- **Manual metadata entry form** (improved): When OCR fails, a form pre-populated with the best available values lets users correct fields directly without leaving the interface.
- **Temperature unit standardization** (new): Temperature values are stored in Celsius; Fahrenheit values are detected and converted automatically.

### 3.4 Environmental and Day/Night Classification (Improved)

**What exists:** Grayscale/infrared detection; brightness-based day/night classification with threshold slider.

**Improvements:**
- **Astronomy-based day/night** (new): In addition to brightness, the app calculates theoretical sunrise/sunset times from the GPS coordinates and date, providing a ground-truth check against the brightness-based classification.
- **Moon phase recording** (new): Lunar phase (new moon to full moon) is calculated from the image date and stored — relevant for nocturnal species activity analysis.
- **Weather condition field** (new): Optional manual field (Clear / Overcast / Rain / Fog) added to the image record for environmental context.

### 3.5 Interactive Review and Data Curation (Improved)

**What exists:** Gallery view and inspector view; manual correction of animal names; verification and notes fields.

**Improvements:**
- **Structured HITL review queue** (new): A dedicated "Review Queue" tab lists all images where confidence < 90%. Reviewers work through the queue sequentially. Each review action (accept / correct / reject) is logged with reviewer ID and timestamp.
- **Side-by-side reference panel** (new): In the inspector, the top-3 predicted species are shown alongside reference images pulled from a local species reference library, helping reviewers make informed corrections.
- **Keyboard shortcuts** (new): Reviewers can navigate and action images using keyboard shortcuts (next image, accept, reject, flag) for faster throughput.
- **Batch accept/reject** (new): Reviewers can apply the same action to a filtered group (e.g., all detections of the same species above 95% confidence) in one click.
- **Correction tracking** (new): Every manual correction is stored as a separate record alongside the original AI prediction, building a correction dataset for future model retraining.
- **Stratum assignment** (new): Each image/station is assigned to a monitoring stratum (wetland habitat / watering point / corridor) during review or at ingest, used downstream for indicator calculations.

### 3.6 Analytics and Database (Major Improvement)

**What exists:** SQLite storage; aggregate statistics (totals, animal counts, vehicle/person counts); historical trends.

**Improvements:**
- **Database upgrade path** (new): SQLite remains the default for single-user/offline use; a configuration toggle switches to PostgreSQL for multi-user deployments with no data migration required.
- **Stratum-aware analytics** (new): All statistics are broken down by monitoring stratum (wetland habitat, watering points, corridors) rather than only aggregate totals.
- **Long-term baseline locking** (new): Once the Data Manager approves the baseline dataset, baseline values are locked and marked read-only. Future uploads are automatically compared against them.
- **Full audit log** (new): Every change to any record (detection result, species correction, verification status, note) is logged with user ID, timestamp, and before/after values.

### 3.7 Data Export (Improved)

**What exists:** Excel (.xlsx) export with filename, date, time, temperature, species, confidence, day/night, and notes.

**Improvements:**
- **CSV export** (new): Lightweight alternative to Excel for use in R, Python, or GIS software.
- **GeoJSON/Shapefile export** (new): Detection events exported as georeferenced point layers for direct import into ArcGIS, QGIS, or Google Earth.
- **PDF summary report** (new): Auto-generated report including key indicators, charts, and a station map — suitable for sharing with non-technical stakeholders.
- **Filtered exports** (improved): Export by date range, species, station, stratum, confidence level, or verification status.

### 3.8 Diagnostics and Logging (Improved)

**What exists:** Diagnostics tab for cropped area inspection and OCR debugging; model loading log expander.

**Improvements:**
- **Per-station health dashboard** (new): Visual summary of each station's trap nights, images processed, camera-down events, and QC flag status.
- **Model performance log** (new): Tracks AI confidence score distributions over time; alerts the user if average confidence drops (possible model drift or image quality degradation).
- **Exportable diagnostic report** (new): The diagnostics tab can generate a PDF report of all QC flags for submission to the data manager.

---

## 4. New Features (Not in Current App)

### 4.1 Automated 30-Minute Independence Rule Engine

This is the single most important new feature for scientific validity.

- After images are processed, detections are automatically grouped into **Independent Detection Events (IDE)** by applying the 30-minute rule: same species, same station, detections < 30 minutes apart = one event.
- Each IDE is assigned a unique IDE ID stored in the database.
- **Group-living species**: the entire group is one IDE; individual count per IDE is stored separately.
- **Continuous presence**: a sequence of images of the same animal(s) within a 30-minute window = one IDE, not multiple.
- **Clock drift correction**: the app detects cameras whose timestamps are inconsistent with surrounding cameras (> ±5 min deviation) and flags them for correction before the independence rule is applied.
- **Configurable window**: the 30-minute threshold is adjustable per project (e.g., some protocols use 60 minutes) via the project settings panel.
- **IDE timeline view**: a visual timeline per station shows IDEs as colour-coded blocks, making it easy to spot clustering or gaps.

### 4.2 Ecological Indicators Dashboard

A dedicated analytics module that automatically computes all indicators required by the survey methodology.

#### Relative Abundance Index (RAI)
```
RAI = Independent Detection Events / Total Camera Trap Nights
Trap Nights = Number of Active Cameras × Number of Sampling Nights
```
- Computed per species, per station, per corridor, per stratum, and at landscape level.
- Camera-down nights (detected from metadata gaps or empty sequences) automatically excluded from the trap night count.
- Displayed as a sortable table and bar chart; exportable to CSV, Excel, and PDF.
- Temporal breakdown: RAI by week, month, and full baseline period.

#### Species Richness
- Total unique species detected per stratum over the full baseline period.
- **Species accumulation curve** generated automatically: plots cumulative new species discovered against trap nights. Curve approaching asymptote = sampling effort is adequate.
- Singleton species (detected only once) flagged separately.

#### Visitation Rate (Watering Points)
```
Visit Rate = Independent Visits / Trap Nights
```
- Computed per species, per watering point.
- Time-of-day breakdown: visits binned into 2-hour blocks and displayed as a heatmap (species × time of day × station).
- Diurnal vs. nocturnal ratio reported per species.

#### Mean Group Size
```
Mean Group Size = Total Individuals Observed / Number of Independent Visits
```
- Computed per species, per station.
- Min, max, and standard deviation also reported.
- Outlier detection: group sizes > 3 standard deviations from the mean flagged for review.

#### Percentage Change (Future Monitoring Cycles)
```
% Change = ((Current Value − Baseline Value) / Baseline Value) × 100
```
- Baseline values locked at project inception.
- Future monitoring uploads automatically trigger a comparison report.
- Direction indicator (↑ / ↓) and percentage displayed for each indicator.

### 4.3 Station and Deployment Manager (New)

A structured module for managing camera station records — currently absent from the app.

- **Station registry**: each station has a record containing Station ID, GPS coordinates, habitat stratum, camera model, deployment start/end dates, and assigned team member.
- **Deployment history**: each time a camera is redeployed (new SD card, repositioned), a new deployment record is created under the same station ID.
- **Trap night calculator**: automatically computes active trap nights per station from deployment dates and camera-down events.
- **Station map**: interactive map showing all active stations colour-coded by stratum and QC health status.
- **Camera functionality rate**: computed per station; stations below 90% functionality are flagged automatically.

### 4.4 Privacy Scrubbing Module (New)

The existing app detects people and vehicles but does not blur or remove them. This module adds full privacy compliance.

- **Automatic face blurring**: all human faces detected by MegaDetector are blurred in the derivative copy before any image is shown in the review interface or exported.
- **Vehicle anonymisation**: license plates and distinctive vehicle markings are blurred.
- **Non-destructive workflow**: the original unblurred image is retained in a restricted-access archive; only the Data Manager role can access originals.
- **Scrubbing audit log**: every scrubbed image is logged with the bounding boxes affected and the scrubbing timestamp.
- **Bulk scrub on ingest**: scrubbing runs as part of the ingestion pipeline so no unblurred images are ever visible to Field Technician or Data Analyst roles.

### 4.5 Species Reference Library (New)

A local, searchable reference database that supports species identification and name standardization.

- Pre-loaded with all mammal species known or likely to occur in the Gambella wetland landscape.
- Each species record contains: common name, scientific name, family, order, IUCN Red List status, Kenya-specific conservation status, reference images (3–5 per species), and brief ecological notes.
- **Synonym resolution**: if a reviewer enters an outdated or alternative name, the system resolves it to the accepted taxonomic name automatically.
- **Custom additions**: the Data Manager can add new species to the library as the project develops.
- Used by the HITL review panel to show reference images alongside flagged detections.

### 4.6 Corridor Movement Analysis (New)

For cameras deployed along wildlife corridors (spaced every 500–800 m), a dedicated analysis module infers movement patterns.

- **Directional flow detection**: sequential detections of the same species across corridor cameras within a defined time window (configurable, default 6 hours) are linked into a movement event with a direction (inbound / outbound relative to the wetland).
- **Passage frequency**: number of directional movement events per species per corridor per week/month.
- **Bottleneck identification**: corridor segments with zero detections despite active cameras are flagged as potential movement barriers.
- **Corridor utilisation map**: detection density heat map overlaid on the corridor geometry, exportable to GeoJSON for ArcGIS.

### 4.7 Community Observer Data Entry (New)

Enables cross-verification of camera trap data with field observer records, as required by the methodology.

- A simple data entry form (accessible to Field Technician role) allows entry of: date, time, location (GPS or station ID), species observed, count, observer name, and observation type (direct sighting / track / scat / kill site / vocalisation).
- Community records are stored in a separate table linked to the nearest camera station.
- Displayed alongside camera trap detections in the inspector view for cross-referencing.
- Species observed by community but not yet detected by cameras are flagged as "Community-only record" and listed in the species inventory with a distinct marker.

### 4.8 Project and Survey Configuration Panel (New)

A settings module that makes the app configurable for any survey, not just Gambella — enabling use across different projects and landscapes.

- **Project setup wizard**: define project name, study area, survey start/end dates, monitoring strata, and target species list.
- **Indicator thresholds**: configure the minimum trap nights (default 60), minimum camera functionality rate (default 90%), confidence threshold per stratum, and independence window (default 30 minutes).
- **Baseline locking**: Data Manager can lock the baseline at the end of the sampling period; locked values are timestamped and cannot be altered.
- **Multi-project support**: the app can manage multiple independent projects simultaneously, each with its own database, species list, and configuration. Switch between projects from the sidebar.
- **Project export**: the full project (database, config, reports) can be exported as a single archive file for archiving or transfer to another machine.

### 4.9 Automated QC Flag System (New)

- Cameras with < 60 trap nights at end of survey → flagged as **INSUFFICIENT EFFORT**.
- Camera functionality < 90% → flagged as **LOW FUNCTIONALITY**.
- Stations with zero detections and > 10 trap nights → flagged as **POSSIBLE FAILURE**.
- Images with null confidence scores → flagged as **MODEL ERROR**.
- Timestamps outside the defined survey period → flagged as **OUT OF RANGE**.
- Duplicate station IDs → flagged as **DUPLICATE STATION**.
- Clock drift > ±5 minutes detected → flagged as **TIMESTAMP INCONSISTENCY**.
- All flags displayed in a QC dashboard with filter, sort, and export options.
- Weekly QC summary auto-generated and surfaced on the home screen.

### 4.10 Spatial Outputs and ArcGIS Integration (New)

- **GeoJSON export**: all detection events, station locations, and corridor layers exported as GeoJSON for QGIS or web mapping.
- **Shapefile export**: same layers in Shapefile format for ArcGIS Desktop.
- **ArcGIS Portal sync** (optional): if ArcGIS Online credentials are configured, validated records are pushed to a feature layer automatically.
- **Built-in map viewer**: an interactive Leaflet.js map embedded in the app displays station locations, detection heat maps, and corridor layers without needing external GIS software.

---

## 5. User Roles and Access Control (New)

| Role | Key Permissions |
|---|---|
| **Field Technician** | Upload images, enter community observations, view own station status |
| **Data Analyst** | Process batches, run QC, edit detection metadata, access diagnostics |
| **Reviewer / Expert** | Access HITL review queue, override species identifications, access reference library |
| **Data Manager** | Full access, lock baseline, manage users, access original unblurred images, approve exports |
| **Read-only Stakeholder** | View dashboards, download approved reports only |

- Login is required for all roles. Default authentication is username/password stored locally.
- For cloud deployments, OAuth2/SSO integration is supported.
- All actions by all users are written to an immutable audit log.

---

## 6. Technology Stack Recommendation

| Component | Current | Proposed Overhaul |
|---|---|---|
| **UI framework** | Streamlit | Streamlit (retained) + packaged via PyInstaller for desktop |
| **Object detector** | MegaDetector V5a | MegaDetector V5a (retained) + GPU auto-routing |
| **Species classifier** | BioClip / OpenCLIP | BioClip fine-tuned on Gambella species list |
| **OCR** | EasyOCR | EasyOCR (retained) + EXIF fallback |
| **Database** | SQLite | SQLite (default) / PostgreSQL (multi-user, config switch) |
| **Spatial** | None | PostGIS extension + Leaflet.js map viewer + GeoJSON/Shapefile export |
| **Privacy scrubbing** | Not implemented | OpenCV blur on MegaDetector person/vehicle bounding boxes |
| **Independence rule** | Manual | Automated engine using Pandas time-window grouping |
| **Indicators** | Manual | Native Python (Pandas, NumPy, SciPy) calculator module |
| **Reporting** | Excel only | Excel + CSV + PDF (ReportLab or WeasyPrint) |
| **Packaging** | None (raw script) | PyInstaller (Windows .exe, macOS .app, Linux binary) + Docker |
| **Authentication** | None | Local username/password + optional OAuth2 |

---

## 7. What the Existing App Already Does Well (Retained As-Is)

The following features are production-ready and carried forward without fundamental changes:

- MegaDetector V5a detection pipeline (animals, people, vehicles with bounding boxes)
- BioClip species classification from cropped bounding boxes
- EasyOCR metadata strip reading (date, time, temperature)
- Brightness-based day/night classification
- Gallery and inspector view layouts
- Interactive data editor for label correction
- Verification and notes fields
- SQLite database storage
- Excel export
- Model loading log and diagnostics tab

---

## 8. Development Priority Order

| Priority | Feature | Rationale |
|---|---|---|
| 1 | Cross-platform packaging (PyInstaller + Docker) | Enables field deployment on any machine |
| 2 | 30-minute independence rule engine | Core scientific requirement |
| 3 | Ecological indicators dashboard (RAI, richness, visitation) | Core output of the survey |
| 4 | Station and deployment manager | Required for multi-station operations |
| 5 | Privacy scrubbing module | Ethical and legal compliance |
| 6 | Structured HITL review queue | Scientific validity of species IDs |
| 7 | Species reference library | Supports accurate review |
| 8 | QC flag system | Data quality assurance |
| 9 | Spatial outputs and map viewer | Reporting and stakeholder communication |
| 10 | Community observer data entry | Cross-verification |
| 11 | Corridor movement analysis | Advanced ecological analysis |
| 12 | Project configuration panel | Multi-project scalability |
| 13 | ArcGIS Portal sync | Integration with institutional GIS infrastructure |

---

*Version 2.0 — This document supersedes v1.0. To be reviewed at the start of each development sprint and updated to reflect completed features.*
