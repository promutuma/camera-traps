# Wildlife Camera Trap Auto-Analyzer

A user-friendly **Streamlit-based dashboard** for automated analysis of wildlife camera trap images. This system extracts metadata via OCR, detects animals using state-of-the-art models (**MegaDetector V5a**), identifies species using **BioClip**, classifies day/night conditions, and generates downloadable reports.

![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![MegaDetector](https://img.shields.io/badge/MegaDetector-V5a-blue)
![BioClip](https://img.shields.io/badge/BioClip-Enabled-green)

## Features

- **Batch Image Upload**: Upload multiple camera trap images (JPG/PNG) simultaneously
- **OCR Metadata Extraction**: Automatically extracts date, time, and temperature from image metadata strips
- **Advanced Animal Detection**:
  - **MegaDetector V5a** for localizing animals, people, and vehicles
  - **BioClip (OpenCLIP)** for fine-grained species classification
- **Diagnostics Tool**: Specialized tab to debug OCR crops and view raw model candidates
- **Day/Night Classification**: Automatically classifies images based on brightness analysis
- **History & Analytics**: Save results to SQLite database and view long-term trends
- **Editable Results**: Review and manually correct detected animal names in an interactive table
- **Excel Reports**: Generate downloadable Excel (.xlsx) reports with all extracted data

## Project Structure

```text
camera-traps/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Container orchestration
├── wildlife_data.db            # SQLite database (auto-created)
├── README.md                   # Documentation
└── core/                       # Processing modules
    ├── animal_detector.py      # Ensemble (MD + BioClip)
    ├── bioclip_classifier.py   # BioClip implementation
    ├── day_night_classifier.py # Day/night classification
    ├── ocr_processor.py        # OCR metadata extraction
    └── image_processor.py      # Unified processing pipeline
```

## Deployment Scenarios

Choose the deployment method that best fits your needs.

### Scenario A: Local Installation (Windows/Mac/Linux)

Best for development and running on a personal laptop.

**Prerequisites:**

- **Python 3.11** (Strict requirement for dependency compatibility)
- git
- **System Dependencies** (Linux only): OpenCV requires missing shared libraries. On Ubuntu/Debian run:
  ```bash
  sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
  ```

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd camera-traps
   ```

2. **Environment Setup:**

   **Option A: Virtual Environment (Recommended for keeping system clean)**
   
   _Windows:_
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   _Mac/Linux:_
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   **Option B: Bare-Metal (No Virtual Environment)**
   
   If virtual environments are causing permission or networking issues on your machine, you can install directly to your user space using the `--user` flag.
   _Warning: This may conflict with other global Python projects you have installed._
   
3. **Install dependencies:**

   _If using a virtual environment (Option A):_
   ```bash
   pip install -r requirements.txt
   ```
   
   _If installing Bare-Metal (Option B):_
   ```bash
   # On recent Linux versions (like Ubuntu 24.04), you must bypass PEP 668 protections:
   pip3 install --user --break-system-packages -r requirements.txt
   ```

   _Tip: If you experience timeout errors during installation on slow or unreliable connections, try using `pip3 install --default-timeout=1000 -r requirements.txt` instead._

   _Note: During Bare-Metal installation, you may see multiple warnings stating that scripts are installed in `~/.local/bin` which is not on PATH. You can safely ignore these warnings, as we bypass them by running `python3 -m streamlit` later._

   > [!WARNING]
   > The first time you run an analysis, the app requires the **MegaDetector** and **BioClip** AI models (~1.5 GB). 
   > 
   > **CRITICAL SETUP STEP:** Downloading these massive models via Streamlit's background threads from within a Python virtual environment (`venv`) is unstable and prone to crashing or hanging indefinitely due to network timeouts.
   
   To permanently install the models to your cache *before* running the dashboard, please run the included dedicated download script:
   ```bash
   python3 force_download.py
   ```
   _(If this script fails due to a timeout, just re-run the command. It is designed to resume exactly where it left off until finished.)_

4. **Run the application:**
   ```bash
   python3 -m streamlit run app.py
   ```
   Access at `http://localhost:8501`.

---

### Scenario B: Docker (Containerized)

Best for reproducible environments and easy cleanup. Requires **Docker Desktop**.

1. **Build and Run with Docker Compose:**

   ```bash
   docker-compose up --build
   ```

   _Alternatively, using plain Docker:_

   ```bash
   docker build -t wildlife-analyzer .
   docker run -p 8501:8501 wildlife-analyzer
   ```

2. **Access the App:**
   Open `http://localhost:8501` in your browser.

   _Note: Model weights are downloaded inside the container. To persist them between runs, you may want to mount a volume for the cache directories._

---

### Scenario C: Streamlit Community Cloud

Best for sharing with others without hosting infrastructure.

1. **Push your code to GitHub.**
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select your repository.
4. Set the "Main file path" to `app.py`.
5. Click **Deploy**.
   _Streamlit Cloud will automatically detect `requirements.txt` and install dependencies (including system dependencies for OpenCV)._

## Usage Guide

1. **Upload Images** (Tab 1):
   - Drag and drop camera trap images.
   - Click "Process Images".
2. **Review Results** (Tab 2):

   - Switch between **Gallery View** and **Inspector View**.
   - **Inspector View**: Allows deep-diving into individual images, seeing bounding boxes, and editing incorrect detections.
   - **Editing**: Change the "Primary Label" or "Species Name" and click "Save Changes".

3. **History & Analytics** (Tab 3):

   - View aggregate statistics of all processed batches.

4. **Diagnostics** (Tab 4):
   - Use if OCR is failing. Adjust the "OCR Strip Height" in the sidebar to match your camera's metadata bar.

## Configuration

### Sidebar Settings

- **Confidence Threshold**: Defaults to 0.35. Lower this if the AI is missing animals (false negatives). Raise it if it's detecting rocks/trees as animals (false positives).
- **Brightness Threshold**: Adjusts the sensitivity for Day/Night classification.
- **OCR Strip Height**: Percentage of the image bottom to scan for date/time text.

## Technical Details

- **MegaDetector V5a**: A Microsoft AI for Earth model tuned to detect generic "Animal", "Person", and "Vehicle" classes.
- **BioClip**: A foundation model by Imageomics that classifies specific species from the cropped animal regions found by MegaDetector.
- **OCR**: Uses EasyOCR with regex pattern matching to parse standardized camera trap timestamps.

## License

This project is open-source and intended for wildlife research and conservation purposes.
