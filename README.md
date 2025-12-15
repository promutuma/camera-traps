# Wildlife Camera Trap Auto-Analyzer

A user-friendly **Streamlit-based dashboard** for automated analysis of wildlife camera trap images. This system extracts metadata via OCR, detects animals using state-of-the-art models (MegaDetector + MobileNet), classifies day/night conditions, and generates downloadable reports.

![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![MegaDetector](https://img.shields.io/badge/MegaDetector-V5%2FV6-blue)

## 🌟 Features

- **📤 Batch Image Upload**: Upload multiple camera trap images (JPG/PNG) simultaneously
- **🔍 OCR Metadata Extraction**: Automatically extracts date, time, and temperature from image metadata strips
- **🦁 Advanced Animal Detection**: Uses **MegaDetector (V6b/V5a)** for detection and **MobileNetV2 (PyTorch)** for classification
- **🔧 Diagnostics Tool**: specialized tab to debug OCR crops and view raw model candidates (all confidence levels)
- **🌓 Day/Night Classification**: Automatically classifies images based on brightness analysis
- **💾 History & Analytics**: Save results to SQLite database and view long-term trends
- **✏️ Editable Results**: Review and manually correct detected animal names in an interactive table
- **📊 Excel Reports**: Generate downloadable Excel (.xlsx) reports with all extracted data

## 🏗️ Project Structure

```
camera-traps/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── wildlife_data.db                # SQLite database (auto-created)
└── core/                           # Processing modules
    ├── __init__.py
    ├── ocr_processor.py           # OCR metadata extraction
    ├── animal_detector.py         # Ensemble (MD + MobileNet via PyTorch)
    ├── day_night_classifier.py    # Day/night classification
    ├── db_manager.py              # Database interactions
    └── image_processor.py         # Unified processing pipeline
```

## 🚀 Installation

### Prerequisites

- **Python 3.11** (Required for ML dependencies on Windows)
- pip package manager

### Setup Steps

1. **Clone or navigate to the project directory:**

   ```bash
   cd c:\wamp64\www\camera-traps
   ```

2. **Create a virtual environment (recommended):**
   _Ensure you use Python 3.11_

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** The first run will download the MobileNetV2 model weights and MegaDetector models. This is a one-time download.

## 💻 Usage

### Running the Application

1. **Start the Streamlit server:**

   ```bash
   streamlit run app.py
   ```

   _(Or `python -m streamlit run app.py` if streamlit is not in PATH)_

2. **Open your browser:**
   - The application will automatically open at `http://localhost:8501`

### Using the Dashboard

1. **Upload Images** (Tab 1):

   - Click "Browse files" to select images
   - Adjust "OCR Strip Height" in sidebar if your metadata is getting cut off
   - Click "🚀 Process Images"

2. **Review Results** (Tab 2):

   - View extracted data and detections
   - Use "Save to History" to persist data to the database

3. **History & Analytics** (Tab 3):

   - View past processing runs and aggregate statistics

4. **Diagnostics** (Tab 4):
   - Debug specific images that failed detection or OCR
   - View raw OCR text and Model internal candidates (red boxes)

## ⚙️ Configuration

### Sidebar Settings

- **Processing Options:**
  - Enable/disable modules (OCR, Det, Day/Night)
- **OCR Settings:**
  - **Metadata Bottom Strip (%)**: Adjust crop area for timestamp reading. Default 0.10.
- **Advanced Settings:**
  - **Detection Confidence**: Threshold to accept predictions (default 0.3)
  - **Brightness Threshold**: Cutoff for day/night (default 100)

## 🔧 Technical Details

### Animal Detection (Ensemble)

- **Primary**: Uses **MegaDetector (YOLOv5)** to find bounding boxes of animals/people/vehicles.
- **Secondary**: Uses **MobileNetV2** (via PyTorch/Torchvision) to classify the species within the box.
- **Fallback**: If MegaDetector fails to load, falls back to full-image MobileNet classification.

### OCR Metadata Extraction

- Uses **EasyOCR** to read text from the configurable bottom strip of images.
- Parses format: `M [Temp] [Date] [Time]` (customizable regex in `core/ocr_processor.py`).

## 🚀 Future Enhancements

- [x] Integration with **MegaDetector**
- [x] Database storage for long-term analysis
- [x] Diagnostics Debugging Tool
- [x] Migration to PyTorch backend
- [ ] Video file processing
- [ ] Batch export of cropped animal detections
- [ ] GPS coordinate extraction

## 📝 License

This project is provided as-is for wildlife research and conservation purposes.
