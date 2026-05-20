"""
Wildlife Camera Trap Auto-Analyzer
A Streamlit application for processing camera trap images.
"""

import os
import sys

if sys.platform == 'win32':
    # Fix Windows console encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    # Prevent the most common Windows PyTorch crash: multiple OpenMP runtimes
    # (libiomp5md.dll from PyTorch/MKL + vcomp*.dll from OpenCV) deadlocking.
    # Must be set before cv2, torch, or easyocr are imported.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    # Allow duplicate OpenMP libs rather than hard-crashing when the constraint
    # above is not enough (e.g. third-party YOLO wheels ship their own copy).
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

    # Block CUDA/GPU driver access entirely on Windows.
    # torch.cuda.is_available() opens a CUDA context even when gpu=False is set.
    # On Windows, this GPU driver probe can TDR-crash the driver hard enough
    # that Windows reboots instantly with no BSOD (common with Optimus/hybrid
    # GPU laptops or outdated drivers).  Setting this before any torch import
    # makes CUDA see zero devices so the driver is never touched.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Required on Windows so that libraries using multiprocessing (EasyOCR, YOLO)
    # can spawn worker processes safely when the entry point is not __main__.
    import multiprocessing as _mp
    _mp.freeze_support()

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import base64
import json
import re
from datetime import datetime
from io import BytesIO
from PIL import Image
import cv2
import logging

# A more aggressive filter to catch Streamlit's missing context warnings 
# which are often spawned dynamically by libraries like tqdm and annoyingly bypass basic log levels.
class NoScriptRunContextFilter(logging.Filter):
    def filter(self, record):
        if "missing ScriptRunContext" in record.getMessage():
            return False
        return True

# Apply to root logger and known Streamlit loggers
logging.getLogger().addFilter(NoScriptRunContextFilter())
logging.getLogger("streamlit").addFilter(NoScriptRunContextFilter())
try:
    from streamlit.runtime.scriptrunner.script_run_context import SCRIPT_RUN_CONTEXT_LOGGER_NAME
    logging.getLogger(SCRIPT_RUN_CONTEXT_LOGGER_NAME).addFilter(NoScriptRunContextFilter())
except ImportError:
    pass

from core.image_processor import ImageProcessor
from core.db_manager import DatabaseManager
from core.animal_detector import AnimalDetector, MegaDetectorWrapper
from core.bioclip_classifier import BioClipClassifier
from core.ocr_processor import OCRProcessor
from core.day_night_classifier import DayNightClassifier
from core.independence_engine import IndependenceEngine
from core.qc_engine import QCEngine
from core.station_manager import StationManager, STRATA
from core.privacy_scrubber import PrivacyScrubber
from core.review_engine import ReviewEngine
from core.community_observer import CommunityObserver, OBSERVATION_TYPES
from core.spatial_exporter import SpatialExporter
from core.species_library import SpeciesLibrary
from core.corridor_analyzer import CorridorAnalyzer
from core.project_config import ProjectConfig, DEFAULT_THRESHOLDS
from core.arcgis_sync import ArcGISSync
import contextlib

class StreamlitLogRedirector:
    """Redirects stdout/stderr to a Streamlit UI element."""
    def __init__(self, container, prefix=""):
        self.container = container
        self.prefix = prefix
        self.buffer = []
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            self.ctx = get_script_run_ctx()
        except ImportError:
            self.ctx = None
        
    def write(self, text):
        if text:
            # text could be bytes from some underlying C streams (like in OpenCV/Torch)
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
                
            if text.strip():
                 # Assign Streamlit context to background threads (e.g. HuggingFace downloads)
                 if self.ctx:
                     try:
                         from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
                         import threading
                         if get_script_run_ctx() is None:
                             add_script_run_ctx(threading.current_thread(), self.ctx)
                     except Exception:
                         pass

                 self.buffer.append(text)
                 # Update the container
                 # We join the last N lines to avoid UI lag if buffer gets huge
                 display_text = "".join(str(item) for item in self.buffer[-20:]) 
                 self.container.code(display_text, language="text")
             
    def flush(self):
        pass

@st.cache_resource(show_spinner="Loading AI Models...")
def load_models_v2(low_spec: bool = False):
    """Load and cache heavy AI models, with optional low-spec optimization."""
    
    # Create a place for logs
    log_expander = st.expander("Show Model Loading Logs", expanded=True)
    with log_expander:
        log_placeholder = st.empty()
        
    redirector = StreamlitLogRedirector(log_placeholder)
    
    # Capture stdio
    with contextlib.redirect_stdout(redirector), contextlib.redirect_stderr(redirector):
        print(f"Starting Model Loading (Low-Spec Mode: {low_spec})...")
        
        # On Windows, always cap PyTorch threads to 1 to prevent the
        # thread-stack exhaustion crash that occurs when EasyOCR + YOLO +
        # BioClip each spin up cpu_count threads simultaneously.
        # On Linux/Mac, only apply the cap in low-spec mode.
        try:
            import torch
            if sys.platform == 'win32' or low_spec:
                torch.set_num_threads(1)
                print(f"PyTorch threads set to 1 ({'Windows' if sys.platform == 'win32' else 'low-spec'} mode).")
        except Exception as t_err:
            print(f"Could not set thread limit: {t_err}")
                
        ocr = OCRProcessor(low_spec=low_spec)
        
        print("Initializing MegaDetector Wrapper...")
        # Initialize with default threshold, can be updated later
        md = MegaDetectorWrapper(confidence_threshold=0.2, low_spec=low_spec)
        
        print("Loading BioClip Classifier...")
        # Add a manual context manager context to avoid streamlit context missing warnings
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            import threading
            add_script_run_ctx(threading.current_thread())
        except ImportError:
            pass
        bio = BioClipClassifier(species_list=AnimalDetector.WILDLIFE_CLASSES, low_spec=low_spec)
        
        print("Initializing Day/Night Classifier...")
        dn = DayNightClassifier()
        print("All models loaded successfully.")
        
    return ocr, md, bio, dn

# Page configuration
st.set_page_config(
    page_title="WildlifeID Pro",
    page_icon="line-chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'image_paths' not in st.session_state:
    st.session_state.image_paths = []
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
if 'independence_data' not in st.session_state:
    st.session_state.independence_data = None  # IDE-enriched DataFrame
if 'ide_summary' not in st.session_state:
    st.session_state.ide_summary = None        # One row per IDE
if 'qc_flags' not in st.session_state:
    st.session_state.qc_flags = None           # QC flags DataFrame
if 'station_manager' not in st.session_state:
    st.session_state.station_manager = StationManager()
if 'review_engine' not in st.session_state:
    st.session_state.review_engine = ReviewEngine()
if 'scrub_audit' not in st.session_state:
    st.session_state.scrub_audit = []      # list of audit dicts from last scrub run
if 'scrubber' not in st.session_state:
    st.session_state.scrubber = PrivacyScrubber()
if 'community_observer' not in st.session_state:
    st.session_state.community_observer = CommunityObserver()
if 'species_library' not in st.session_state:
    st.session_state.species_library = SpeciesLibrary()
if 'spatial_exporter' not in st.session_state:
    st.session_state.spatial_exporter = SpatialExporter()
if 'corridor_analyzer' not in st.session_state:
    st.session_state.corridor_analyzer = CorridorAnalyzer()
if 'project_config' not in st.session_state:
    st.session_state.project_config = ProjectConfig()
if 'arcgis_sync' not in st.session_state:
    st.session_state.arcgis_sync = ArcGISSync()


def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory."""
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(file_path)
    
    return temp_dir, image_paths


def process_images_with_progress(image_paths, processor):
    """Process images and update progress bar."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total = len(image_paths)
    
    for idx, image_path in enumerate(image_paths):
        # Update progress
        progress = (idx + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"Processing image {idx + 1}/{total}: {os.path.basename(image_path)}")
        
        # Process image
        result = processor.process_single_image(image_path)
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)
    
    progress_bar.empty()
    status_text.empty()
    
    return results


def get_trap_nights_map(stations, default: int) -> dict:
    """
    Return {station_id: trap_nights} using deployment records where available,
    falling back to `default` for stations with no deployment history.
    """
    computed = st.session_state.station_manager.compute_trap_nights()
    return {s: computed.get(s, default) for s in stations}


def create_excel_report(df):
    """Create Excel report from dataframe."""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write main data
        # Select columns that exist in the dataframe
        desired_columns = ['filename', 'date', 'time', 'temperature', 
                          'detected_animal', 'detection_confidence', 'detection_method',
                          'day_night', 'brightness', 'user_notes']
        available_columns = [col for col in desired_columns if col in df.columns]
        
        df_export = df[available_columns].copy()
        
        # Rename columns for better readability
        column_mapping = {
            'filename': 'Filename',
            'date': 'Date',
            'time': 'Time',
            'temperature': 'Temperature',
            'detected_animal': 'Detected Animal',
            'detection_confidence': 'Confidence',
            'detection_method': 'Detection Method',
            'day_night': 'Day/Night',
            'brightness': 'Brightness',
            'user_notes': 'User Notes'
        }
        df_export = df_export.rename(columns=column_mapping)
        df_export.to_excel(writer, sheet_name='Wildlife Data', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Wildlife Data']
        for idx, col in enumerate(df_export.columns):
            # Calculate maximum length safely, avoiding TypeErrors on nulls/missing values with newer pandas/arrow dtypes
            lengths = df_export[col].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
            max_length = max(
                lengths.max() if not lengths.empty else 0,
                len(col)
            ) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = max_length
    
    output.seek(0)
    return output


def display_image_with_info(image_path, detections):
    """Display image with detection information overlay and bounding boxes for ALL detections."""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None: return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Draw all bounding boxes
        for det in detections:
            if det.get('bbox') is not None:
                bbox = det['bbox']
                # Convert normalized bbox to pixel coordinates
                x, y, box_w, box_h = bbox
                x1 = int(x * w)
                y1 = int(y * h)
                x2 = int((x + box_w) * w)
                y2 = int((y + box_h) * h)
                
                # Draw green rectangle
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add label background
                label = f"{det.get('detected_animal', 'Unknown')} {det.get('detection_confidence', 0):.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw label background
                cv2.rectangle(image_rgb, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 255, 0), -1)
                cv2.putText(image_rgb, label, (x1 + 5, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        # Add overall detection info overlay (Summary)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Summarize detections
        animal_count = len([d for d in detections if d.get('primary_label') == 'Animal'])
        text = f"Detected: {len(detections)} Objects ({animal_count} Animals)"
        
        # Add semi-transparent background for text
        overlay = image_rgb.copy()
        cv2.rectangle(overlay, (10, 10), (450, 60), (0, 0, 0), -1)
        image_rgb = cv2.addWeighted(overlay, 0.3, image_rgb, 0.7, 0)
        
        # Add text
        cv2.putText(image_rgb, text, (20, 45), font, 0.8, (255, 255, 255), 2)
        
        return Image.fromarray(image_rgb)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        return None


# Main UI
st.markdown('<div class="main-header">Wildlife Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Camera Trap Image Analysis</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Processing Options")
    enable_ocr = st.checkbox("Enable OCR Metadata Extraction", value=True)
    enable_detection = st.checkbox("Enable Animal Detection", value=True)
    enable_day_night = st.checkbox("Enable Day/Night Classification", value=True)
    enable_scrubbing = st.checkbox(
        "Auto-Scrub Person/Vehicle Images",
        value=True,
        help="Blur faces and vehicles after processing. Scrubbed copies are shown in the review UI; originals are preserved."
    )
    enable_low_spec = st.checkbox(
        "Low-Spec / Low-Memory Mode",
        value=False,
        help="Enables dynamic PyTorch INT8 model quantization and reduces thread usage. Recommended for computers with < 8 GB RAM."
    )
    
    # Detection mode selector
    # Professional Pipeline (Fixed Mode)
    st.info("Active Pipeline: MegaDetector V5a + BioClip")
    
    st.subheader("Advanced Settings")
    detection_confidence = st.slider(
        "Detection Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Minimum confidence score to accept animal predictions. Lower this if animals are missed."
    )
    
    brightness_threshold = st.slider(
        "Day/Night Brightness Threshold",
        min_value=0,
        max_value=255,
        value=100,
        step=5,
        help="Pixel brightness threshold (0-255) to distinguish day from night"
    )
    
    ocr_strip_height = st.slider(
        "Metadata Bottom Strip (%)",
        min_value=0.05,
        max_value=0.30,
        value=0.10,
        step=0.01,
        help="Percentage of image height to scan for metadata (bottom margin)"
    )
    
    st.subheader("Privacy & Review Settings")
    blur_strength = st.slider(
        "Blur Strength",
        min_value=11, max_value=101, value=51, step=10,
        help="Gaussian kernel size for privacy blur. Higher = stronger blur."
    )
    review_confidence_threshold = st.slider(
        "Review Queue Threshold",
        min_value=0.1, max_value=1.0, value=0.9, step=0.05,
        help="Animal detections below this confidence appear in the Review Queue."
    )
    reviewer_id = st.text_input("Reviewer ID", value="anonymous",
                                help="Name or ID logged with each review action.")

    st.subheader("Station Settings")
    default_station_id = st.text_input(
        "Default Station ID",
        value="Station-1",
        help="Assigned to images when it cannot be inferred from the filename. "
             "Filenames matching YYYYMMDD_StationID_DeploymentID are parsed automatically."
    )

    independence_window = st.slider(
        "Independence Window (minutes)",
        min_value=5,
        max_value=120,
        value=30,
        step=5,
        help="Same species at the same station within this window = one Independent Detection Event (IDE)."
    )

    trap_nights_input = st.number_input(
        "Default Trap Nights (fallback)",
        min_value=1,
        value=30,
        help="Used when no deployment records exist for a station. Define real deployments in the Stations & Deployments tab."
    )

    st.divider()
    st.info("**Tip:** You can manually edit the detected animal names in the results table below.")
    st.info("**Night Vision Support:** The system automatically detects infrared/grayscale images and classifies them as 'Night' regardless of brightness.")

# Dependency health check — shown once at startup so users know what to install
import importlib.util as _ilu
_missing_deps = []
if _ilu.find_spec("megadetector") is None:
    _missing_deps.append("**MegaDetector** (`megadetector` package not found)")
if _ilu.find_spec("open_clip") is None:
    _missing_deps.append("**BioClip** (`open_clip_torch` package not found)")

if _missing_deps:
    import sys as _sys
    _installer = "install.bat" if _sys.platform == "win32" else "./install.sh"
    st.error(
        "**AI models could not be loaded — animal detection is disabled.**\n\n"
        "Missing packages:\n" + "\n".join(f"- {d}" for d in _missing_deps) + "\n\n"
        f"**Fix:** Re-run the installer from your project folder:\n"
        f"```\n{_installer}\n```\n"
        "Or, if your virtual environment is already activated, install manually:\n"
        "```\npip install megadetector open_clip_torch\npython force_download.py\n```"
    )

# Main content area
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15 = st.tabs([
    "Upload & Process", "Review Results", "Statistics",
    "History & Analytics", "Diagnostics", "Ecological Analytics",
    "QC Dashboard", "Stations & Deployments", "Review Queue",
    "Community Observer", "Spatial & Map", "Species Library",
    "Corridor Analysis", "Project Config", "ArcGIS Sync",
])

with tab1:
    st.header("Upload Camera Trap Images")
    
    uploaded_files = st.file_uploader(
        "Choose images (JPG/PNG)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Select multiple camera trap images to process"
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
        
        # Processing section
        with st.container():
            if st.button("Process Images", type="primary"):
                try:
                    with st.spinner("Initializing processing modules..."):
                        # Initialize processor
                        st.info("**Loading AI Models...**\n\n*Note: The first time you run this, it will download ~1.5 GB of AI models. This may take several minutes depending on your internet connection.*")
                        
                        # Load Cached Models
                        ocr_model, md_model, bio_model, dn_model = load_models_v2(enable_low_spec)
                        
                        # Apply Runtime Settings
                        md_model.set_confidence_threshold(detection_confidence)
                        dn_model.brightness_threshold = brightness_threshold
                        
                        # Create Detector Wrapper
                        animal_detector = AnimalDetector(md_model, bio_model, confidence_threshold=detection_confidence)
                         
                        processor = ImageProcessor(
                            ocr_processor=ocr_model,
                            animal_detector=animal_detector,
                            day_night_classifier=dn_model,
                            ocr_enabled=enable_ocr,
                            detection_enabled=enable_detection,
                            day_night_enabled=enable_day_night,
                            ocr_strip_percent=ocr_strip_height
                        )

                        st.success("Models loaded successfully!")
                    
                    with st.spinner("Processing images..."):
                        # Save uploaded files
                        temp_dir, image_paths = save_uploaded_files(uploaded_files)
                        st.session_state.temp_dir = temp_dir
                        st.session_state.image_paths = image_paths
                        
                        # Process images
                        results = process_images_with_progress(image_paths, processor)
                        
                        # Convert to DataFrame and stamp station_id
                        df = pd.DataFrame(results)
                        if "station_id" not in df.columns:
                            df["station_id"] = default_station_id
                        st.session_state.processed_data = df
                        # Reset derived data whenever new images are processed
                        st.session_state.independence_data = None
                        st.session_state.ide_summary = None
                        st.session_state.qc_flags = None

                    # Auto-scrub Person/Vehicle images if enabled
                    if enable_scrubbing:
                        with st.spinner("Scrubbing Person/Vehicle images for privacy..."):
                            scrubber = PrivacyScrubber(blur_strength=blur_strength)
                            st.session_state.scrubber = scrubber
                            audit = scrubber.scrub_batch(st.session_state.processed_data)
                            st.session_state.scrub_audit = audit
                            scrubbed_count = sum(1 for a in audit if not a.get("skipped"))
                            if scrubbed_count:
                                st.info(f"Privacy: {scrubbed_count} image(s) scrubbed.")
                        
                        # st.balloons()
                        st.success("Processing complete! Switch to the 'Review Results' tab to see the data.")
                
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.error(f"Error type: {type(e).__name__}")
                    import traceback
                    st.code(traceback.format_exc())


with tab2:
    if st.session_state.processed_data is not None:
        st.header("Review and Edit Results")
        
        # Initialize viewing state
        if 'view_mode' not in st.session_state:
            st.session_state.view_mode = 'gallery'  # 'gallery' or 'inspector'
        if 'current_image_index' not in st.session_state:
            st.session_state.current_image_index = 0

        raw_df = st.session_state.processed_data

        # --- Aggregation Logic for Display ---
        # We want one row per image.
        # We need to aggregate species labels and confidences.
        if 'filepath' not in raw_df.columns and hasattr(st.session_state, 'image_paths'):
             # Fallback repair if needed (though we fixed it locally previously)
             pass 

        # Create distinct image view
        # We group by filepath and aggregate
        # First ensure we have necessary cols
        if 'detected_animal' not in raw_df.columns: raw_df['detected_animal'] = 'Unknown'
        
        # Helper to join unique strings
        def join_unique(series):
            # Clean species strings: Remove confidence scores (e.g., " 0.98", " 1.00", " (Low Conf)")
            cleaned_vals = []
            for s in series:
                if s and s != 'N/A' and s != 'Unknown':
                    # Regex: Space followed by digits, dot, digits OR " (Low Conf)"
                    # e.g. "Impala 0.95" -> "Impala"
                    cleaned = re.sub(r'\s\d+(\.\d+)?', '', str(s)) 
                    cleaned = cleaned.replace(" (Low Conf)", "").strip()
                    if cleaned:
                        cleaned_vals.append(cleaned)
            
            unique_vals = sorted(set(cleaned_vals))
            return ", ".join(unique_vals) if unique_vals else "Empty"

        # Aggregation rules
        agg_rules = {
            'filename': 'first',
            'date': 'first', 
            'time': 'first',
            'temperature': 'first',
            'day_night': 'first',
            'brightness': 'first',
            'user_notes': 'first',  # Assuming notes are per-image
            'detection_confidence': 'max', # Show max confidence
            'primary_label': lambda x: sorted(x.tolist())[0], # Simple pick
            'species_label': join_unique, # List all species (String) - Cleaned
            'species_data': lambda x: [item for sublist in x for item in (sublist if isinstance(sublist, list) else [])], # Aggregated list of all detection objects (species+bbox)
            'raw_text': 'first', # Keep raw OCR text
            'image_id': 'first', # Unique ID
            'md_confidence': lambda x: list(x), # List of MD confidences
            'md_bbox': lambda x: list(x), # List of bboxes
            'md_category': 'first',
            'bioclip_confidence': lambda x: list(x) # List of BioClip confidences
        }
        
        # Only aggregate existing columns
        valid_rules = {k: v for k, v in agg_rules.items() if k in raw_df.columns}
        
        # Group by filepath
        if 'filepath' in raw_df.columns:
            display_df = raw_df.groupby('filepath', as_index=False).agg(valid_rules)
        else:
            # Fallback if no filepath column (shouldn't happen with valid data)
            display_df = raw_df.drop_duplicates(subset=['filename']) 
        
        # --- Top Control Bar ---
        col_ctrl_1, col_ctrl_2, col_ctrl_3 = st.columns([2, 1, 1])
        with col_ctrl_1:
            view_mode = st.radio("View Mode", ["Gallery View", "Inspector View"], horizontal=True, label_visibility="collapsed")
            st.session_state.view_mode = 'gallery' if view_mode == "Gallery View" else 'inspector'
            
        with col_ctrl_2:
            st.caption(f"Total Images: {len(display_df)}")
            
        # --- Advanced Sidebar Filters ---
        st.sidebar.markdown("### Filter Options")
        
        # 1. Species Filter
        # Extract unique species from the 'detected_animal' or primary label column
        all_species = sorted(display_df['primary_label'].unique().tolist())
        # Try to get more granular species from species_label if available
        # This is a bit tricky with aggregation, but let's stick to primary labels for now 
        # as they are cleaner for filtering.
        
        selected_species = st.sidebar.multiselect(
            "Species",
            options=all_species,
            default=[]
        )
        
        # 2. Confidence Filter
        min_conf, max_conf = st.sidebar.slider(
            "Confidence Range",
            min_value=0.0, max_value=1.0, value=(0.0, 1.0)
        )
        
        # 3. Day/Night Filter
        day_night_opts = ["All", "Day", "Night"]
        day_night_filter = st.sidebar.radio("Time of Day", day_night_opts)
        
        # 4. Date Filter
        # Convert date strings to datetime objects if possible for filtering
        try:
            # Clean date column
            display_df['date_dt'] = pd.to_datetime(display_df['date'], errors='coerce')
            min_date = display_df['date_dt'].min()
            max_date = display_df['date_dt'].max()
            
            if pd.notnull(min_date) and pd.notnull(max_date):
                date_range = st.sidebar.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                date_range = None
        except Exception:
            date_range = None
            
        # Apply Filters
        filtered_display_df = display_df.copy()
        
        # Species
        if selected_species:
            filtered_display_df = filtered_display_df[filtered_display_df['primary_label'].isin(selected_species)]
            
        # Confidence
        filtered_display_df = filtered_display_df[
            (filtered_display_df['detection_confidence'] >= min_conf) & 
            (filtered_display_df['detection_confidence'] <= max_conf)
        ]
        
        # Day/Night
        if day_night_filter != "All":
            filtered_display_df = filtered_display_df[filtered_display_df['day_night'] == day_night_filter]
            
        # Date
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            # Ensure row date is comparable
            mask = (filtered_display_df['date_dt'].dt.date >= start_date) & (filtered_display_df['date_dt'].dt.date <= end_date)
            filtered_display_df = filtered_display_df[mask]
            
        st.sidebar.markdown(f"**Showing {len(filtered_display_df)} / {len(display_df)} images**")
        

        # --- Statistics Dashboard ---
        st.markdown("### Overview")
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        
        total_imgs = len(display_df)
        total_animals = len(display_df[display_df['primary_label'] == 'Animal'])
        total_vehicles = len(display_df[display_df['primary_label'] == 'Vehicle'])
        total_persons = len(display_df[display_df['primary_label'] == 'Person'])
        
        stat_c1.metric("Total Images", total_imgs)
        stat_c2.metric("Animals Detected", total_animals)
        stat_c3.metric("Vehicles", total_vehicles)
        stat_c4.metric("Persons", total_persons)
        
        st.divider()

        # --- VIEW MODES ---
        
        if st.session_state.view_mode == 'gallery':
            c_head, c_grid = st.columns([3, 1])
            with c_head:
                st.subheader("Image Gallery")
            with c_grid:
                grid_cols = st.slider("Grid Size", 2, 6, 4)
            
            cols = st.columns(grid_cols)
            for idx, row in filtered_display_df.iterrows():
                col = cols[idx % grid_cols]
                with col:
                    img_path = row['filepath']
                    display_img_path = st.session_state.scrubber.get_display_path(str(img_path)) if img_path else img_path

                    if img_path and os.path.exists(str(img_path)):
                        st.image(display_img_path, width="stretch")
                        if st.button(f"Inspect", key=f"btn_inspect_{idx}"):
                            st.session_state.view_mode = 'inspector'
                            # Find index in full display_df
                            # Reset index to find the integer loc
                            matches = display_df.index[display_df['filepath'] == img_path].tolist()
                            if matches:
                                st.session_state.current_image_index = matches[0]
                            st.rerun()
                        # Show aggregated species
                        label_display = row['species_label'] if row['species_label'] else row['primary_label']
                        
                        # Info Overlay (Caption)
                        date_str = row.get('date', '') if row.get('date') else ''
                        time_str = row.get('time', '') if row.get('time') else ''
                        st.caption(f"**{label_display}**\n{date_str} {time_str}")
                    else:
                        st.warning("Image missing")

        else: # Inspector View
            st.subheader("Inspector View (One Record per Image)")
            
            # Navigation
            col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 1])
            total_imgs = len(display_df)
            
            with col_nav_1:
                if st.button("<-Previous"):
                    st.session_state.current_image_index = max(0, st.session_state.current_image_index - 1)
            with col_nav_2:
                if total_imgs > 1:
                    new_idx = st.slider("Jump to Image", 0, total_imgs-1, st.session_state.current_image_index)
                    st.session_state.current_image_index = new_idx
                else:
                    st.caption("Single record view")
            with col_nav_3:
                if st.button("Next ->"):
                    st.session_state.current_image_index = min(total_imgs-1, st.session_state.current_image_index + 1)

            # Content
            current_idx = st.session_state.current_image_index
            if 0 <= current_idx < total_imgs:
                row = display_df.iloc[current_idx]
                image_path = row['filepath']
                    
                # --- 1. Image Display (Top) ---
                show_box = st.toggle("Show Bounding Box", value=True)
                
                # Gather ALL detections for this image from the RAW dataframe
                image_records = raw_df[raw_df['filepath'] == image_path].to_dict('records')
                
                if not show_box: 
                    for rec in image_records: rec['bbox'] = None
                
                display_img = display_image_with_info(image_path, image_records)
                if display_img:
                    st.image(display_img, width="stretch")
                

                # --- Auto-Repair Logic (Lazy Analysis) ---
                # Calculate ID of the image we are viewing
                current_hash = ImageProcessor.get_image_hash(image_path)
                
                # Check if the data is stale (missing species_data)
                needs_update = False
                if 'image_id' not in row or not row['image_id'] or row['image_id'] != current_hash:
                     needs_update = True
                elif isinstance(row.get('species_data'), list) and len(row.get('species_data')) == 0 and row['primary_label'] == 'Animal' and row['detection_confidence'] > 0:
                     pass

                if 'image_id' not in row or pd.isna(row['image_id']):
                     needs_update = True
                
                if needs_update:
                    st.info("Auto-linking data to image content (First run for this file)...")
                    # Auto-run analysis
                    with st.spinner("Analyzing high-resolution details..."):
                        ocr_model, md_model, bio_model, dn_model = load_models_v2(enable_low_spec)
                        md_model.set_confidence_threshold(detection_confidence)
                        dn_model.brightness_threshold = brightness_threshold
                        animal_detector = AnimalDetector(md_model, bio_model, confidence_threshold=detection_confidence)
                        processor = ImageProcessor(
                            ocr_processor=ocr_model, animal_detector=animal_detector, day_night_classifier=dn_model,
                            ocr_enabled=enable_ocr, detection_enabled=enable_detection, day_night_enabled=enable_day_night, ocr_strip_percent=ocr_strip_height
                        )
                        
                        new_results = processor.process_single_image(image_path)
                        
                        # Update Dataframe
                        current_df = st.session_state.processed_data
                        # Remove old by filepath (fallback)
                        current_df = current_df[current_df['filepath'] != image_path]
                        
                        if new_results:
                            new_df = pd.DataFrame(new_results)
                            current_df = pd.concat([current_df, new_df], ignore_index=True)
                            st.session_state.processed_data = current_df
                            st.rerun()
                        
                    # --- 2. Details & Editing (Below Image) ---
                    st.divider()
                    st.subheader("Details & Edits")
                    
                    # --- Editing Form ---
                    with st.form(key=f"edit_form_{current_idx}"):
                        st.caption("Detailed Detections List (Edit Species and Primary Label)")
                        
                        current_species_data = row.get('species_data', [])
                        if not isinstance(current_species_data, list): current_species_data = [] 
                        
                        # Configure column config for better UX
                        column_cfg = {
                            "primary_label": st.column_config.SelectboxColumn("Primary Label", options=["Animal", "Person", "Vehicle", "Empty"], required=True),
                            "species_label": st.column_config.TextColumn("Species Name", required=True, width="medium"),
                            "detection_confidence": st.column_config.NumberColumn("Conf (0-1)", min_value=0.0, max_value=1.0, format="%.2f"),
                            "detected_animal": st.column_config.TextColumn("Original Detection", disabled=True), # Read-only original
                            "bbox": st.column_config.TextColumn("BBox", disabled=True) # Hide or disable
                        }

                        edited_species_data = st.data_editor(
                            current_species_data,
                            column_config=column_cfg,
                            num_rows="dynamic",
                            width="stretch",
                            key=f"species_editor_{current_idx}"
                        )
                        
                        # User Notes
                        current_notes = row.get('user_notes')
                        if pd.isna(current_notes): current_notes = ""
                        new_notes = st.text_area("User Notes", value=str(current_notes), height=100)
                        
                        # Validation Flag
                        is_verified = "[VERIFIED]" in str(current_notes)
                        new_verified = st.checkbox("Mark as Verified", value=is_verified)
                        
                        # Save Button
                        if st.form_submit_button("Save Changes"):
                             # Reconstruct notes
                            final_notes = new_notes
                            if new_verified and "[VERIFIED]" not in final_notes:
                                final_notes = f"[VERIFIED] {final_notes}".strip()
                            elif not new_verified:
                                final_notes = final_notes.replace("[VERIFIED]", "").strip()

                            # Reconstruct string label for display and infer main Primary Label
                            new_species_label_str = ""
                            inferred_primary = "Empty" # Default
                            
                            if edited_species_data:
                                parts = []
                                primary_votes = []
                                for item in edited_species_data:
                                    s = item.get('species_label', 'Unknown')
                                    c = item.get('detection_confidence', 0.0)
                                    p = item.get('primary_label', 'Animal')
                                    parts.append(f"{s} {c:.2f}")
                                    primary_votes.append(p)
                                new_species_label_str = ", ".join(parts)
                                
                                # Infer Main Primary Label from table rows
                                # Priority: Person > Vehicle > Animal > Empty
                                if "Person" in primary_votes:
                                    inferred_primary = "Person"
                                elif "Vehicle" in primary_votes:
                                    inferred_primary = "Vehicle"
                                elif "Animal" in primary_votes:
                                    inferred_primary = "Animal"
                                else:
                                    inferred_primary = "Empty"
                            else:
                                new_species_label_str = "Empty"
                                inferred_primary = "Empty"
                            
                            # Update DataFrame (All rows for this file)
                            mask = st.session_state.processed_data['filepath'] == image_path
                            
                            # Update columns
                            st.session_state.processed_data.loc[mask, 'primary_label'] = inferred_primary
                            st.session_state.processed_data.loc[mask, 'user_notes'] = final_notes
                            
                            # Update species data for all rows
                            for i in st.session_state.processed_data.index[mask]:
                                st.session_state.processed_data.at[i, 'species_data'] = edited_species_data
                                st.session_state.processed_data.at[i, 'species_label'] = new_species_label_str
                            
                            st.success("Changes saved successfully!")
                            st.rerun()

                    # Read-only Info (Metadata)
                    st.markdown("---")
                    st.caption(f"**Date:** {row.get('date', 'Unknown')}")
                    st.caption(f"**Time:** {row.get('time', 'Unknown')}")
                    st.caption(f"**Temp:** {row.get('temperature', 'Unknown')}")
                    st.caption(f"**Condition:** {row.get('day_night', 'Unknown')}")
                    
                    # (Removed redundant static species list since we have the editor above)


                    st.divider()
                    st.metric("Max Confidence", f"{row['detection_confidence']:.2%}")
                    st.metric("Time", str(row['time']))
                    st.metric("Temperature", str(row['temperature']))
                    
                    st.caption("Raw OCR Text")
                    st.text_area("Raw Metadata", row.get('raw_text', 'N/A'), height=68, disabled=True, label_visibility="collapsed")
            else:
                st.info("No image selected.")

        st.divider()
        # Bulk Editor (Read-Only or Limited)
        st.subheader("Results Summary (One Row per Image)")
        st.dataframe(
            display_df,
            column_order=[
                "image_id", "filename", "species_label", "primary_label", 
                "detection_confidence", "date", "time", "day_night", 
                "temperature", "brightness", "user_notes", "raw_text"
            ],
            column_config={
                 "detection_confidence": st.column_config.ProgressColumn("Max Conf", min_value=0, max_value=1),
                 "species_label": st.column_config.TextColumn("Species List"),
                 "image_id": st.column_config.TextColumn("Image ID", help="Unique content hash"),
                 "raw_text": st.column_config.TextColumn("OCR Text", width="small")
            },
            width="stretch",
            hide_index=True
        )

        # Download / Save Area
        st.subheader("out Export & Save")
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            excel_data = create_excel_report(st.session_state.processed_data)
            st.download_button("Download Excel", data=excel_data, file_name=f"report_{datetime.now().strftime('%Y%m%d')}.xlsx")
        
        with c2:
            # Custom JSON generation with Base64 and specific structure
            def generate_custom_json():
                export_list = []
                # Use aggregated view logic to ensure unique images
                # (Simple approach: iterate unique filepaths)
                df = st.session_state.processed_data
                unique_files = df['filepath'].unique()
                
                for fp in unique_files:
                    # Get rows for this file
                    rows = df[df['filepath'] == fp]
                    first_row = rows.iloc[0]
                    
                    # 1. Base64 Encode Image
                    try:
                        with open(fp, "rb") as img_f:
                            b64_str = base64.b64encode(img_f.read()).decode('utf-8')
                    except Exception:
                        b64_str = None
                        
                    # 3. Build Detections Array
                    detections = []
                    
                    id_counter = 1
                    for _, row in rows.iterrows():
                        # Each row behaves as a "Detection"
                        # We use the 'species_data' list mainly to get the confidence/names if we need detail
                        # BUT, effectively, the row itself is the primary detection unit (Animal, Person, etc)
                        
                        # However, 'species_label' column in the row holds the "Baboon 0.48, Warthog 0.39"
                        
                        det_item = {
                            "detection_id": id_counter,
                            "primary_label": row['primary_label'],
                            "detected_animal": row['detected_animal'],
                            # We want the rich string with confidence here, so we use species_label column 
                            # (which we ensure has confidence in Save logic).
                            # If it was stripped for view, we might need to rely on species_data reconstruction
                            "species_label": row['species_label'], 
                            "detection_confidence": float(row['detection_confidence']),
                            "bbox": row['bbox'],
                            "detection_method": row.get('detection_method', 'Unknown')
                        }
                        detections.append(det_item)
                        id_counter += 1
                    
                    # 4. Construct Item
                    item = {
                        "image_id": first_row.get('image_id'),
                        "filename": first_row['filename'],
                        "filepath": first_row['filepath'],
                        "image_data": b64_str,
                        "temperature": first_row.get('temperature'),
                        "date": first_row.get('date'),
                        "time": first_row.get('time'),
                        "day_night": first_row.get('day_night'),
                        "brightness": float(first_row.get('brightness', 0)),
                        "user_notes": first_row.get('user_notes', ''),
                        "processing_status": first_row.get('processing_status', 'Success'),
                        "raw_text": first_row.get('raw_text'),
                        "detections": detections
                    }
                    export_list.append(item)
                
                return json.dumps(export_list, indent=2)

            json_str = generate_custom_json()
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        with c3:
            if st.button("Save to Database", type="secondary"):
                count = st.session_state.db_manager.save_results(st.session_state.processed_data)
                st.success(f"Saved {count} records!")
    
    else:
        st.info("Please upload and process images first.")

with tab3:
    if st.session_state.processed_data is not None:
        st.header("Analysis Statistics")
        
        df = st.session_state.processed_data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", len(df))
        
        with col2:
            identified = len(df[df['primary_label'] == 'Animal'])
            st.metric("Animals Identified", identified)
        
        with col3:
            day_count = len(df[df['day_night'] == 'Day'])
            st.metric("Day Images", day_count)
        
        with col4:
            night_count = len(df[df['day_night'] == 'Night'])
            st.metric("Night Images", night_count)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Species Distribution")
            # Filter for animals only
            animal_df = df[df['primary_label'] == 'Animal']
            species_counts = animal_df['species_label'].value_counts()
            st.bar_chart(species_counts)
        
        with col2:
            st.subheader("Day/Night Distribution")
            day_night_counts = df['day_night'].value_counts()
            st.bar_chart(day_night_counts)
        
        # Detection confidence distribution
        st.subheader("Detection Confidence Distribution")
        st.line_chart(df['detection_confidence'])
        
    else:
        st.info("Please upload and process images first in the 'Upload & Process' tab.")

with tab4:
    st.header("Analysis History")
    
    # Load history
    history_df = st.session_state.db_manager.get_history_df()
    
    if not history_df.empty:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("Total Historical Records", len(history_df))
        with col2:
            unique_species = history_df['detected_animal'].nunique()
            st.metric("Unique Species", unique_species)
            
        st.dataframe(
            history_df,
            column_config={
                "processed_at": "Processed Date",
                "filename": "Filename",
                "detected_animal": "Species",
                "detection_confidence": st.column_config.ProgressColumn(
                    "Confidence", format="%.2f", min_value=0, max_value=1
                ),
                "day_night": "Day/Night",
                "temperature": "Temp",
                "user_notes": "Notes"
            },
            width="stretch",
            hide_index=True
        )
        
        # Helper to convert df to csv
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(history_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Export Full History (CSV)",
                data=csv,
                file_name=f"wildlife_history_export_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        with col2:
            if st.button("️ Clear History", type="secondary"):
                if st.checkbox("Confirm Clear History? This cannot be undone."):
                    st.session_state.db_manager.clear_history()
                    st.rerun()
    else:
        st.info("No historical data found. Process and save images to build your history!")

with tab5:
    st.header("Deep Inspection Tool")
    st.warning("This tool helps debug model performance issues. It runs models without filtering to show all candidates.")
    
    debug_mode = st.radio("Select Image Source", ["Processed Image", "Upload Debug Image"], horizontal=True)
    target_debug_img = None
    
    if debug_mode == "Processed Image":
        if 'image_paths' in st.session_state and st.session_state.image_paths:
            target_debug_img = st.selectbox("Select an image to inspect", st.session_state.image_paths, format_func=lambda x: os.path.basename(x))
        else:
            st.info("No processed images available. Please upload and process images in the first tab.")
    else:
        debug_upload = st.file_uploader("Upload an image for debugging", type=['jpg', 'jpeg', 'png'])
        if debug_upload:
            temp_debug_dir = os.path.join(st.session_state.get('temp_dir', 'temp'), 'debug')
            os.makedirs(temp_debug_dir, exist_ok=True)
            target_debug_img = os.path.join(temp_debug_dir, debug_upload.name)
            with open(target_debug_img, "wb") as f:
                f.write(debug_upload.getbuffer())
    
    if target_debug_img:
        col_dbg_1, col_dbg_2 = st.columns([1, 2])
        with col_dbg_1:
            st.image(target_debug_img, caption="Target Image", width="stretch")
            
            ocr_strip_pct = st.slider("OCR Strip Height (%)", 0.05, 0.30, 0.10, 0.01, help="Adjust crop area for metadata")
            
        with col_dbg_2:
            if st.button("Run Deep Inspection", type="primary"):
                with st.spinner("Analyzing internals..."):
                    # Init specific debug processor
                    ocr_model, md_model, bio_model, dn_model = load_models_v2(enable_low_spec)
                    
                    # Ensure debug settings
                    # md_model.set_confidence_threshold(0.0) # Debug often wants low threshold, but detect_all handles this independently
                    
                    animal_detector = AnimalDetector(md_model, bio_model)
                    
                    debug_processor = ImageProcessor(
                        ocr_processor=ocr_model,
                        animal_detector=animal_detector,
                        day_night_classifier=dn_model,
                        ocr_enabled=True,
                        detection_enabled=True,
                        day_night_enabled=True
                    )
                    
                    debug_info = debug_processor.get_debug_info(target_debug_img, ocr_strip_percent=ocr_strip_pct)
                    
                    # 1. OCR Debug
                    st.subheader("1. OCR Data Extraction")
                    ocr_data = debug_info.get('ocr')
                    if ocr_data:
                        st.image(ocr_data['crop'], caption="Metadata Strip used for OCR")
                        st.text(f"Raw Text: {ocr_data['raw_text']}")
                        st.write("Parsed Metadata:", ocr_data['parsed'])
                    else:
                        st.error("OCR Debug data missing")
                        
                    st.divider()
                    
                    # 2. MegaDetector Internals
                    st.subheader("2. MegaDetector Raw Candidates")
                    
                    md_status = debug_info.get('megadetector_status')
                    if md_status:
                        if md_status.get('error'):
                            st.error(f"MegaDetector Failed to Load: {md_status['error']}")
                            st.info("Try restarting the app to pick up recent fixes.")
                        elif not md_status.get('loaded'):
                             st.warning("MegaDetector not loaded (unknown reason).")
                        else:
                             st.success("MegaDetector Loaded Successfully")
                    st.markdown("Shows **ALL** detections found by the model, including those below the confidence threshold.")
                    md_data = debug_info.get('megadetector')
                    if md_data:
                        md_df = pd.DataFrame(md_data)
                        st.dataframe(md_df, width="stretch")
                        
                        # Visualizer
                        img_vis = cv2.imread(target_debug_img)
                        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
                        h, w, _ = img_vis.shape
                        
                        for d in md_data:
                            bbox = d.get('bbox')
                            if bbox:
                                x, y, bw, bh = bbox
                                x, y, bw, bh = int(x * w), int(y * h), int(bw * w), int(bh * h)
                                conf = d.get('conf', 0)
                                color = (255, 0, 0) # Red
                                if conf > 0.2: color = (255, 165, 0) # Orange
                                if conf > 0.8: color = (0, 255, 0) # Green
                                
                                cv2.rectangle(img_vis, (x, y), (x+bw, y+bh), color, 2)
                                label = f"{conf:.2f} {d.get('category', '?')}"
                                cv2.putText(img_vis, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                        st.image(img_vis, caption="Visualized Raw Detections (Red=<0.2, Orange=<0.8, Green=>0.8)", width="stretch")
                    else:
                        st.warning("No MegaDetector candidates found.")

                    st.divider()

                    # 3. BioClip Classifier
                    st.subheader("3. BioClip Top-20 Predictions")
                    bc_data = debug_info.get('bioclip')
                    if bc_data:
                        # Convert list of tuples to DataFrame
                        bc_df = pd.DataFrame(bc_data, columns=['Species', 'Confidence'])
                        # Sort by confidence
                        bc_df = bc_df.sort_values(by='Confidence', ascending=False)
                        st.dataframe(
                            bc_df, 
                            column_config={
                                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.4f")
                            },
                            width="stretch"
                        )
                    else:
                        st.warning("BioClip data unavailable")

with tab6:
    st.header("Ecological Analytics")
    st.markdown(
        "All indicators are derived from **Independent Detection Events (IDEs)**. "
        "Run the IDE computation first, then explore each indicator below."
    )

    if st.session_state.processed_data is None:
        st.info("Process images first in the Upload & Process tab.")
    else:
        run_col, info_col = st.columns([1, 3])
        with run_col:
            run_ide = st.button("Compute IDEs", type="primary")
        with info_col:
            st.caption(
                f"Window: **{independence_window} min** | "
                f"Station default: **{default_station_id}** | "
                f"Trap nights (for RAI): **{trap_nights_input}**"
            )

        if run_ide:
            with st.spinner("Applying independence rule..."):
                engine = IndependenceEngine(window_minutes=independence_window)
                enriched = engine.compute_ides(
                    st.session_state.processed_data,
                    default_station=default_station_id
                )
                summary = engine.get_ide_summary(enriched)
                st.session_state.independence_data = enriched
                st.session_state.ide_summary = summary

        if st.session_state.ide_summary is not None:
            summary = st.session_state.ide_summary
            enriched = st.session_state.independence_data

            # --- Headline metrics ---
            m1, m2, m3, m4 = st.columns(4)
            animal_ides = summary[summary["species"].str.lower() != "empty"] if not summary.empty else summary
            m1.metric("Total IDEs", len(summary))
            m2.metric("Animal IDEs", len(animal_ides))
            m3.metric("Unique Species", summary["species"].nunique() if not summary.empty else 0)
            m4.metric("Stations", summary["station_id"].nunique() if not summary.empty else 0)

            st.divider()

            # --- IDE Summary Table ---
            st.subheader("IDE Summary Table")
            display_summary = summary.copy()
            if "first_detection" in display_summary.columns:
                display_summary["first_detection"] = display_summary["first_detection"].astype(str)
                display_summary["last_detection"] = display_summary["last_detection"].astype(str)

            st.dataframe(
                display_summary,
                column_config={
                    "ide_id": st.column_config.TextColumn("IDE ID"),
                    "station_id": "Station",
                    "species": "Species",
                    "first_detection": "First Detection",
                    "last_detection": "Last Detection",
                    "duration_minutes": st.column_config.NumberColumn("Duration (min)", format="%.1f"),
                    "image_count": "Images",
                    "max_confidence": st.column_config.ProgressColumn(
                        "Max Confidence", min_value=0, max_value=1, format="%.2f"
                    ),
                },
                column_order=[
                    "ide_id", "station_id", "species", "first_detection",
                    "last_detection", "duration_minutes", "image_count", "max_confidence"
                ],
                hide_index=True,
                width="stretch",
            )

            st.divider()

            # --- IDE counts per species bar chart ---
            st.subheader("IDEs per Species")
            if not summary.empty:
                species_counts = (
                    summary[summary["species"].str.lower() != "empty"]
                    .groupby("species")
                    .size()
                    .reset_index(name="IDE Count")
                    .sort_values("IDE Count", ascending=False)
                )
                st.bar_chart(species_counts.set_index("species")["IDE Count"])

            st.divider()

            # --- RAI table ---
            st.subheader("Relative Abundance Index (RAI)")
            st.caption(
                "RAI = Independent Detection Events / Trap Nights. "
                "Edit trap nights per station in the sidebar."
            )

            if not summary.empty:
                engine = IndependenceEngine(window_minutes=independence_window)
                stations = summary["station_id"].unique().tolist()
                trap_nights_map = get_trap_nights_map(stations, trap_nights_input)
                rai_df = engine.compute_rai(summary, trap_nights_map)

                if not rai_df.empty:
                    st.dataframe(
                        rai_df,
                        column_config={
                            "station_id": "Station",
                            "species": "Species",
                            "ide_count": "IDEs",
                            "trap_nights": "Trap Nights",
                            "rai": st.column_config.NumberColumn("RAI", format="%.4f"),
                        },
                        hide_index=True,
                        width="stretch",
                    )

            st.divider()

            # --- IDE timeline per station ---
            st.subheader("Detection Timeline")
            if not summary.empty and "first_detection" in summary.columns:
                timeline_df = summary.dropna(subset=["first_detection"]).copy()
                timeline_df["first_detection"] = pd.to_datetime(timeline_df["first_detection"], errors="coerce")
                timeline_df = timeline_df.dropna(subset=["first_detection"])

                if not timeline_df.empty:
                    # Count IDEs per date per species
                    timeline_df["date"] = timeline_df["first_detection"].dt.date
                    pivot = (
                        timeline_df.groupby(["date", "species"])
                        .size()
                        .unstack(fill_value=0)
                    )
                    st.line_chart(pivot)
                else:
                    st.info("Timestamps could not be parsed — check that OCR extracted dates correctly.")

            st.divider()

            # --- Species Richness ---
            st.subheader("Species Richness")
            richness_df = engine.compute_species_richness(summary)
            if not richness_df.empty:
                r1, r2 = st.columns([2, 1])
                with r1:
                    st.dataframe(
                        richness_df,
                        column_config={
                            "station_id": "Station",
                            "species_count": "Total Species",
                            "singleton_count": "Singletons",
                            "species_list": st.column_config.TextColumn("Species Detected", width="large"),
                        },
                        hide_index=True,
                        width="stretch",
                    )
                with r2:
                    st.bar_chart(richness_df.set_index("station_id")["species_count"])

            st.divider()

            # --- Species Accumulation Curve ---
            st.subheader("Species Accumulation Curve")
            st.caption("Curve approaching flat = sampling effort is adequate.")
            accum_df = engine.compute_species_accumulation(summary)
            if not accum_df.empty:
                st.line_chart(accum_df.set_index("cumulative_ides")["cumulative_species"])

            st.divider()

            # --- Mean Group Size ---
            st.subheader("Mean Group Size per Species")
            group_df = engine.compute_mean_group_size(enriched)
            if not group_df.empty:
                st.dataframe(
                    group_df,
                    column_config={
                        "station_id": "Station",
                        "species": "Species",
                        "mean_group_size": st.column_config.NumberColumn("Mean Group Size", format="%.2f"),
                        "min_group": "Min",
                        "max_group": "Max",
                        "std_group": st.column_config.NumberColumn("Std Dev", format="%.2f"),
                        "ide_count": "IDEs",
                        "outlier_ides": st.column_config.NumberColumn("Outlier IDEs", help="Group size > mean + 3×std"),
                    },
                    column_order=["station_id", "species", "mean_group_size", "min_group",
                                  "max_group", "std_group", "ide_count", "outlier_ides"],
                    hide_index=True,
                    width="stretch",
                )

            st.divider()

            # --- Visitation Rate & Time-of-Day Heatmap ---
            st.subheader("Visitation Rate")
            trap_nights_map = get_trap_nights_map(summary["station_id"].unique().tolist(), trap_nights_input)
            visit_df, heatmap_df = engine.compute_visitation_rate(summary, trap_nights_map)
            if not visit_df.empty:
                st.dataframe(
                    visit_df,
                    column_config={
                        "station_id": "Station",
                        "species": "Species",
                        "visit_count": "Visits (IDEs)",
                        "trap_nights": "Trap Nights",
                        "visit_rate": st.column_config.NumberColumn("Visit Rate", format="%.4f"),
                        "diurnal_pct": st.column_config.NumberColumn("Diurnal %", format="%.1f"),
                        "nocturnal_pct": st.column_config.NumberColumn("Nocturnal %", format="%.1f"),
                    },
                    hide_index=True,
                    width="stretch",
                )

            if not heatmap_df.empty:
                st.subheader("Time-of-Day Activity Heatmap (IDEs per 2-hour block)")
                st.dataframe(heatmap_df, width="stretch")

            st.divider()

            # --- Export ---
            st.subheader("Export")
            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                csv_bytes = summary.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download IDE Summary (CSV)",
                    data=csv_bytes,
                    file_name=f"ide_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            with ex_col2:
                if st.button("Save IDEs to Database"):
                    saved = st.session_state.db_manager.save_independence_events(summary)
                    st.success(f"Saved {saved} new IDE records to database.")

with tab7:
    st.header("QC Dashboard")
    st.markdown(
        "Automated quality-control checks against the current session data. "
        "Run after processing images and — ideally — after computing IDEs."
    )

    if st.session_state.processed_data is None:
        st.info("Process images first in the Upload & Process tab.")
    else:
        # --- QC configuration (inline) ---
        with st.expander("QC Settings", expanded=False):
            qc_col1, qc_col2 = st.columns(2)
            with qc_col1:
                qc_min_effort = st.number_input(
                    "Min trap nights (INSUFFICIENT_EFFORT)", min_value=1, value=60
                )
                qc_min_func = st.number_input(
                    "Min functionality % (LOW_FUNCTIONALITY)", min_value=1.0, max_value=100.0, value=90.0
                )
                qc_min_zero = st.number_input(
                    "Min nights for zero-detection flag (POSSIBLE_FAILURE)", min_value=1, value=10
                )
            with qc_col2:
                qc_survey_start = st.date_input("Survey start (OUT_OF_RANGE, optional)", value=None)
                qc_survey_end   = st.date_input("Survey end (OUT_OF_RANGE, optional)", value=None)
                qc_drift = st.number_input(
                    "Clock drift threshold (minutes, TIMESTAMP_INCONSISTENCY)", min_value=1.0, value=5.0
                )

        if st.button("Run QC Checks", type="primary"):
            with st.spinner("Running quality-control checks..."):
                ide_df = st.session_state.processed_data
                # Attach datetime_parsed if IDE computation has been run
                if st.session_state.independence_data is not None:
                    ide_df = st.session_state.independence_data

                stations = (
                    ide_df["station_id"].unique().tolist()
                    if "station_id" in ide_df.columns
                    else [default_station_id]
                )
                trap_nights_map = get_trap_nights_map(stations, trap_nights_input)

                qc = QCEngine(
                    min_trap_nights=qc_min_effort,
                    min_functionality_pct=qc_min_func,
                    min_zero_detection_nights=qc_min_zero,
                    confidence_threshold=detection_confidence,
                    clock_drift_minutes=qc_drift,
                    survey_start=qc_survey_start if qc_survey_start else None,
                    survey_end=qc_survey_end if qc_survey_end else None,
                )
                flags = qc.run_all_checks(
                    ide_df,
                    trap_nights_map,
                    ide_summary=st.session_state.ide_summary,
                )
                st.session_state.qc_flags = flags

        if st.session_state.qc_flags is not None:
            flags = st.session_state.qc_flags

            if flags.empty:
                st.success("No QC issues found.")
            else:
                # --- Summary metrics ---
                sev_counts = flags["severity"].value_counts() if not flags.empty else pd.Series()
                qm1, qm2, qm3, qm4 = st.columns(4)
                qm1.metric("Total Flags", len(flags))
                qm2.metric("Errors", int(sev_counts.get("Error", 0)))
                qm3.metric("Warnings", int(sev_counts.get("Warning", 0)))
                qm4.metric("Info", int(sev_counts.get("Info", 0)))

                st.divider()

                # --- Filter controls ---
                fc1, fc2 = st.columns(2)
                with fc1:
                    sev_filter = st.multiselect(
                        "Filter by severity",
                        options=["Error", "Warning", "Info"],
                        default=["Error", "Warning", "Info"],
                    )
                with fc2:
                    code_filter = st.multiselect(
                        "Filter by flag code",
                        options=sorted(flags["flag_code"].unique().tolist()),
                        default=[],
                    )

                filtered_flags = flags[flags["severity"].isin(sev_filter)]
                if code_filter:
                    filtered_flags = filtered_flags[filtered_flags["flag_code"].isin(code_filter)]

                # --- Colour-coded table ---
                def _severity_icon(sev):
                    return {"Error": "🔴", "Warning": "🟡", "Info": "🔵"}.get(sev, "⚪")

                display_flags = filtered_flags.copy()
                display_flags["severity"] = display_flags["severity"].apply(
                    lambda s: f"{_severity_icon(s)} {s}"
                )

                st.dataframe(
                    display_flags,
                    column_config={
                        "flag_code": st.column_config.TextColumn("Flag", width="medium"),
                        "severity": "Severity",
                        "station_id": "Station",
                        "detail": st.column_config.TextColumn("Detail", width="large"),
                        "affected_value": "Affected Value",
                        "checked_at": "Checked At",
                    },
                    column_order=["severity", "flag_code", "station_id", "detail",
                                  "affected_value", "checked_at"],
                    hide_index=True,
                    width="stretch",
                )

                st.divider()

                # --- Flags by code bar chart ---
                st.subheader("Flags by Type")
                code_counts = filtered_flags["flag_code"].value_counts()
                st.bar_chart(code_counts)

                st.divider()

                # --- Export ---
                qc_csv = filtered_flags.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download QC Report (CSV)",
                    data=qc_csv,
                    file_name=f"qc_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )

with tab8:
    st.header("Stations & Deployments")
    sm = st.session_state.station_manager

    # ── Sub-tabs inside the station manager ──────────────────────────
    s_tab1, s_tab2, s_tab3 = st.tabs(["Station Registry", "Deployment History", "Summary & Map"])

    # ----------------------------------------------------------------
    # Station Registry
    # ----------------------------------------------------------------
    with s_tab1:
        st.subheader("Station Registry")

        station_df = sm.get_stations()
        if not station_df.empty:
            st.dataframe(
                station_df,
                column_config={
                    "station_id": "Station ID",
                    "gps_lat": st.column_config.NumberColumn("Latitude", format="%.6f"),
                    "gps_lon": st.column_config.NumberColumn("Longitude", format="%.6f"),
                    "habitat_stratum": "Stratum",
                    "camera_model": "Camera Model",
                    "team_member": "Team Member",
                    "notes": "Notes",
                    "created_at": "Registered",
                },
                column_order=["station_id", "habitat_stratum", "camera_model",
                              "gps_lat", "gps_lon", "team_member", "notes"],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No stations registered yet. Add one below.")

        st.divider()

        # Add station form
        with st.expander("Add New Station", expanded=station_df.empty):
            with st.form("add_station_form", clear_on_submit=True):
                fc1, fc2 = st.columns(2)
                with fc1:
                    new_sid   = st.text_input("Station ID *", placeholder="e.g. ST-01")
                    new_lat   = st.number_input("GPS Latitude", value=0.0, format="%.6f")
                    new_model = st.text_input("Camera Model", placeholder="e.g. Bushnell Core")
                with fc2:
                    new_stratum = st.selectbox("Habitat Stratum", STRATA)
                    new_lon     = st.number_input("GPS Longitude", value=0.0, format="%.6f")
                    new_member  = st.text_input("Assigned Team Member")
                new_notes = st.text_area("Notes", height=68)

                if st.form_submit_button("Register Station", type="primary"):
                    if not new_sid.strip():
                        st.error("Station ID is required.")
                    else:
                        ok = sm.add_station(
                            station_id=new_sid.strip(),
                            gps_lat=new_lat if new_lat != 0.0 else None,
                            gps_lon=new_lon if new_lon != 0.0 else None,
                            habitat_stratum=new_stratum,
                            camera_model=new_model,
                            team_member=new_member,
                            notes=new_notes,
                        )
                        if ok:
                            st.success(f"Station '{new_sid}' registered.")
                            st.rerun()
                        else:
                            st.error(f"Station ID '{new_sid}' already exists.")

        # Delete station
        if not station_df.empty:
            with st.expander("Delete Station"):
                del_sid = st.selectbox(
                    "Select station to delete",
                    station_df["station_id"].tolist(),
                    key="del_station_select",
                )
                if st.button("Delete Station", type="secondary"):
                    sm.delete_station(del_sid)
                    st.warning(f"Station '{del_sid}' and its deployments deleted.")
                    st.rerun()

    # ----------------------------------------------------------------
    # Deployment History
    # ----------------------------------------------------------------
    with s_tab2:
        st.subheader("Deployment History")

        station_df = sm.get_stations()
        dep_df = sm.get_deployments()

        if not dep_df.empty:
            # Filter by station
            station_filter = st.selectbox(
                "Filter by station",
                ["All stations"] + station_df["station_id"].tolist() if not station_df.empty
                else ["All stations"],
                key="dep_filter",
            )
            view_dep = dep_df if station_filter == "All stations" else dep_df[dep_df["station_id"] == station_filter]

            st.dataframe(
                view_dep,
                column_config={
                    "id": "Deployment ID",
                    "station_id": "Station",
                    "deployment_date": "Deployed",
                    "retrieval_date": st.column_config.TextColumn("Retrieved", help="Blank = ongoing"),
                    "team_member": "Team Member",
                    "sd_card_id": "SD Card",
                    "camera_down_days": "Down Days",
                    "notes": "Notes",
                },
                column_order=["id", "station_id", "deployment_date", "retrieval_date",
                              "team_member", "sd_card_id", "camera_down_days", "notes"],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No deployments recorded yet.")

        st.divider()

        # Add deployment form
        if not station_df.empty:
            with st.expander("Add Deployment Record", expanded=dep_df.empty):
                with st.form("add_deployment_form", clear_on_submit=True):
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        dep_station = st.selectbox("Station *", station_df["station_id"].tolist())
                        dep_date    = st.date_input("Deployment Date *", value=None)
                        dep_sd      = st.text_input("SD Card ID", placeholder="e.g. SD-003")
                    with dc2:
                        dep_team    = st.text_input("Team Member")
                        ret_date    = st.date_input("Retrieval Date (leave blank if ongoing)", value=None)
                        dep_down    = st.number_input("Camera-Down Days", min_value=0, value=0,
                                                      help="Days camera was non-functional during this deployment")
                    dep_notes = st.text_area("Notes", height=68)

                    if st.form_submit_button("Save Deployment", type="primary"):
                        if dep_date is None:
                            st.error("Deployment date is required.")
                        else:
                            row_id = sm.add_deployment(
                                station_id=dep_station,
                                deployment_date=str(dep_date),
                                retrieval_date=str(ret_date) if ret_date else None,
                                team_member=dep_team,
                                sd_card_id=dep_sd,
                                camera_down_days=int(dep_down),
                                notes=dep_notes,
                            )
                            if row_id > 0:
                                st.success(f"Deployment saved (ID {row_id}).")
                                st.rerun()
                            else:
                                st.error("Failed to save deployment.")

        # Delete deployment
        if not dep_df.empty:
            with st.expander("Delete Deployment"):
                del_dep_id = st.selectbox(
                    "Select deployment ID to delete",
                    dep_df["id"].tolist(),
                    key="del_dep_select",
                )
                if st.button("Delete Deployment", type="secondary", key="del_dep_btn"):
                    sm.delete_deployment(int(del_dep_id))
                    st.warning(f"Deployment {del_dep_id} deleted.")
                    st.rerun()
        else:
            st.info("Register stations above before adding deployments.")

    # ----------------------------------------------------------------
    # Summary & Map
    # ----------------------------------------------------------------
    with s_tab3:
        st.subheader("Station Summary")

        summary_df = sm.get_station_summary()

        if summary_df.empty:
            st.info("No stations registered yet. Add stations and deployments in the other tabs.")
        else:
            # --- Headline metrics ---
            sm1, sm2, sm3, sm4 = st.columns(4)
            trap_nights_all = sm.compute_trap_nights()
            func_rates_all  = sm.compute_functionality_rates()

            sm1.metric("Total Stations", len(summary_df))
            sm2.metric("Total Trap Nights", sum(trap_nights_all.values()))
            low_func = sum(1 for v in func_rates_all.values() if v < 90)
            sm3.metric("Low Functionality", low_func)
            active = len(summary_df[summary_df["status"] == "Active"])
            sm4.metric("Active Stations", active)

            st.divider()

            # --- Summary table ---
            st.dataframe(
                summary_df,
                column_config={
                    "station_id": "Station ID",
                    "habitat_stratum": "Stratum",
                    "camera_model": "Camera",
                    "team_member": "Team Member",
                    "gps_lat": st.column_config.NumberColumn("Lat", format="%.4f"),
                    "gps_lon": st.column_config.NumberColumn("Lon", format="%.4f"),
                    "trap_nights": "Trap Nights",
                    "functionality_pct": st.column_config.NumberColumn(
                        "Functionality %", format="%.1f"
                    ),
                    "deployment_count": "Deployments",
                    "status": "Status",
                },
                column_order=["station_id", "habitat_stratum", "camera_model",
                              "trap_nights", "functionality_pct", "deployment_count",
                              "status", "gps_lat", "gps_lon", "team_member"],
                hide_index=True,
                width="stretch",
            )

            st.divider()

            # --- Trap nights bar chart ---
            st.subheader("Trap Nights per Station")
            tn_series = pd.Series(trap_nights_all, name="Trap Nights")
            if not tn_series.empty:
                st.bar_chart(tn_series)

            # --- Station map ---
            map_df = summary_df.dropna(subset=["gps_lat", "gps_lon"]).copy()
            map_df = map_df[(map_df["gps_lat"] != 0) | (map_df["gps_lon"] != 0)]
            if not map_df.empty:
                st.divider()
                st.subheader("Station Map")
                st.map(
                    map_df.rename(columns={"gps_lat": "lat", "gps_lon": "lon"})[["lat", "lon"]],
                    zoom=10,
                )
                st.caption("Tip: Stations at (0, 0) are hidden — enter GPS coordinates in the Station Registry.")
            else:
                st.info("Add GPS coordinates in the Station Registry to see the station map.")

            st.divider()

            # --- Export ---
            csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Station Summary (CSV)",
                data=csv_bytes,
                file_name=f"station_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

with tab9:
    st.header("Review Queue")
    st.markdown(
        "All animal detections below the **Review Queue Threshold** (set in the sidebar) "
        "that have not yet been reviewed. Work through them sequentially — "
        "each action is logged with your Reviewer ID and a timestamp."
    )

    re = st.session_state.review_engine

    if st.session_state.processed_data is None:
        st.info("Process images first in the Upload & Process tab.")
    else:
        df_all = st.session_state.processed_data

        # --- Stats bar ---
        stats = re.get_stats()
        rs1, rs2, rs3, rs4, rs5 = st.columns(5)
        queue_df = re.get_queue(df_all, review_confidence_threshold)
        rs1.metric("In Queue", len(queue_df))
        rs2.metric("Reviewed (total)", stats["total"])
        rs3.metric("Accepted", stats["accepted"])
        rs4.metric("Corrected", stats["corrected"])
        rs5.metric("Rejected", stats["rejected"])

        st.divider()

        if queue_df.empty:
            st.success(
                f"Review queue is empty — all animal detections are above "
                f"{review_confidence_threshold:.0%} confidence or have already been reviewed."
            )
        else:
            # --- Batch accept ---
            with st.expander(f"Batch Accept ({len(queue_df)} items in queue)"):
                st.caption(
                    "Accept all queued detections at once. Use when confident the AI "
                    "predictions are correct for this batch."
                )
                if st.button("Batch Accept All in Queue", type="secondary"):
                    rows = queue_df.to_dict("records")
                    saved = re.batch_accept(rows, reviewer_id=reviewer_id)
                    st.success(f"Accepted {saved} detection(s).")
                    st.rerun()

            st.divider()

            # --- Sequential review ---
            st.subheader("Sequential Review")

            if "review_queue_idx" not in st.session_state:
                st.session_state.review_queue_idx = 0

            total_q = len(queue_df)
            idx = min(st.session_state.review_queue_idx, total_q - 1)

            nav1, nav2, nav3 = st.columns([1, 3, 1])
            with nav1:
                if st.button("← Previous", key="rq_prev"):
                    st.session_state.review_queue_idx = max(0, idx - 1)
                    st.rerun()
            with nav2:
                st.progress((idx + 1) / total_q, text=f"Image {idx + 1} of {total_q}")
            with nav3:
                if st.button("Next →", key="rq_next"):
                    st.session_state.review_queue_idx = min(total_q - 1, idx + 1)
                    st.rerun()

            row = queue_df.iloc[idx]
            image_path = row.get("filepath", "")
            image_id   = row.get("image_id", "")
            filename   = row.get("filename", os.path.basename(str(image_path)))
            ai_species = row.get("species_label", row.get("detected_animal", "Unknown"))
            ai_label   = row.get("primary_label", "Animal")
            confidence = float(row.get("detection_confidence", 0))

            img_col, action_col = st.columns([2, 1])

            with img_col:
                # Show scrubbed image if available
                display_path = st.session_state.scrubber.get_display_path(str(image_path)) \
                    if image_path and os.path.exists(str(image_path)) else image_path

                if display_path and os.path.exists(str(display_path)):
                    # Draw bounding boxes on the image for review context
                    raw_rows = df_all[df_all["filepath"] == image_path].to_dict("records")
                    annotated = display_image_with_info(str(display_path), raw_rows)
                    if annotated:
                        st.image(annotated, width="stretch")
                    else:
                        st.image(str(display_path), width="stretch")
                else:
                    st.warning("Image file not found.")

                st.caption(
                    f"**File:** {filename} | "
                    f"**AI prediction:** {ai_species} ({confidence:.1%} confidence) | "
                    f"**Date:** {row.get('date', 'Unknown')} {row.get('time', '')}"
                )

            with action_col:
                st.markdown("### Review Decision")
                review_notes = st.text_input("Notes (optional)", key=f"rq_notes_{idx}")

                # Accept
                if st.button("Accept", type="primary", key=f"rq_accept_{idx}"):
                    re.accept(
                        image_id=image_id, filename=filename,
                        original_species=ai_species, original_label=ai_label,
                        confidence=confidence, reviewer_id=reviewer_id, notes=review_notes,
                    )
                    st.session_state.review_queue_idx = min(total_q - 1, idx + 1)
                    st.rerun()

                st.divider()

                # Correct
                corrected_species = st.text_input(
                    "Correct species to:", value=ai_species, key=f"rq_corr_{idx}"
                )
                corrected_label = st.selectbox(
                    "Correct label:",
                    ["Animal", "Person", "Vehicle", "Empty"],
                    index=0, key=f"rq_label_{idx}"
                )
                if st.button("Save Correction", key=f"rq_save_{idx}"):
                    re.correct(
                        image_id=image_id, filename=filename,
                        original_species=ai_species, corrected_species=corrected_species,
                        original_label=ai_label, corrected_label=corrected_label,
                        confidence=confidence, reviewer_id=reviewer_id, notes=review_notes,
                    )
                    st.session_state.review_queue_idx = min(total_q - 1, idx + 1)
                    st.rerun()

                st.divider()

                # Reject
                if st.button("Reject (False Positive)", type="secondary", key=f"rq_reject_{idx}"):
                    re.reject(
                        image_id=image_id, filename=filename,
                        original_species=ai_species, original_label=ai_label,
                        confidence=confidence, reviewer_id=reviewer_id, notes=review_notes,
                    )
                    st.session_state.review_queue_idx = min(total_q - 1, idx + 1)
                    st.rerun()

        st.divider()

        # --- Correction log ---
        st.subheader("Correction Log")
        corrections = re.get_corrections_df()
        if not corrections.empty:
            st.dataframe(
                corrections,
                column_config={
                    "image_id": st.column_config.TextColumn("Image ID", width="small"),
                    "filename": "File",
                    "original_species": "AI Prediction",
                    "corrected_species": "Human Correction",
                    "original_label": "AI Label",
                    "corrected_label": "Corrected Label",
                    "confidence": st.column_config.ProgressColumn(
                        "Conf", min_value=0, max_value=1, format="%.2f"
                    ),
                    "reviewer_id": "Reviewer",
                    "reviewed_at": "Reviewed At",
                    "notes": "Notes",
                },
                column_order=["filename", "original_species", "corrected_species",
                              "original_label", "corrected_label", "confidence",
                              "reviewer_id", "reviewed_at", "notes"],
                hide_index=True,
                width="stretch",
            )
            csv_corr = corrections.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Correction Log (CSV)",
                data=csv_corr,
                file_name=f"corrections_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No corrections recorded yet.")

        # --- Scrub audit ---
        if st.session_state.scrub_audit:
            st.divider()
            st.subheader("Privacy Scrub Audit")
            audit_df = pd.DataFrame(st.session_state.scrub_audit)
            scrubbed = audit_df[~audit_df["skipped"]]
            st.caption(
                f"{len(scrubbed)} of {len(audit_df)} images scrubbed. "
                "Original files are untouched; scrubbed copies are in the `scrubbed/` subfolder."
            )
            st.dataframe(
                scrubbed[["filename", "boxes_blurred", "scrubbed_path", "scrubbed_at"]],
                hide_index=True, width="stretch"
            )
            audit_csv = audit_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Full Scrub Audit (CSV)",
                data=audit_csv,
                file_name=f"scrub_audit_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

# ── Tab 10: Community Observer ─────────────────────────────────────────────
with tab10:
    st.header("Community Observer Data Entry")
    st.markdown(
        "Record field observer sightings to cross-verify camera trap data. "
        "Species seen by observers but not yet detected by cameras are flagged "
        "as **Community-only records**."
    )

    co = st.session_state.community_observer

    co_stats = co.get_stats()
    cs1, cs2, cs3 = st.columns(3)
    cs1.metric("Total Observations", co_stats["total"])
    cs2.metric("Unique Species", co_stats["unique_species"])
    cs3.metric("Total Individuals", co_stats["total_individuals"])

    st.divider()

    # ── Entry form ──────────────────────────────────────────────────────────
    with st.expander("Add Observation", expanded=co_stats["total"] == 0):
        with st.form("community_obs_form", clear_on_submit=True):
            fc1, fc2 = st.columns(2)
            with fc1:
                co_species  = st.text_input("Species *", placeholder="e.g. Nile Lechwe")
                co_date     = st.date_input("Date *", value=None)
                co_time     = st.text_input("Time (HH:MM)", placeholder="e.g. 06:30")
                co_count    = st.number_input("Individual Count", min_value=1, value=1)
            with fc2:
                co_observer = st.text_input("Observer Name")
                co_obs_type = st.selectbox("Observation Type", OBSERVATION_TYPES)
                co_station  = st.text_input("Nearest Station ID", placeholder="e.g. ST-01")
                co_lat      = st.number_input("GPS Latitude (optional)", value=0.0, format="%.6f")
                co_lon      = st.number_input("GPS Longitude (optional)", value=0.0, format="%.6f")
            co_notes = st.text_area("Notes", height=68)

            if st.form_submit_button("Save Observation", type="primary"):
                if not co_species.strip():
                    st.error("Species name is required.")
                elif co_date is None:
                    st.error("Date is required.")
                else:
                    # Synonym resolution
                    sl = st.session_state.species_library
                    resolved = sl.resolve_synonym(co_species.strip())
                    final_species = resolved if resolved else co_species.strip()
                    if resolved:
                        st.info(f"Species name resolved: '{co_species}' → '{resolved}'")

                    row_id = co.add_observation(
                        species=final_species,
                        obs_date=str(co_date),
                        obs_time=co_time or None,
                        station_id=co_station or None,
                        gps_lat=co_lat if co_lat != 0.0 else None,
                        gps_lon=co_lon if co_lon != 0.0 else None,
                        individual_count=int(co_count),
                        observer_name=co_observer,
                        observation_type=co_obs_type,
                        notes=co_notes,
                    )
                    st.success(f"Observation saved (ID {row_id}).")
                    st.rerun()

    st.divider()

    # ── Observations table ──────────────────────────────────────────────────
    st.subheader("Recorded Observations")
    obs_df = co.get_observations()
    if not obs_df.empty:
        station_filter_co = st.selectbox(
            "Filter by station", ["All"] + sorted(obs_df["station_id"].dropna().unique().tolist()),
            key="co_station_filter"
        )
        view_obs = obs_df if station_filter_co == "All" else obs_df[obs_df["station_id"] == station_filter_co]

        st.dataframe(
            view_obs,
            column_config={
                "id": "ID", "obs_date": "Date", "obs_time": "Time",
                "station_id": "Station", "species": "Species",
                "individual_count": "Count", "observer_name": "Observer",
                "observation_type": "Type", "notes": "Notes",
                "gps_lat": st.column_config.NumberColumn("Lat", format="%.4f"),
                "gps_lon": st.column_config.NumberColumn("Lon", format="%.4f"),
            },
            column_order=["id", "obs_date", "obs_time", "station_id", "species",
                          "individual_count", "observer_name", "observation_type", "notes"],
            hide_index=True, width="stretch",
        )

        with st.expander("Delete Observation"):
            del_obs_id = st.selectbox("Select ID to delete", obs_df["id"].tolist(), key="del_obs")
            if st.button("Delete", type="secondary", key="del_obs_btn"):
                co.delete_observation(int(del_obs_id))
                st.rerun()

        st.download_button(
            "Download Observations (CSV)",
            data=obs_df.to_csv(index=False).encode("utf-8"),
            file_name=f"community_observations_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No observations recorded yet.")

    # ── Cross-verification ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Cross-Verification with Camera Detections")
    if st.session_state.processed_data is not None and not obs_df.empty:
        camera_sp = set(
            st.session_state.processed_data["species_label"].dropna().unique().tolist()
            if "species_label" in st.session_state.processed_data.columns else []
        )
        comparison = co.get_species_comparison(camera_sp)

        xcol1, xcol2, xcol3 = st.columns(3)
        with xcol1:
            st.markdown("**Camera only**")
            for s in comparison["camera_only"] or ["—"]:
                st.caption(s)
        with xcol2:
            st.markdown("**Both camera + observer**")
            for s in comparison["both"] or ["—"]:
                st.caption(s)
        with xcol3:
            st.markdown("**Community-only records**")
            for s in comparison["community_only"] or ["—"]:
                st.caption(f"⚑ {s}")
    else:
        st.info("Process images and add observations to see the cross-verification comparison.")


# ── Tab 11: Spatial & Map ───────────────────────────────────────────────────
with tab11:
    st.header("Spatial Outputs & Map Viewer")
    st.markdown(
        "Interactive map of stations and detection heat map. "
        "Export station locations and detection events as GeoJSON."
    )

    se = st.session_state.spatial_exporter
    sm_data = st.session_state.station_manager.get_station_summary()

    # ── Map ─────────────────────────────────────────────────────────────────
    st.subheader("Interactive Map")
    if sm_data.empty:
        st.info("Register stations with GPS coordinates in the Stations & Deployments tab to see them on the map.")
    else:
        try:
            import pydeck as pdk

            station_layer_data = se.stations_layer_data(sm_data)

            layers = []
            if station_layer_data:
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=station_layer_data,
                    get_position=["lon", "lat"],
                    get_color="colour",
                    get_radius="radius",
                    pickable=True,
                    auto_highlight=True,
                ))

            if st.session_state.ide_summary is not None:
                heat_data = se.detections_layer_data(st.session_state.ide_summary, sm_data)
                if heat_data:
                    layers.append(pdk.Layer(
                        "HeatmapLayer",
                        data=heat_data,
                        get_position=["lon", "lat"],
                        get_weight="weight",
                        radiusPixels=60,
                        opacity=0.6,
                    ))

            if layers:
                center_lat = sm_data["gps_lat"].dropna().mean()
                center_lon = sm_data["gps_lon"].dropna().mean()
                view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=9, pitch=0)
                st.pydeck_chart(pdk.Deck(
                    layers=layers,
                    initial_view_state=view,
                    tooltip={"text": "{station_id}\n{stratum}\nTrap nights: {trap_nights}"},
                    map_style="mapbox://styles/mapbox/satellite-streets-v11",
                ))
                # Legend
                st.caption(
                    "**Map legend:** Blue = Wetland Habitat · Green = Watering Point · "
                    "Orange = Corridor · Grey = Other · Heat map = detection density (IDEs)"
                )
            else:
                st.info("Add GPS coordinates to stations to display them on the map.")
        except ImportError:
            # Fallback to st.map if pydeck not available
            map_df = sm_data.dropna(subset=["gps_lat", "gps_lon"])
            map_df = map_df[(map_df["gps_lat"] != 0) | (map_df["gps_lon"] != 0)]
            if not map_df.empty:
                st.map(map_df.rename(columns={"gps_lat": "lat", "gps_lon": "lon"})[["lat", "lon"]])

    st.divider()

    # ── GeoJSON exports ──────────────────────────────────────────────────────
    st.subheader("GeoJSON Export")
    geo_col1, geo_col2 = st.columns(2)

    with geo_col1:
        st.markdown("**Station Locations**")
        if not sm_data.empty:
            stations_geojson = se.stations_to_geojson(sm_data)
            st.download_button(
                "Download Stations GeoJSON",
                data=stations_geojson.encode("utf-8"),
                file_name=f"stations_{datetime.now().strftime('%Y%m%d')}.geojson",
                mime="application/geo+json",
            )
            with st.expander("Preview GeoJSON"):
                st.code(stations_geojson[:1200] + ("\n..." if len(stations_geojson) > 1200 else ""), language="json")
        else:
            st.info("Register stations first.")

    with geo_col2:
        st.markdown("**Detection Events**")
        if st.session_state.processed_data is not None:
            det_geojson = se.detections_to_geojson(
                st.session_state.processed_data,
                stations_df=sm_data if not sm_data.empty else None,
                ide_summary=st.session_state.ide_summary,
            )
            st.download_button(
                "Download Detections GeoJSON",
                data=det_geojson.encode("utf-8"),
                file_name=f"detections_{datetime.now().strftime('%Y%m%d')}.geojson",
                mime="application/geo+json",
            )
        else:
            st.info("Process images first.")

    st.divider()

    # ── Georeferenced CSV ────────────────────────────────────────────────────
    st.subheader("Georeferenced CSV")
    st.caption("Detection records with station lat/lon columns appended — ready for GIS import.")
    if st.session_state.processed_data is not None:
        geo_csv = se.detections_to_csv(
            st.session_state.processed_data,
            stations_df=sm_data if not sm_data.empty else None,
        )
        st.download_button(
            "Download Georeferenced CSV",
            data=geo_csv.encode("utf-8"),
            file_name=f"detections_geo_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("Process images first.")


# ── Tab 12: Species Library ─────────────────────────────────────────────────
with tab12:
    st.header("Species Reference Library")
    st.markdown(
        f"Pre-loaded with **{len(st.session_state.species_library.get_names_list())} species** "
        "known or likely to occur in the Gambella wetland landscape. "
        "Search, browse, and add new entries."
    )

    sl = st.session_state.species_library

    # ── Search ───────────────────────────────────────────────────────────────
    search_query = st.text_input(
        "Search by common name, scientific name, family, or notes",
        placeholder="e.g. Nile Lechwe, Bovidae, Vulnerable…"
    )
    lib_df = sl.search(search_query)

    # ── IUCN filter ──────────────────────────────────────────────────────────
    iucn_options = ["All"] + sorted(lib_df["iucn_status"].dropna().unique().tolist()) if not lib_df.empty else ["All"]
    iucn_filter = st.selectbox("Filter by IUCN status", iucn_options, key="lib_iucn")
    if iucn_filter != "All":
        lib_df = lib_df[lib_df["iucn_status"] == iucn_filter]

    st.caption(f"Showing {len(lib_df)} species")

    st.dataframe(
        lib_df,
        column_config={
            "common_name":     st.column_config.TextColumn("Common Name", width="medium"),
            "scientific_name": st.column_config.TextColumn("Scientific Name", width="medium"),
            "family":          "Family",
            "order_name":      "Order",
            "iucn_status":     "IUCN Status",
            "kenya_status":    "Kenya Status",
            "notes":           st.column_config.TextColumn("Ecological Notes", width="large"),
            "added_by":        "Added By",
        },
        column_order=["common_name", "scientific_name", "family", "order_name",
                      "iucn_status", "kenya_status", "notes"],
        hide_index=True,
        width="stretch",
    )

    st.divider()

    # ── Synonym resolver ─────────────────────────────────────────────────────
    st.subheader("Synonym Resolver")
    syn_input = st.text_input("Enter a name to resolve", placeholder="e.g. painted dog, hippo")
    if syn_input:
        resolved = sl.resolve_synonym(syn_input)
        if resolved:
            entry = sl.get_by_name(resolved)
            st.success(f"'{syn_input}' → **{resolved}**")
            if entry:
                st.json({k: v for k, v in entry.items()
                         if k not in ("id", "created_at", "added_by")})
        else:
            direct = sl.get_by_name(syn_input)
            if direct:
                st.info(f"'{syn_input}' is already the accepted name.")
                st.json({k: v for k, v in direct.items()
                         if k not in ("id", "created_at", "added_by")})
            else:
                st.warning(f"'{syn_input}' not found in the library.")

    st.divider()

    # ── Add new species (Data Manager) ───────────────────────────────────────
    with st.expander("Add New Species (Data Manager)"):
        with st.form("add_species_form", clear_on_submit=True):
            ac1, ac2 = st.columns(2)
            with ac1:
                new_common = st.text_input("Common Name *")
                new_sci    = st.text_input("Scientific Name")
                new_family = st.text_input("Family")
                new_order  = st.text_input("Order")
            with ac2:
                new_iucn   = st.selectbox("IUCN Status",
                    ["Least Concern", "Near Threatened", "Vulnerable",
                     "Endangered", "Critically Endangered", "Data Deficient", "Not Evaluated"])
                new_kenya  = st.text_input("Kenya Conservation Status")
                new_added  = st.text_input("Added By", value="Data Manager")
            new_notes_sp = st.text_area("Ecological Notes", height=80)

            if st.form_submit_button("Add Species", type="primary"):
                if not new_common.strip():
                    st.error("Common name is required.")
                else:
                    ok = sl.add_species(
                        common_name=new_common, scientific_name=new_sci,
                        family=new_family, order_name=new_order,
                        iucn_status=new_iucn, kenya_status=new_kenya,
                        notes=new_notes_sp, added_by=new_added,
                    )
                    if ok:
                        st.success(f"'{new_common}' added to the library.")
                        st.rerun()
                    else:
                        st.error(f"'{new_common}' already exists.")

    # ── Add synonym ──────────────────────────────────────────────────────────
    with st.expander("Add Synonym"):
        with st.form("add_synonym_form", clear_on_submit=True):
            syn_new   = st.text_input("Synonym / Alternative name")
            syn_target = st.selectbox("Resolves to", sl.get_names_list())
            if st.form_submit_button("Add Synonym"):
                if syn_new.strip():
                    sl.add_synonym(syn_new.strip(), syn_target)
                    st.success(f"'{syn_new}' → '{syn_target}' saved.")
                    st.rerun()

    st.divider()

    # ── Quick lookup from Review Queue ───────────────────────────────────────
    st.subheader("Quick Species Lookup")
    st.caption("Look up any species that appears in your detection data.")
    if st.session_state.processed_data is not None and "species_label" in st.session_state.processed_data.columns:
        detected_names = sorted(st.session_state.processed_data["species_label"].dropna().unique().tolist())
        lookup_name = st.selectbox("Select detected species", ["—"] + detected_names, key="lib_lookup")
        if lookup_name and lookup_name != "—":
            import re as _re
            clean = _re.sub(r"\s+\d+(\.\d+)?", "", lookup_name).strip()
            entry = sl.get_by_name(clean)
            if entry:
                st.json({k: v for k, v in entry.items()
                         if k not in ("id", "created_at")})
            else:
                resolved = sl.resolve_synonym(clean)
                if resolved:
                    entry = sl.get_by_name(resolved)
                    st.info(f"Resolved '{clean}' → '{resolved}'")
                    if entry:
                        st.json({k: v for k, v in entry.items()
                                 if k not in ("id", "created_at")})
                else:
                    st.warning(f"'{clean}' not found in the species library.")
    else:
        st.info("Process images to enable species lookup from detection data.")

    st.download_button(
        "Download Full Species Library (CSV)",
        data=sl.get_all().to_csv(index=False).encode("utf-8"),
        file_name=f"species_library_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# ── Tab 13: Corridor Analysis ────────────────────────────────────────────────
with tab13:
    st.header("Corridor Movement Analysis")
    st.caption(
        "Detects directional wildlife movement between paired camera stations. "
        "Requires GPS coordinates on stations and IDEs computed in Ecological Analytics."
    )

    ca: CorridorAnalyzer = st.session_state.corridor_analyzer
    sm = st.session_state.station_manager
    stations_df_c = sm.get_station_summary()

    # ── Settings ─────────────────────────────────────────────────────────────
    with st.expander("Corridor Settings", expanded=False):
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            max_dist = st.number_input(
                "Max inter-station distance (m)", min_value=100, max_value=10000,
                value=1000, step=100,
                help="Only station pairs within this distance are considered a corridor."
            )
        with c_col2:
            move_window = st.number_input(
                "Movement window (hours)", min_value=1, max_value=72, value=24,
                help="Max hours between detections at two stations to count as a directed movement."
            )
        with c_col3:
            bn_pct = st.number_input(
                "Bottleneck percentile", min_value=5, max_value=50, value=25,
                help="Pairs below this passage-frequency percentile are flagged as bottlenecks."
            )
        if st.button("Apply Settings"):
            st.session_state.corridor_analyzer = CorridorAnalyzer(
                max_distance_m=float(max_dist),
                movement_window_hours=float(move_window),
                bottleneck_percentile=float(bn_pct),
            )
            ca = st.session_state.corridor_analyzer
            st.success("Settings applied.")

    # ── Station pairs ─────────────────────────────────────────────────────────
    st.subheader("Station Pairs within Corridor Distance")
    if stations_df_c is None or stations_df_c.empty:
        st.info("Register stations with GPS coordinates in the Stations & Deployments tab.")
    else:
        pairs_df = ca.compute_station_pairs(stations_df_c)
        if pairs_df.empty:
            st.warning(
                f"No station pairs found within {max_dist} m. "
                "Check that stations have valid GPS coordinates."
            )
        else:
            st.dataframe(pairs_df, use_container_width=True)

            # ── Corridor flows ─────────────────────────────────────────────
            st.subheader("Directional Movement Events")
            ide_sum_c = st.session_state.get("ide_summary")
            if ide_sum_c is None or ide_sum_c.empty:
                st.info(
                    "Run 'Compute IDEs' in the Ecological Analytics tab first to enable flow analysis."
                )
            else:
                flows = ca.compute_corridor_flows(ide_sum_c, pairs_df)

                if flows.empty:
                    st.info(
                        "No corridor movements detected. "
                        "Try increasing the movement window or check that station IDs "
                        "in your detection data match the station registry."
                    )
                else:
                    trap_map = sm.compute_trap_nights()
                    flows_rate = ca.compute_passage_frequency(flows, trap_map)

                    display_cols = [
                        "station_a", "station_b", "distance_m", "species",
                        "flow_a_to_b", "flow_b_to_a", "total_passages",
                        "dominant_direction", "asymmetry_ratio",
                        "passage_rate_per_100tn",
                    ]
                    show_cols = [c for c in display_cols if c in flows_rate.columns]
                    st.dataframe(flows_rate[show_cols], use_container_width=True)

                    # ── Charts ────────────────────────────────────────────
                    ch_col1, ch_col2 = st.columns(2)

                    with ch_col1:
                        st.subheader("Passages by Species")
                        sp_totals = (
                            flows_rate.groupby("species")["total_passages"]
                            .sum()
                            .sort_values(ascending=False)
                            .reset_index()
                        )
                        st.bar_chart(sp_totals.set_index("species")["total_passages"])

                    with ch_col2:
                        st.subheader("Passages by Corridor Pair")
                        pair_label = flows_rate["station_a"] + " ↔ " + flows_rate["station_b"]
                        pair_totals = (
                            flows_rate.assign(pair=pair_label)
                            .groupby("pair")["total_passages"]
                            .sum()
                            .sort_values(ascending=False)
                            .reset_index()
                        )
                        st.bar_chart(pair_totals.set_index("pair")["total_passages"])

                    # ── Bottlenecks ───────────────────────────────────────
                    st.subheader("Bottleneck Pairs")
                    bottlenecks = ca.identify_bottlenecks(flows_rate)
                    if bottlenecks.empty:
                        st.success("No bottlenecks identified.")
                    else:
                        st.warning(
                            f"{len(bottlenecks)} corridor pair(s) flagged as bottlenecks "
                            f"(passage frequency below {bn_pct}th percentile)."
                        )
                        st.dataframe(bottlenecks, use_container_width=True)

                    # ── Corridor utilisation ──────────────────────────────
                    st.subheader("Corridor Utilisation by Species")
                    util_df = ca.compute_corridor_utilisation(flows_rate, pairs_df)
                    if not util_df.empty:
                        st.dataframe(util_df, use_container_width=True)

                    # ── Export ────────────────────────────────────────────
                    st.divider()
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button(
                            "Download Flow Table (CSV)",
                            data=flows_rate.to_csv(index=False).encode("utf-8"),
                            file_name=f"corridor_flows_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )
                    with dl_col2:
                        if not bottlenecks.empty:
                            st.download_button(
                                "Download Bottlenecks (CSV)",
                                data=bottlenecks.to_csv(index=False).encode("utf-8"),
                                file_name=f"bottlenecks_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                            )


# ── Tab 14: Project Configuration ────────────────────────────────────────────
with tab14:
    st.header("Project Configuration")
    st.caption(
        "Manage survey projects, configure indicator thresholds, "
        "lock reference baselines, and export project data."
    )

    pc: ProjectConfig = st.session_state.project_config

    p_tab1, p_tab2, p_tab3, p_tab4 = st.tabs(
        ["Projects", "Indicator Thresholds", "Baseline", "Export"]
    )

    # ── Sub-tab 1: Projects ────────────────────────────────────────────────
    with p_tab1:
        st.subheader("Survey Projects")

        projects_df = pc.get_projects()
        active_proj = pc.get_active_project()

        if not projects_df.empty:
            st.dataframe(
                projects_df[["id", "name", "survey_area", "lead_org",
                             "start_date", "end_date", "is_active", "created_at"]],
                use_container_width=True,
            )
            if active_proj:
                st.success(f"Active project: **{active_proj['name']}**")
            else:
                st.warning("No active project selected.")

            # Set active
            proj_names = projects_df["name"].tolist()
            proj_ids   = projects_df["id"].tolist()
            sel_name = st.selectbox("Set active project", proj_names, key="set_active_proj")
            if st.button("Set as Active"):
                sel_id = proj_ids[proj_names.index(sel_name)]
                pc.set_active_project(sel_id)
                st.success(f"'{sel_name}' is now the active project.")
                st.rerun()

            # Delete
            del_name = st.selectbox("Delete project", ["— select —"] + proj_names, key="del_proj")
            if del_name != "— select —" and st.button("Delete Project", type="secondary"):
                del_id = proj_ids[proj_names.index(del_name)]
                pc.delete_project(del_id)
                st.warning(f"'{del_name}' deleted.")
                st.rerun()
        else:
            st.info("No projects yet. Create one below.")

        st.divider()
        st.subheader("Create New Project")
        with st.form("new_project_form", clear_on_submit=True):
            np_col1, np_col2 = st.columns(2)
            with np_col1:
                np_name   = st.text_input("Project name *")
                np_area   = st.text_input("Survey area")
                np_org    = st.text_input("Lead organisation")
            with np_col2:
                np_start  = st.date_input("Start date", value=None)
                np_end    = st.date_input("End date",   value=None)
            np_notes  = st.text_area("Notes", height=80)
            if st.form_submit_button("Create Project", type="primary"):
                if not np_name.strip():
                    st.error("Project name is required.")
                else:
                    pid = pc.create_project(
                        name=np_name.strip(),
                        survey_area=np_area,
                        lead_org=np_org,
                        start_date=str(np_start) if np_start else "",
                        end_date=str(np_end) if np_end else "",
                        notes=np_notes,
                    )
                    if pid:
                        st.success(f"Project '{np_name}' created (ID {pid}).")
                        st.rerun()
                    else:
                        st.error(f"A project named '{np_name}' already exists.")

    # ── Sub-tab 2: Indicator Thresholds ───────────────────────────────────
    with p_tab2:
        st.subheader("Indicator Thresholds")
        active_proj2 = pc.get_active_project()
        if not active_proj2:
            st.warning("Select an active project in the Projects tab first.")
        else:
            pid2 = active_proj2["id"]
            current = pc.get_thresholds(pid2)
            descs   = pc.get_threshold_descriptions()

            st.caption(
                f"Configuring thresholds for project: **{active_proj2['name']}**. "
                "These values drive QC flags, alerts, and ecological analytics."
            )

            with st.form("thresholds_form"):
                new_vals = {}
                t_col1, t_col2 = st.columns(2)
                for i, (key, (label, unit, lo, hi)) in enumerate(descs.items()):
                    col = t_col1 if i % 2 == 0 else t_col2
                    with col:
                        cur_val = current.get(key, DEFAULT_THRESHOLDS.get(key, 0))
                        if isinstance(lo, float) or isinstance(hi, float):
                            new_vals[key] = st.slider(
                                f"{label} ({unit})",
                                min_value=float(lo), max_value=float(hi),
                                value=float(cur_val), step=0.05,
                            )
                        else:
                            new_vals[key] = st.slider(
                                f"{label} ({unit})",
                                min_value=int(lo), max_value=int(hi),
                                value=int(cur_val),
                            )
                if st.form_submit_button("Save Thresholds", type="primary"):
                    pc.save_thresholds(pid2, new_vals)
                    st.success("Thresholds saved.")
                    st.rerun()

            st.divider()
            st.caption("Current saved thresholds:")
            st.json(current)

    # ── Sub-tab 3: Baseline ────────────────────────────────────────────────
    with p_tab3:
        st.subheader("Baseline Locking")
        active_proj3 = pc.get_active_project()
        if not active_proj3:
            st.warning("Select an active project in the Projects tab first.")
        else:
            pid3 = active_proj3["id"]
            existing_baseline = pc.get_baseline(pid3)

            if existing_baseline:
                st.success(
                    f"Baseline locked on **{existing_baseline.get('locked_at', 'unknown')}**"
                )
                st.json({k: v for k, v in existing_baseline.items() if k != "locked_at"})
                if st.button("Clear Baseline", type="secondary"):
                    pc.clear_baseline(pid3)
                    st.warning("Baseline cleared.")
                    st.rerun()
            else:
                st.info("No baseline locked for this project.")

            st.divider()
            st.subheader("Lock Current Metrics as Baseline")
            st.caption(
                "This snapshot will serve as the reference for all future comparisons. "
                "Locking overwrites any previous baseline."
            )

            ide_sum_b = st.session_state.get("ide_summary")
            proc_data_b = st.session_state.get("processed_data")

            baseline_data = {}
            if ide_sum_b is not None and not ide_sum_b.empty:
                baseline_data["species_richness"] = int(ide_sum_b["species"].nunique())
                baseline_data["total_ides"]       = int(len(ide_sum_b))
                baseline_data["stations_active"]  = int(ide_sum_b["station_id"].nunique())
                baseline_data["species_list"]     = sorted(
                    ide_sum_b["species"].dropna().unique().tolist()
                )
            if proc_data_b is not None and not proc_data_b.empty:
                baseline_data["total_detections"] = int(len(proc_data_b))

            if baseline_data:
                st.write("Preview of metrics to lock:")
                st.json(baseline_data)
                if st.button("Lock as Baseline", type="primary"):
                    pc.lock_baseline(pid3, baseline_data)
                    st.success("Baseline locked.")
                    st.rerun()
            else:
                st.warning(
                    "No metrics available to lock. "
                    "Process images and compute IDEs in the Ecological Analytics tab first."
                )

    # ── Sub-tab 4: Export ──────────────────────────────────────────────────
    with p_tab4:
        st.subheader("Export Project")
        active_proj4 = pc.get_active_project()
        if not active_proj4:
            st.warning("Select an active project in the Projects tab first.")
        else:
            pid4 = active_proj4["id"]
            export_json = pc.export_project(pid4)
            st.json(json.loads(export_json))
            st.download_button(
                "Download Project JSON",
                data=export_json.encode("utf-8"),
                file_name=f"project_{active_proj4['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            )


# ── Tab 15: ArcGIS Sync ──────────────────────────────────────────────────────
with tab15:
    st.header("ArcGIS / Spatial Exports & Sync")
    st.caption(
        "Download spatial files for desktop GIS (Shapefile, GeoJSON, KML) "
        "or push directly to an ArcGIS Online / Enterprise feature layer."
    )

    ag: ArcGISSync = st.session_state.arcgis_sync
    se_ag: SpatialExporter = st.session_state.spatial_exporter
    sm_ag = st.session_state.station_manager
    stations_ag = sm_ag.get_station_summary()
    ide_ag = st.session_state.get("ide_summary")

    # ── Offline exports (no ArcGIS account needed) ────────────────────────────
    st.subheader("Offline Exports")
    st.caption("Download files directly — no ArcGIS account required.")

    ex_col1, ex_col2 = st.columns(2)

    with ex_col1:
        st.markdown("**Station Locations**")
        has_stations = stations_ag is not None and not stations_ag.empty

        if has_stations:
            # GeoJSON
            st.download_button(
                "GeoJSON (.geojson)",
                data=se_ag.stations_to_geojson(stations_ag).encode("utf-8"),
                file_name=f"stations_{datetime.now().strftime('%Y%m%d')}.geojson",
                mime="application/geo+json",
                key="dl_stations_geojson",
            )
            # Shapefile
            try:
                shp_bytes = se_ag.stations_to_shapefile(stations_ag)
                st.download_button(
                    "Shapefile (.zip)",
                    data=shp_bytes,
                    file_name=f"stations_{datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip",
                    key="dl_stations_shp",
                )
            except ImportError:
                st.warning("Shapefile export requires `pyshp`. Run: `pip install pyshp`")
            # KML
            st.download_button(
                "KML (.kml)",
                data=se_ag.stations_to_kml(stations_ag).encode("utf-8"),
                file_name=f"stations_{datetime.now().strftime('%Y%m%d')}.kml",
                mime="application/vnd.google-earth.kml+xml",
                key="dl_stations_kml",
            )
        else:
            st.info("Register stations with GPS coordinates in the Stations & Deployments tab.")

    with ex_col2:
        st.markdown("**Detection Events (IDEs)**")
        has_ides = ide_ag is not None and not ide_ag.empty

        if has_ides and has_stations:
            # GeoJSON
            st.download_button(
                "GeoJSON (.geojson)",
                data=se_ag.detections_to_geojson(
                    pd.DataFrame(), stations_ag, ide_ag
                ).encode("utf-8"),
                file_name=f"detections_{datetime.now().strftime('%Y%m%d')}.geojson",
                mime="application/geo+json",
                key="dl_det_geojson",
            )
            # Shapefile
            try:
                det_shp = se_ag.detections_to_shapefile(ide_ag, stations_ag)
                st.download_button(
                    "Shapefile (.zip)",
                    data=det_shp,
                    file_name=f"detections_{datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip",
                    key="dl_det_shp",
                )
            except ImportError:
                st.warning("Shapefile export requires `pyshp`. Run: `pip install pyshp`")
            # KML
            st.download_button(
                "KML (.kml)",
                data=se_ag.detections_to_kml(ide_ag, stations_ag).encode("utf-8"),
                file_name=f"detections_{datetime.now().strftime('%Y%m%d')}.kml",
                mime="application/vnd.google-earth.kml+xml",
                key="dl_det_kml",
            )
        elif not has_ides:
            st.info("Compute IDEs in the Ecological Analytics tab to enable detection exports.")
        else:
            st.info("Register stations with GPS coordinates to enable detection exports.")

    st.divider()

    # ── Live ArcGIS sync (requires account) ───────────────────────────────────
    st.subheader("Live Push to ArcGIS")

    with st.expander("Connection Settings", expanded=not ag.is_authenticated):
        ag_col1, ag_col2 = st.columns(2)
        with ag_col1:
            portal_url = st.text_input(
                "Portal URL",
                value="https://www.arcgis.com",
                help="ArcGIS Online: https://www.arcgis.com  |  Enterprise: https://your-server/portal",
            )
        with ag_col2:
            auth_method = st.radio("Authentication", ["Username / Password", "API Token"], horizontal=True)

        if auth_method == "Username / Password":
            u_col1, u_col2 = st.columns(2)
            with u_col1:
                ag_user = st.text_input("Username")
            with u_col2:
                ag_pass = st.text_input("Password", type="password")
            if st.button("Connect", type="primary"):
                try:
                    ag_new = ArcGISSync(portal_url=portal_url)
                    ok = ag_new.authenticate(ag_user, ag_pass)
                    if ok:
                        st.session_state.arcgis_sync = ag_new
                        ag = ag_new
                        st.success("Connected to ArcGIS Portal.")
                        st.rerun()
                    else:
                        st.error("Authentication failed. Check credentials.")
                except Exception as e:
                    st.error(f"Connection error: {e}")
        else:
            api_token = st.text_input("API Token / Access Token", type="password")
            if st.button("Connect with Token", type="primary"):
                ag_new = ArcGISSync(portal_url=portal_url)
                ag_new.authenticate_with_token(api_token)
                st.session_state.arcgis_sync = ag_new
                ag = ag_new
                st.success("Token applied.")
                st.rerun()

    if not ag.is_authenticated:
        st.info("Connect to ArcGIS Portal above to push data directly to a hosted feature layer.")
    else:
        st.success("Connected to ArcGIS Portal.")

        push_col1, push_col2 = st.columns(2)

        with push_col1:
            st.markdown("**Push Stations**")
            stations_layer_url = st.text_input(
                "Station Feature Layer URL",
                placeholder="https://services.arcgis.com/…/FeatureServer/0",
                key="ag_stations_url",
            )
            if has_stations:
                st.caption(f"{len(stations_ag)} stations ready.")
                if st.button("Push Stations", type="primary", key="push_stations_btn"):
                    if not stations_layer_url.strip():
                        st.error("Enter a feature layer URL.")
                    else:
                        with st.spinner("Pushing stations..."):
                            result = ag.push_stations(stations_ag, stations_layer_url.strip())
                        if result["errors"]:
                            st.warning(f"Pushed {result['added']} — errors: {result['errors'][:2]}")
                        else:
                            st.success(f"Pushed {result['added']} station(s).")
            else:
                st.info("No stations registered.")

        with push_col2:
            st.markdown("**Push Detection Events**")
            detections_layer_url = st.text_input(
                "Detections Feature Layer URL",
                placeholder="https://services.arcgis.com/…/FeatureServer/1",
                key="ag_detections_url",
            )
            if has_ides:
                st.caption(f"{len(ide_ag)} IDEs ready.")
                if st.button("Push Detections", type="primary", key="push_det_btn"):
                    if not detections_layer_url.strip():
                        st.error("Enter a feature layer URL.")
                    else:
                        with st.spinner("Pushing detection events..."):
                            result = ag.push_detections(ide_ag, stations_ag, detections_layer_url.strip())
                        if result["errors"]:
                            st.warning(f"Pushed {result['added']} — errors: {result['errors'][:2]}")
                        else:
                            st.success(f"Pushed {result['added']} detection event(s).")
            else:
                st.info("Compute IDEs first.")

        st.divider()

        # ── Sync log ──────────────────────────────────────────────────────
        st.subheader("Sync History")
        sync_log = ag.get_sync_log()
        if sync_log.empty:
            st.info("No sync history yet.")
        else:
            st.dataframe(sync_log, use_container_width=True)
            st.download_button(
                "Download Sync Log (CSV)",
                data=sync_log.to_csv(index=False).encode("utf-8"),
                file_name=f"arcgis_sync_log_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )


# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>Wildlife Camera Trap Auto-Analyzer v1.0 | Built with Streamlit</p>
        <p>Tip: The animal detection model can be swapped for specialized models like MegaDetector</p>
    </div>
""", unsafe_allow_html=True)
