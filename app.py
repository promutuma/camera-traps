"""
Wildlife Camera Trap Auto-Analyzer
A Streamlit application for processing camera trap images.
"""

import os
import sys

# Fix Windows console encoding issues
# Set environment variable and reconfigure streams
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Reconfigure stdout/stderr encoding without wrapping
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

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
from core.db_manager import DatabaseManager
from core.animal_detector import AnimalDetector, MegaDetectorWrapper
from core.bioclip_classifier import BioClipClassifier
from core.ocr_processor import OCRProcessor
from core.day_night_classifier import DayNightClassifier
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
def load_models_v2():
    """Load and cache heavy AI models."""
    
    # Create a place for logs
    log_expander = st.expander("Show Model Loading Logs", expanded=True)
    with log_expander:
        log_placeholder = st.empty()
        
    redirector = StreamlitLogRedirector(log_placeholder)
    
    # Capture stdio
    with contextlib.redirect_stdout(redirector), contextlib.redirect_stderr(redirector):
        print("Starting Model Loading...")
        ocr = OCRProcessor()
        
        print("Initializing MegaDetector Wrapper...")
        # Initialize with default threshold, can be updated later
        md = MegaDetectorWrapper(confidence_threshold=0.2)
        
        
        print("Loading BioClip Classifier...")
        # Add a manual context manager context to avoid streamlit context missing warnings
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            import threading
            add_script_run_ctx(threading.current_thread())
        except ImportError:
            pass
        bio = BioClipClassifier(species_list=AnimalDetector.WILDLIFE_CLASSES)
        
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
            max_length = max(
                df_export[col].astype(str).apply(len).max(),
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
    
    st.divider()
    st.info("**Tip:** You can manually edit the detected animal names in the results table below.")
    st.info("**Night Vision Support:** The system automatically detects infrared/grayscale images and classifies them as 'Night' regardless of brightness.")

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload & Process", "Review Results", "Statistics", "History & Analytics", "Diagnostics"])

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
                        ocr_model, md_model, bio_model, dn_model = load_models_v2()
                        
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
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(results)
                        st.session_state.processed_data = df
                        
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
                    
                    if img_path and os.path.exists(str(img_path)):
                        st.image(img_path, width="stretch")
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
                        ocr_model, md_model, bio_model, dn_model = load_models_v2()
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
                    # Init specific debug processor
                    ocr_model, md_model, bio_model, dn_model = load_models_v2()
                    
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

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>Wildlife Camera Trap Auto-Analyzer v1.0 | Built with Streamlit</p>
        <p>Tip: The animal detection model can be swapped for specialized models like MegaDetector</p>
    </div>
""", unsafe_allow_html=True)
