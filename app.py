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
from datetime import datetime
from io import BytesIO
from PIL import Image
import cv2

from core.image_processor import ImageProcessor
from core.db_manager import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="Wildlife Analysis Dashboard",
    page_icon="🦁",
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


def display_image_with_info(image_path, detection_info):
    """Display image with detection information overlay and bounding boxes."""
    try:
        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Draw bounding box if available
        if detection_info.get('bbox') is not None:
            bbox = detection_info['bbox']
            # Convert normalized bbox to pixel coordinates
            x, y, box_w, box_h = bbox
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + box_w) * w)
            y2 = int((y + box_h) * h)
            
            # Draw green rectangle
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add label background
            label = f"{detection_info['detected_animal']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(image_rgb, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 255, 0), -1)
            cv2.putText(image_rgb, label, (x1 + 5, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        # Add overall detection info overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{detection_info['detected_animal']} ({detection_info['detection_confidence']:.2f})"
        
        # Add semi-transparent background for text
        overlay = image_rgb.copy()
        cv2.rectangle(overlay, (10, 10), (400, 60), (0, 0, 0), -1)
        image_rgb = cv2.addWeighted(overlay, 0.3, image_rgb, 0.7, 0)
        
        # Add text
        cv2.putText(image_rgb, text, (20, 45), font, 0.8, (255, 255, 255), 2)
        
        return Image.fromarray(image_rgb)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        return None


# Main UI
st.markdown('<div class="main-header">🦁 Wildlife Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Camera Trap Image Analysis</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Processing Options")
    enable_ocr = st.checkbox("Enable OCR Metadata Extraction", value=True)
    enable_detection = st.checkbox("Enable Animal Detection", value=True)
    enable_day_night = st.checkbox("Enable Day/Night Classification", value=True)
    
    # Detection mode selector
    if enable_detection:
        detection_mode = st.radio(
            "Detection Model",
            options=["ensemble", "megadetector", "mobilenet"],
            format_func=lambda x: {
                "ensemble": "🎯 Ensemble (MegaDetector + MobileNetV2) - Best Accuracy",
                "megadetector": "🔍 MegaDetector Only - Fast Detection",
                "mobilenet": "🤖 MobileNetV2 Only - Species Classification"
            }[x],
            index=0,
            help="Ensemble mode combines both models for highest accuracy"
        )
    else:
        detection_mode = "ensemble"
    
    st.subheader("Advanced Settings")
    detection_confidence = st.slider(
        "Detection Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.15,
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
    st.info("💡 **Tip:** You can manually edit the detected animal names in the results table below.")
    st.info("🌙 **Night Vision Support:** The system automatically detects infrared/grayscale images and classifies them as 'Night' regardless of brightness.")

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload & Process", "📊 Review Results", "📈 Statistics", "📚 History & Analytics", "🔧 Diagnostics"])

with tab1:
    st.header("Upload Camera Trap Images")
    
    uploaded_files = st.file_uploader(
        "Choose images (JPG/PNG)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Select multiple camera trap images to process"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🚀 Process Images", type="primary"):
                try:
                    with st.spinner("Initializing processing modules..."):
                        # Initialize processor
                        st.info("📥 Loading OCR, detection, and classification models. First run may take a few minutes to download models (~100MB)...")
                        processor = ImageProcessor(
                            ocr_enabled=enable_ocr,
                            detection_enabled=enable_detection,
                            day_night_enabled=enable_day_night,
                            detection_confidence=detection_confidence,
                            brightness_threshold=brightness_threshold,
                            detection_mode=detection_mode,
                            ocr_strip_percent=ocr_strip_height
                        )

                        st.success("✅ Models loaded successfully!")
                    
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
                        
                        st.balloons()
                        st.success("🎉 Processing complete! Switch to the 'Review Results' tab to see the data.")
                
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.error(f"Error type: {type(e).__name__}")
                    import traceback
                    st.code(traceback.format_exc())


with tab2:
    if st.session_state.processed_data is not None:
        st.header("Review and Edit Results")
        
        df = st.session_state.processed_data
        
        # Display editable table
        st.subheader("📋 Processed Data (Editable)")
        
        # Create editable dataframe
        edited_df = st.data_editor(
            df[['filename', 'date', 'time', 'temperature', 'detected_animal', 
                'day_night', 'brightness', 'detection_confidence', 'detection_method', 'user_notes']],
            column_order=[
                "filename", "detected_animal", "detection_confidence", "detection_method", 
                "day_night", "brightness", "temperature", "date", "time", "user_notes"
            ],
            column_config={
                "filename": "Image File",
                "detected_animal": "Detected Animal",
                "detection_confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    help="Detection confidence score",
                    format="%.2f",
                    min_value=0,
                    max_value=1
                ),
                "detection_method": st.column_config.TextColumn(
                    "Method",
                    help="Model used for detection"
                ),
                "day_night": st.column_config.SelectboxColumn(
                    'Day/Night',
                    options=['Day', 'Night', 'Unknown']
                ),
                "brightness": st.column_config.NumberColumn(
                    "Brightness",
                    help="Average image brightness (0-255)",
                    format="%.1f"
                ),
                "temperature": "Temp (°C)",
                "date": "Date",
                "time": "Time",
                "user_notes": "Notes"
            },
            hide_index=True,
            use_container_width=True,
            key="results_editor"
        )
        
        # Update session state with edited data
        st.session_state.processed_data = edited_df
        
        # Debug info
        with st.expander("🔍 Debug Information"):
            st.write("Raw data with debug fields:")
            st.json(edited_df.to_dict(orient='records'))
        
        # Image viewer
        st.subheader("🖼️ Image Viewer")
        
        if len(st.session_state.image_paths) > 0:
            selected_idx = st.selectbox(
                "Select image to view:",
                range(len(st.session_state.image_paths)),
                format_func=lambda x: df.iloc[x]['filename']
            )
            
            if selected_idx is not None:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    image_path = st.session_state.image_paths[selected_idx]
                    detection_info = df.iloc[selected_idx].to_dict()
                    
                    display_img = display_image_with_info(image_path, detection_info)
                    if display_img:
                        st.image(display_img, use_container_width=True)
                
                with col2:
                    st.markdown("### 📝 Image Details")
                    info = df.iloc[selected_idx]
                    st.write(f"**Filename:** {info['filename']}")
                    st.write(f"**Date:** {info['date'] or 'N/A'}")
                    st.write(f"**Time:** {info['time'] or 'N/A'}")
                    st.write(f"**Temperature:** {info['temperature'] or 'N/A'}")
                    st.write(f"**Animal:** {info['detected_animal']}")
                    st.write(f"**Confidence:** {info['detection_confidence']:.2%}")
                    st.write(f"**Day/Night:** {info['day_night']}")
                    st.write(f"**Brightness:** {info.get('brightness', 0.0):.1f}")
        
        # Download report
        st.divider()
        st.subheader("📥 Export Data")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            excel_data = create_excel_report(st.session_state.processed_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_data,
                file_name=f"wildlife_analysis_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        
        with col2:
            csv_data = st.session_state.processed_data.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"wildlife_analysis_{timestamp}.csv",
                mime="text/csv"
            )
            
        with col3:
            if st.button("💾 Save to History", type="secondary", use_container_width=True):
                count = st.session_state.db_manager.save_results(st.session_state.processed_data)
                if count > 0:
                    st.success(f"Successfully saved {count} records to database!")
                else:
                    st.warning("No new data to save.")
    
    else:
        st.info("👈 Please upload and process images first in the 'Upload & Process' tab.")

with tab3:
    if st.session_state.processed_data is not None:
        st.header("📈 Analysis Statistics")
        
        df = st.session_state.processed_data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", len(df))
        
        with col2:
            identified = len(df[df['detected_animal'] != 'Unidentified'])
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
            st.subheader("🦁 Animal Distribution")
            animal_counts = df['detected_animal'].value_counts()
            st.bar_chart(animal_counts)
        
        with col2:
            st.subheader("🌓 Day/Night Distribution")
            day_night_counts = df['day_night'].value_counts()
            st.bar_chart(day_night_counts)
        
        # Detection confidence distribution
        st.subheader("📊 Detection Confidence Distribution")
        st.line_chart(df['detection_confidence'])
        
    else:
        st.info("👈 Please upload and process images first in the 'Upload & Process' tab.")

with tab4:
    st.header("📚 Analysis History")
    
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
            use_container_width=True,
            hide_index=True
        )
        
        # Helper to convert df to csv
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(history_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Export Full History (CSV)",
                data=csv,
                file_name=f"wildlife_history_export_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        with col2:
            if st.button("🗑️ Clear History", type="secondary"):
                if st.checkbox("Confirm Clear History? This cannot be undone."):
                    st.session_state.db_manager.clear_history()
                    st.rerun()
    else:
        st.info("No historical data found. Process and save images to build your history!")

with tab5:
    st.header("🔧 Deep Inspection Tool")
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
            st.image(target_debug_img, caption="Target Image", use_container_width=True)
            
            ocr_strip_pct = st.slider("OCR Strip Height (%)", 0.05, 0.30, 0.10, 0.01, help="Adjust crop area for metadata")
            
        with col_dbg_2:
            if st.button("🔍 Run Deep Inspection", type="primary"):
                with st.spinner("Analyzing internals..."):
                    # Init specific debug processor
                    debug_processor = ImageProcessor(
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
                        st.dataframe(md_df, use_container_width=True)
                        
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
                                
                        st.image(img_vis, caption="Visualized Raw Detections (Red=<0.2, Orange=<0.8, Green=>0.8)", use_container_width=True)
                    else:
                        st.warning("No MegaDetector candidates found.")

                    st.divider()

                    # 3. MobileNet Fallback
                    st.subheader("3. MobileNetV2 Top-5 Predictions")
                    mn_data = debug_info.get('mobilenet')
                    if mn_data:
                        st.write(mn_data)
                    else:
                        st.warning("MobileNet data unavailable")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>Wildlife Camera Trap Auto-Analyzer v1.0 | Built with Streamlit</p>
        <p>💡 Tip: The animal detection model can be swapped for specialized models like MegaDetector</p>
    </div>
""", unsafe_allow_html=True)
