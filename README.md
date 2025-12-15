# Wildlife Camera Trap Auto-Analyzer

A user-friendly **Streamlit-based dashboard** for automated analysis of wildlife camera trap images. This system extracts metadata via OCR, detects animals using pre-trained models, classifies day/night conditions, and generates downloadable reports.

![Wildlife Analysis Dashboard](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)

## 🌟 Features

- **📤 Batch Image Upload**: Upload multiple camera trap images (JPG/PNG) simultaneously
- **🔍 OCR Metadata Extraction**: Automatically extracts date, time, and temperature from image metadata strips
- **🦁 Animal Detection**: Uses MobileNetV2 to identify wildlife (with modular design for easy model swapping)
- **🌓 Day/Night Classification**: Automatically classifies images based on brightness analysis
- **✏️ Editable Results**: Review and manually correct detected animal names in an interactive table
- **📊 Excel Reports**: Generate downloadable Excel (.xlsx) reports with all extracted data
- **📈 Statistics Dashboard**: View analysis statistics and distributions

## 🏗️ Project Structure

```
camera-traps/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── utils/                          # Processing modules
    ├── __init__.py
    ├── ocr_processor.py           # OCR metadata extraction
    ├── animal_detector.py         # Animal detection (MobileNetV2)
    ├── day_night_classifier.py   # Day/night classification
    └── image_processor.py         # Unified processing pipeline
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or navigate to the project directory:**

   ```bash
   cd c:\wamp64\www\camera-traps
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** The first run will download the MobileNetV2 model weights (~14MB) and EasyOCR language models (~100MB). This is a one-time download.

## 💻 Usage

### Running the Application

1. **Start the Streamlit server:**

   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   - The application will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

### Using the Dashboard

1. **Upload Images** (Tab 1):

   - Click "Browse files" to select multiple camera trap images
   - Adjust processing options in the sidebar if needed
   - Click "🚀 Process Images" to start analysis
   - Watch the progress bar as images are processed

2. **Review Results** (Tab 2):

   - View all extracted data in an editable table
   - Click on any cell to edit (especially useful for correcting animal names)
   - Select images from the dropdown to view them with detection overlays
   - Add custom notes in the "User Notes" column

3. **Download Reports**:

   - Click "📊 Download Excel Report" for a formatted .xlsx file
   - Or click "📄 Download CSV" for a simple CSV export

4. **View Statistics** (Tab 3):
   - See summary metrics (total images, identified animals, day/night counts)
   - View animal distribution charts
   - Analyze detection confidence trends

## ⚙️ Configuration

### Sidebar Settings

- **Processing Options:**
  - Enable/disable OCR, animal detection, or day/night classification
- **Advanced Settings:**
  - **Detection Confidence Threshold** (0.0-1.0): Minimum confidence to accept predictions
  - **Brightness Threshold** (0-255): Pixel brightness cutoff for day/night classification

## 🔧 Technical Details

### OCR Metadata Extraction

- Uses **EasyOCR** to read text from the bottom 10% of images
- Parses format: `M [Temp] [Date] [Time]`
- Supports multiple date/time formats

### Animal Detection

- Uses **MobileNetV2** pre-trained on ImageNet
- Filters predictions to wildlife-related classes
- Returns "Unidentified" for low-confidence or non-wildlife predictions
- **Modular design** allows easy model swapping (see below)

### Day/Night Classification

- Analyzes mean pixel brightness in grayscale
- Configurable threshold (default: 100/255)
- Returns classification with confidence score

## 🔄 Swapping the Detection Model

The system is designed for easy model replacement. To use **MegaDetector** or another model:

1. **Modify `utils/animal_detector.py`:**

   ```python
   # Replace the _load_model() method
   def _load_model(self):
       # Load your custom model here
       self.model = load_megadetector_model()

   # Update the detect() method to match your model's output format
   ```

2. **Or use the `swap_model()` method:**

   ```python
   from utils.animal_detector import AnimalDetector

   detector = AnimalDetector()
   detector.swap_model(your_custom_model)
   ```

## 📋 Output Format

### Excel Report Columns

- **Filename**: Original image filename
- **Date**: Extracted date from metadata
- **Time**: Extracted time from metadata
- **Temperature**: Extracted temperature reading
- **Detected Animal**: Identified animal species
- **Day/Night**: Classification result
- **User Notes**: Custom notes added during review

## 🐛 Troubleshooting

### Common Issues

1. **Unicode encoding errors on Windows:**

   - **Fixed automatically**: The application now includes UTF-8 encoding configuration
   - This prevents errors when EasyOCR downloads model files with Unicode progress bars
   - No action needed from users

2. **"Could not read image" errors:**

   - Ensure images are valid JPG/PNG files
   - Check file permissions

3. **Low detection accuracy:**

   - MobileNetV2 is a general-purpose model
   - Consider lowering the confidence threshold
   - Manually correct predictions in the review table
   - For better accuracy, swap to a wildlife-specific model

4. **Night vision/infrared images:**

   - **Automatically handled**: The system detects grayscale/low-saturation images
   - Night vision images are automatically classified as "Night"
   - MegaDetector works well with infrared images
   - No special configuration needed

5. **OCR not extracting metadata:**

   - Verify the metadata strip is in the bottom 10% of the image
   - Check that text follows the expected format
   - Ensure sufficient image quality and contrast

6. **Slow processing:**
   - First run downloads model weights (one-time)
   - Processing speed depends on image size and quantity
   - Consider resizing very large images

## 🚀 Future Enhancements

- [ ] Integration with **MegaDetector** for improved wildlife detection
- [ ] Support for video file processing
- [ ] Batch export of cropped animal detections
- [ ] Database storage for long-term analysis
- [ ] GPS coordinate extraction and mapping
- [ ] Species-specific behavior analysis

## 📝 License

This project is provided as-is for wildlife research and conservation purposes.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional OCR engines (Tesseract integration)
- More sophisticated animal detection models
- Enhanced UI/UX features
- Performance optimizations

---

**Built with ❤️ for wildlife conservation**
