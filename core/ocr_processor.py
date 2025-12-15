"""
OCR Metadata Extraction Module
Extracts date, time, and temperature from camera trap image metadata strips.
"""

import os
import sys

# Fix Windows console encoding for EasyOCR progress bars
# Set environment variable before importing easyocr
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Reconfigure stdout/stderr encoding without wrapping
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
import easyocr
import re
from typing import Dict, Optional, Tuple


class OCRProcessor:
    """Handles OCR extraction from camera trap metadata strips."""
    
    def __init__(self):
        """Initialize EasyOCR reader with English language support."""
        self.reader = easyocr.Reader(['en'], gpu=False)
    
    def extract_metadata_strip(self, image: np.ndarray, strip_height_percent: float = 0.10) -> np.ndarray:
        """
        Crop the bottom portion of the image containing metadata.
        
        Args:
            image: Input image as numpy array
            strip_height_percent: Percentage of image height to crop from bottom (default 10%)
            
        Returns:
            Cropped metadata strip as numpy array
        """
        height = image.shape[0]
        strip_height = int(height * strip_height_percent)
        metadata_strip = image[-strip_height:, :]
        return metadata_strip
    
    def parse_metadata_text(self, text: str) -> Dict[str, Optional[str]]:
        """
        Parse metadata text in format "M [Temp] [Date] [Time]".
        
        Expected format examples:
        - "M 25C 2023-12-15 14:30:45"
        - "M 18°C 15/12/2023 2:30 PM"
        
        Args:
            text: OCR extracted text string
            
        Returns:
            Dictionary with keys: temperature, date, time
        """
        result = {
            'temperature': None,
            'date': None,
            'time': None
        }
        
        # Clean up text
        text = text.strip()
        
        # Pattern 1: M [Temp] [Date] [Time]
        # Temperature: digits followed by C or °C or F
        temp_pattern = r'(\d+\.?\d*)\s*[°]?[CF]'
        
        # Date patterns: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, DD-MM-YYYY
        date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})'
        
        # Time patterns: HH:MM:SS, HH:MM, HH:MM AM/PM
        time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)'
        
        # Extract temperature
        temp_match = re.search(temp_pattern, text, re.IGNORECASE)
        if temp_match:
            result['temperature'] = temp_match.group(0)
        
        # Extract date
        date_match = re.search(date_pattern, text)
        if date_match:
            result['date'] = date_match.group(1)
        
        # Extract time
        time_match = re.search(time_pattern, text, re.IGNORECASE)
        if time_match:
            result['time'] = time_match.group(1)
        
        return result
    
    def process_image(self, image_path: str, strip_height_percent: float = 0.10) -> Dict[str, Optional[str]]:
        """
        Main processing function to extract metadata from camera trap image.
        
        Args:
            image_path: Path to the image file
            strip_height_percent: Percentage of image height to crop from bottom
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Extract metadata strip
            metadata_strip = self.extract_metadata_strip(image, strip_height_percent)
            
            # Perform OCR
            ocr_results = self.reader.readtext(metadata_strip)
            
            # Combine all detected text
            full_text = ' '.join([text for (_, text, _) in ocr_results])
            
            # Parse metadata
            metadata = self.parse_metadata_text(full_text)
            
            return metadata
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return {
                'temperature': None,
                'date': None,
                'time': None
            }

    def get_debug_data(self, image_path: str, strip_height_percent: float = 0.10) -> Tuple[Optional[np.ndarray], str, Dict]:
        """
        Get debug info for OCR process.
        
        Args:
            image_path: Path to image
            strip_height_percent: Percent to crop
            
        Returns:
            Tuple of (cropped_image, raw_text, parsed_metadata)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None, "Error loading image", {}
                
            metadata_strip = self.extract_metadata_strip(image, strip_height_percent)
            ocr_results = self.reader.readtext(metadata_strip)
            full_text = ' '.join([text for (_, text, _) in ocr_results])
            metadata = self.parse_metadata_text(full_text)
            
            return metadata_strip, full_text, metadata
        except Exception as e:
            return None, f"Error: {e}", {}


def extract_metadata(image_path: str) -> Dict[str, Optional[str]]:
    """
    Convenience function to extract metadata from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing extracted metadata
    """
    processor = OCRProcessor()
    return processor.process_image(image_path)
