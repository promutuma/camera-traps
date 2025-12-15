"""
Unified Image Processing Pipeline
Orchestrates OCR, animal detection, and day/night classification.
"""

import os
from typing import Dict, Optional, Callable
from .ocr_processor import OCRProcessor
from .animal_detector import EnsembleDetector
from .day_night_classifier import DayNightClassifier


class ImageProcessor:
    """Unified pipeline for processing camera trap images."""
    
    def __init__(self, 
                 ocr_enabled: bool = True,
                 detection_enabled: bool = True,
                 day_night_enabled: bool = True,
                 detection_confidence: float = 0.3,
                 brightness_threshold: int = 100,
                 detection_mode: str = 'ensemble',
                 ocr_strip_percent: float = 0.10):
        """
        Initialize the image processor.
        
        Args:
            ocr_enabled: Enable OCR metadata extraction
            detection_enabled: Enable animal detection
            day_night_enabled: Enable day/night classification
            detection_confidence: Confidence threshold for animal detection
            brightness_threshold: Brightness threshold for day/night classification
            detection_mode: Detection mode - 'ensemble', 'megadetector', or 'mobilenet'
        """
        self.ocr_enabled = ocr_enabled
        self.detection_enabled = detection_enabled
        self.day_night_enabled = day_night_enabled
        self.ocr_strip_percent = ocr_strip_percent
        
        # Initialize processors
        self.ocr_processor = OCRProcessor() if ocr_enabled else None
        self.animal_detector = EnsembleDetector(
            confidence_threshold=detection_confidence
        ) if detection_enabled else None
        self.day_night_classifier = DayNightClassifier(brightness_threshold=brightness_threshold) if day_night_enabled else None
    
    def process_single_image(self, image_path: str, progress_callback: Optional[Callable] = None) -> Dict:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the image file
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing all extracted information
        """
        result = {
            'filename': os.path.basename(image_path),
            'filepath': image_path,
            'temperature': None,
            'date': None,
            'time': None,
            'detected_animal': 'Unidentified',
            'primary_label': 'N/A',
            'species_label': 'N/A',
            'detection_confidence': 0.0,
            'bbox': None,
            'detection_method': 'None',
            'day_night': 'Unknown',
            'brightness': 0.0,
            'user_notes': '',
            'processing_status': 'Success'
        }
        
        try:
            # 1. OCR Processing
            if self.ocr_enabled and self.ocr_processor:
                if progress_callback:
                    progress_callback(f"Extracting metadata from {result['filename']}...")
                ocr_metadata = self.ocr_processor.process_image(image_path, strip_height_percent=self.ocr_strip_percent)
                result.update(ocr_metadata)
            
            # 2. Animal Detection
            if self.detection_enabled and self.animal_detector:
                if progress_callback:
                    progress_callback(f"Detecting animal in {result['filename']}...")
                
                detection_result = self.animal_detector.detect(image_path)
                result['detected_animal'] = detection_result['detected_animal']
                result['primary_label'] = detection_result.get('primary_label', 'Unidentified')
                result['species_label'] = detection_result.get('species_label', 'N/A')
                result['detection_confidence'] = detection_result['detection_confidence']
                result['bbox'] = detection_result['bbox']
                result['detection_method'] = detection_result.get('method', 'Unknown')
            
            # Step 3: Day/Night Classification
            if self.day_night_enabled and self.day_night_classifier:
                if progress_callback:
                    progress_callback(f"Classifying day/night for {result['filename']}...")
                
                classification, brightness = self.day_night_classifier.classify(image_path)
                result['day_night'] = classification
                result['brightness'] = brightness
            
        except Exception as e:
            result['processing_status'] = f'Error: {str(e)}'
            print(f"Error processing {image_path}: {str(e)}")
        
        return result

    def get_debug_info(self, image_path: str, ocr_strip_percent: float = 0.10) -> Dict:
        """Get comprehensive debug info for an image."""
        # OCR Debug
        if self.ocr_processor:
            ocr_crop, ocr_text, ocr_parsed = self.ocr_processor.get_debug_data(image_path, strip_height_percent=ocr_strip_percent)
            ocr_debug = {
                'crop': ocr_crop,
                'raw_text': ocr_text,
                'parsed': ocr_parsed
            }
        else:
            ocr_debug = None
        
        # Detector Debug
        md_debug = []
        md_status = None
        if hasattr(self.animal_detector, 'megadetector') and self.animal_detector.megadetector:
            raw_result = self.animal_detector.megadetector.detect_all(image_path)
            md_debug = raw_result.get('detections', []) if isinstance(raw_result, dict) else []
            md_status = self.animal_detector.megadetector.get_status()
            
        mn_debug = []
        if hasattr(self.animal_detector, 'mobilenet') and self.animal_detector.mobilenet:
            mn_debug = self.animal_detector.mobilenet.get_raw_classifications(image_path)
            
        return {
            'ocr': ocr_debug,
            'megadetector': md_debug,
            'megadetector_status': md_status,
            'mobilenet': mn_debug
        }
    
    def process_batch(self, image_paths: list, progress_callback: Optional[Callable] = None) -> list:
        """
        Process multiple images through the pipeline.
        
        Args:
            image_paths: List of image file paths
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of result dictionaries
        """
        results = []
        total = len(image_paths)
        
        for idx, image_path in enumerate(image_paths, 1):
            if progress_callback:
                progress_callback(f"Processing image {idx}/{total}: {os.path.basename(image_path)}")
            
            result = self.process_single_image(image_path, progress_callback)
            results.append(result)
        
        return results


def process_images(image_paths: list, 
                   progress_callback: Optional[Callable] = None,
                   **kwargs) -> list:
    """
    Convenience function to process multiple images.
    
    Args:
        image_paths: List of image file paths
        progress_callback: Optional callback function for progress updates
        **kwargs: Additional arguments for ImageProcessor initialization
        
    Returns:
        List of result dictionaries
    """
    processor = ImageProcessor(**kwargs)
    return processor.process_batch(image_paths, progress_callback)
