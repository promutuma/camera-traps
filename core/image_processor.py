"""
Unified Image Processing Pipeline
Orchestrates OCR, animal detection, and day/night classification.
"""

import os
import hashlib
import cv2
from PIL import Image
from typing import Dict, Optional, Callable
from .ocr_processor import OCRProcessor
from .animal_detector import EnsembleDetector
from .day_night_classifier import DayNightClassifier


class ImageProcessor:
    """Unified pipeline for processing camera trap images."""
    
    def __init__(self, 
                 ocr_processor,
                 animal_detector,
                 day_night_classifier,
                 ocr_enabled: bool = True,
                 detection_enabled: bool = True,
                 day_night_enabled: bool = True,
                 ocr_strip_percent: float = 0.10):
        """
        Initialize the image processor.
        
        Args:
            ocr_processor: Injected OCRProcessor instance
            animal_detector: Injected AnimalDetector instance
            day_night_classifier: Injected DayNightClassifier instance
            ocr_enabled: Enable OCR metadata extraction
            detection_enabled: Enable animal detection
            day_night_enabled: Enable day/night classification
        """
        self.ocr_enabled = ocr_enabled
        self.detection_enabled = detection_enabled
        self.day_night_enabled = day_night_enabled
        self.ocr_strip_percent = ocr_strip_percent
        
        self.ocr_processor = ocr_processor if ocr_enabled else None
        self.animal_detector = animal_detector if detection_enabled else None
        self.day_night_classifier = day_night_classifier if day_night_enabled else None

    @staticmethod
    def get_image_hash(image_path: str) -> str:
        """Generate a unique ID based on image content (MD5)."""
        try:
            with open(image_path, "rb") as f:
                file_hash = hashlib.md5()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
            return file_hash.hexdigest()
        except Exception:
            return "unknown_hash"
    
    def process_single_image(self, image_path: str, progress_callback: Optional[Callable] = None) -> list:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the image file
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of dictionaries containing all extracted information (one per detected entity)
        """
        base_result = {
            'image_id': self.get_image_hash(image_path),
            'filename': os.path.basename(image_path),
            'filepath': image_path,
            'temperature': None,
            'date': None,
            'time': None,
            'day_night': 'Unknown',
            'brightness': 0.0,
            'species_data': [], # Initialize structured data
            'user_notes': '',
            'processing_status': 'Success'
        }
        
        try:
            # 1. OCR Processing (Common to all)
            if self.ocr_enabled and self.ocr_processor:
                if progress_callback:
                    progress_callback(f"Extracting metadata from {base_result['filename']}...")
                ocr_metadata = self.ocr_processor.process_image(image_path, strip_height_percent=self.ocr_strip_percent)
                base_result.update(ocr_metadata)
                
            # Step 2: Day/Night Classification (Common to all)
            if self.day_night_enabled and self.day_night_classifier:
                if progress_callback:
                    progress_callback(f"Classifying day/night for {base_result['filename']}...")
                
                classification, brightness = self.day_night_classifier.classify(image_path)
                base_result['day_night'] = classification
                base_result['brightness'] = brightness
            
            # 3. Animal Detection (Returns List)
            final_results = []
            
            if self.detection_enabled and self.animal_detector:
                if progress_callback:
                    progress_callback(f"Detecting animal in {base_result['filename']}...")
                
                detections = self.animal_detector.detect(image_path)
                # detections is now a List[Dict], ensuring at least one 'Empty' or valid detections
                
                for det in detections:
                    # Create a copy of base metadata for each detection
                    row = base_result.copy()
                    row['detected_animal'] = det['detected_animal']
                    row['primary_label'] = det.get('primary_label', 'Unidentified')
                    row['species_label'] = det.get('species_label', 'N/A')
                    row['detection_confidence'] = det['detection_confidence']
                    row['bbox'] = det['bbox']
                    row['detection_method'] = det.get('method', 'Unknown')
                    row['species_data'] = det.get('species_data', [])
                    final_results.append(row)
            else:
                # If detection disabled, just return the base metadata as one row
                row = base_result.copy()
                row['detected_animal'] = 'Unidentified'
                row['primary_label'] = 'N/A'
                row['species_label'] = 'N/A'
                row['detection_confidence'] = 0.0
                row['bbox'] = None
                row['detection_method'] = 'None'
                final_results.append(row)
                
        except Exception as e:
            # On error, return one error row
            base_result['processing_status'] = f'Error: {str(e)}'
            base_result['detected_animal'] = 'Error'
            print(f"Error processing {image_path}: {str(e)}")
            return [base_result]
        
        return final_results

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
            
        bc_debug = []
        if hasattr(self.animal_detector, 'bioclip') and self.animal_detector.bioclip:
             try:
                 # Load image for BioClip
                 img = cv2.imread(image_path)
                 if img is not None:
                      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                      pil_img = Image.fromarray(img)
                      bc_debug = self.animal_detector.bioclip.predict_list(pil_img, threshold=0.0, top_k=20)
             except Exception as e:
                 print(f"BioClip debug error: {e}")
            
        return {
            'ocr': ocr_debug,
            'megadetector': md_debug,
            'megadetector_status': md_status,
            'bioclip': bc_debug
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
            results.extend(result)
        
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
