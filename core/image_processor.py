"""
Unified Image Processing Pipeline
Orchestrates OCR, animal detection, and day/night classification.
"""

import os
import hashlib
import threading
from datetime import datetime
import cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import Dict, Optional, Callable
from .ocr_processor import OCRProcessor
from .animal_detector import EnsembleDetector
from .day_night_classifier import DayNightClassifier

# Executor for parallel work within the pipeline (e.g. future OCR batching).
_pipeline_executor = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4)))

# EasyOCR's Reader.readtext() shares internal numpy/model buffers and is not
# safe to call concurrently on the same instance. This lock serializes OCR
# calls while still allowing day/night + detection to run in parallel.
_ocr_lock = threading.Lock()


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
        """Generate a unique ID based on image content (SHA-256)."""
        try:
            with open(image_path, "rb") as f:
                file_hash = hashlib.sha256()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
            return file_hash.hexdigest()
        except Exception:
            return "unknown_hash"

    def _do_ocr(self, image_path: str) -> dict:
        """Run OCR under the shared lock (EasyOCR is not thread-safe)."""
        with _ocr_lock:
            return self.ocr_processor.process_image(
                image_path, strip_height_percent=self.ocr_strip_percent
            )

    def _mtime_fallback(self, image_path: str) -> tuple[Optional[str], Optional[str]]:
        """File modification time as YYYY-MM-DD / HH:MM:SS when OCR has no timestamp."""
        try:
            fallback_dt = datetime.fromtimestamp(os.path.getmtime(image_path))
            return fallback_dt.strftime("%Y-%m-%d"), fallback_dt.strftime("%H:%M:%S")
        except OSError:
            return None, None

    def _do_day_night_pixels(self, image_path: str) -> tuple[str, float]:
        """Pixel brightness + night-vision heuristics (runs in parallel with OCR)."""
        image = cv2.imread(image_path)
        if image is None:
            return "Unknown", 0.0
        brightness = self.day_night_classifier.calculate_brightness(image)
        label = self.day_night_classifier.classify_from_pixels(image)
        return label, brightness

    def process_single_image(
        self,
        image_path: str,
        progress_callback: Optional[Callable] = None,
        station_latitude: Optional[float] = None,
        station_longitude: Optional[float] = None,
    ) -> list:
        """
        Process a single image through the complete pipeline.

        OCR and pixel-based day/night run in parallel (thread pool). After both
        finish, day/night is refined with the OCR or file-mtime timestamp when
        available. Detection runs last (needs the final day/night label).

        Args:
            image_path: Path to the image file
            progress_callback: Optional callback function for progress updates
            station_latitude: Optional station latitude for solar day/night
            station_longitude: Optional station longitude for solar day/night

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
            'species_data': [],
            'user_notes': '',
            'processing_status': 'Success'
        }

        try:
            filename = base_result['filename']
            if progress_callback:
                progress_callback(f"Processing {filename}...")

            ocr_future = None
            dn_future = None

            if self.ocr_enabled and self.ocr_processor:
                ocr_future = _pipeline_executor.submit(self._do_ocr, image_path)
            if self.day_night_enabled and self.day_night_classifier:
                dn_future = _pipeline_executor.submit(self._do_day_night_pixels, image_path)

            if ocr_future:
                base_result.update(ocr_future.result())

            mtime_date, mtime_time = self._mtime_fallback(image_path)
            if not base_result.get("time") and mtime_time:
                if not base_result.get("date"):
                    base_result["date"] = mtime_date
                base_result["time"] = mtime_time

            pixel_label, brightness = "Unknown", 0.0
            if dn_future:
                pixel_label, brightness = dn_future.result()

            if self.day_night_enabled and self.day_night_classifier:
                ts_label = self.day_night_classifier.classify_from_timestamp(
                    base_result.get("date"),
                    base_result.get("time"),
                    station_latitude,
                    station_longitude,
                )
                base_result["day_night"] = ts_label if ts_label else pixel_label
                base_result["brightness"] = brightness

            # 3. Animal Detection (Returns List)
            final_results = []

            if self.detection_enabled and self.animal_detector:
                
                is_night = base_result.get("day_night") == "Night"
                detections = self.animal_detector.detect(image_path, is_night=is_night)
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
                    row['speciesnet_confidence'] = det.get('speciesnet_confidence', 0.0)
                    row['sn_raw_results'] = det.get('sn_raw_results', [])
                    row['model_breakdown'] = det.get('model_breakdown')
                    row['md_confidence'] = det.get('md_confidence')
                    row['raw_model_output'] = det.get('raw_model_output')
                    if '_model_events' in det:
                        row['_model_events'] = det['_model_events']
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
            
        return {
            'ocr': ocr_debug,
            'megadetector': md_debug,
            'megadetector_status': md_status,
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
