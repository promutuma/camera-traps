
"""
Professional Wildlife Identification Pipeline
Step 1: MegaDetector V5a (megadetector) -> Primary Label (Animal/Person/Vehicle/Empty)
Step 2: Filter (Stop if not Animal)
Step 3: Crop (Pad 10%)
Step 4: BioClip (Species Classification)
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any

# Local
from .bioclip_classifier import BioClipClassifier

# MegaDetector (Official Package)
try:
    from megadetector.detection import run_detector
    MD_AVAILABLE = True
except ImportError as e:
    MD_AVAILABLE = False
    print(f"Warning: megadetector not installed. Error: {e}")

class MegaDetectorWrapper:
    """Wrapper for MegaDetector V5a using official megadetector package."""
    
    # MDv5a Classes: 1=Animal, 2=Person, 3=Vehicle
    CLASS_MAP = {'1': 'Animal', '2': 'Person', '3': 'Vehicle'}
    
    def __init__(self, confidence_threshold: float = 0.2):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.load_error = None
        self._load_model()
        
    def _load_model(self):
        if not MD_AVAILABLE:
            self.load_error = "megadetector library not found"
            return
            
        try:
            print("Loading MegaDetector V5a...")
            # Automatically downloads MDv5a if not present
            self.model = run_detector.load_detector("MDV5a")
            print("MDv5a loaded successfully.")
        except Exception as e:
            print(f"Error loading MDv5a: {e}")
            self.load_error = str(e)
            self.model = None

    def get_status(self) -> Dict:
        return {
            'loaded': self.model is not None,
            'error': self.load_error
        }

    def detect_primary(self, image_path: str) -> Tuple[str, float, Optional[List[float]]]:
        """
        Run detection and return primary label.
        Returns: (Label, Confidence, BBox [x,y,w,h])
        """
        if self.model is None:
            return 'Empty', 0.0, None

        try:
            image = Image.open(image_path)
            result = self.model.generate_detections_one_image(image)
            
            # result structure: {'detections': [{'category': '1', 'conf': 0.9, 'bbox': [x,y,w,h]}, ...]}
            detections = result.get('detections', [])
            
            best_det = None
            best_conf = 0.0
            
            for det in detections:
                conf = det['conf']
                if conf > best_conf and conf >= self.confidence_threshold:
                    best_conf = conf
                    best_det = det
            
            if best_det:
                cat_id = best_det['category'] # String '1', '2', '3'
                label = self.CLASS_MAP.get(cat_id, 'Unknown')
                return label, best_conf, best_det['bbox']
            else:
                return 'Empty', 0.0, None

        except Exception as e:
            print(f"Error in MDv5a inference: {e}")
            return 'Empty', 0.0, None

    def detect_all(self, image_path: str) -> Any:
        """Return all raw detections for diagnostics."""
        if self.model is None:
            return {'detections': []}
        try:
            image = Image.open(image_path)
            return self.model.generate_detections_one_image(image)
        except Exception as e:
            return {'error': str(e)}

class AnimalDetector:
    """
    Orchestrator for the Wildlife Identification Pipeline.
    """
    
    WILDLIFE_CLASSES = [
        'zebra', 'elephant', 'lion', 'leopard', 'cheetah', 'giraffe', 'buffalo', 
        'hyena', 'gazelle', 'impala', 'warthog', 'baboon', 'monkey', 'rhinoceros', 
        'hippopotamus', 'crocodile', 'ostrich', 'antelope', 'wildebeest', 'human'
    ]

    def __init__(self, confidence_threshold: float = 0.2):
        self.megadetector = MegaDetectorWrapper(confidence_threshold)
        self.bioclip = BioClipClassifier(species_list=self.WILDLIFE_CLASSES)
        
    def detect(self, image_path: str) -> Dict:
        """
        Pipeline:
        1. MDv5a Detection
        2. Filter (Stop if not Animal)
        3. Crop
        4. BioClip Classification
        """
        result = {
            'detected_animal': 'Empty', # Legacy/Display default
            'primary_label': 'Empty',   # Animal, Person, Vehicle, Empty
            'species_label': 'Not Animal', # Species name or 'Not Animal'
            'detection_confidence': 0.0,
            'bbox': None,
            'method': 'MDv5a',
            'secondary_method': None
        }
        
        # Step 1: MDv5a
        label, conf, bbox = self.megadetector.detect_primary(image_path)
        
        # Default assignment
        result['primary_label'] = label
        result['detected_animal'] = label # Default to primary
        result['species_label'] = 'Not Animal'
        result['detection_confidence'] = conf
        result['bbox'] = bbox
        
        # Step 2: Filter
        if label != 'Animal':
            return result
            
        # If Animal, proceed to Step 3 & 4
        try:
            # Step 3: Crop with padding
            # Helper to crop
            img = cv2.imread(image_path)
            if img is None:
                return result # Fail safe
                
            h, w = img.shape[:2]
            x, y, box_w, box_h = bbox
            
            # Convert normalized to pixel
            x_px = int(x * w)
            y_px = int(y * h)
            w_px = int(box_w * w)
            h_px = int(box_h * h)
            
            # Add 10% padding
            pad_w = int(w_px * 0.1)
            pad_h = int(h_px * 0.1)
            
            x1 = max(0, x_px - pad_w)
            y1 = max(0, y_px - pad_h)
            x2 = min(w, x_px + w_px + pad_w)
            y2 = min(h, y_px + h_px + pad_h)
            
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                return result
            
            # Convert BGR (OpenCV) to RGB (PIL) for BioClip
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            
            # Step 4: BioClip
            species, bio_conf = self.bioclip.predict(crop_pil)
            
            # Update result
            result['primary_label'] = 'Animal'
            result['species_label'] = species.title()
            result['detected_animal'] = species.title() # Update legacy to specific
            result['detection_confidence'] = bio_conf
            result['method'] = 'MDv5a + BioClip'
            result['secondary_method'] = 'BioClip'
            
        except Exception as e:
            print(f"Error in BioClip step: {e}")
            # Fallback to just "Animal"
            
        return result

# Compatibility alias
EnsembleDetector = AnimalDetector
