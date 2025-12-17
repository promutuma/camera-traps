
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
        
    def set_confidence_threshold(self, threshold: float):
        self.confidence_threshold = threshold
        
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

    def detect_all_candidates(self, image_path: str) -> List[Dict]:
        """
        Run detection and return all valid candidates above threshold.
        Returns: List of dicts {'label': str, 'conf': float, 'bbox': [x,y,w,h]}
        """
        if self.model is None:
            return []

        try:
            image = Image.open(image_path)
            result = self.model.generate_detections_one_image(image)
            
            detections = result.get('detections', [])
            candidates = []
            
            for det in detections:
                conf = det['conf']
                if conf >= self.confidence_threshold:
                    cat_id = det['category']
                    label = self.CLASS_MAP.get(cat_id, 'Unknown')
                    candidates.append({
                        'label': label,
                        'conf': conf,
                        'bbox': det['bbox']
                    })
            
            return candidates

        except Exception as e:
            print(f"Error in MDv5a inference: {e}")
            return []

    def detect_primary(self, image_path: str) -> Tuple[str, float, Optional[List[float]]]:
        """Legacy support: Return best detection."""
        candidates = self.detect_all_candidates(image_path)
        if not candidates:
            return 'Empty', 0.0, None
            
        # Sort by confidence
        best = max(candidates, key=lambda x: x['conf'])
        return best['label'], best['conf'], best['bbox']

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

    def __init__(self, megadetector, bioclip, confidence_threshold: float = 0.2):
        self.megadetector = megadetector
        self.bioclip = bioclip
        # Update threshold on the injected instance
        if self.megadetector:
            self.megadetector.set_confidence_threshold(confidence_threshold)
        
    def detect(self, image_path: str) -> List[Dict]:
        """
        Pipeline:
        1. MDv5a Detection (All candidates)
        2. Filter (Stop if not Animal)
        3. Crop
        4. BioClip Classification
        
        Returns: List of detection result dictionaries
        """
        base_result = {
            'detected_animal': 'Empty',
            'primary_label': 'Empty', 
            'species_label': 'N/A',
            'detection_confidence': 0.0,
            'bbox': None,
            'method': 'MDv5a',
            'secondary_method': None
        }
        
        # Step 1: Get all candidates
        candidates = self.megadetector.detect_all_candidates(image_path)
        
        if not candidates:
            return [base_result]
            
        final_results = []
        
        # Optimization: Read image once for cropping
        img_cv2 = None
        
        for cand in candidates:
            label = cand['label']
            conf = cand['conf']
            bbox = cand['bbox']
            
            result = base_result.copy()
            result['primary_label'] = label
            result['detected_animal'] = label
            result['detection_confidence'] = conf
            result['bbox'] = bbox
            
            # Step 2: Filter non-animals (Vehicles, Persons) 
            # We still return them, but don't run BioClip
            if label != 'Animal':
                final_results.append(result)
                continue
                
            # Step 3 & 4 for Animals
            try:
                if img_cv2 is None:
                    img_cv2 = cv2.imread(image_path)
                
                if img_cv2 is None:
                    final_results.append(result)
                    continue

                h, w = img_cv2.shape[:2]
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
                
                crop = img_cv2[y1:y2, x1:x2]
                
                if crop.size == 0:
                    final_results.append(result)
                    continue
                
                # Convert BGR (OpenCV) to RGB (PIL) for BioClip
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)
                
                # Step 4: BioClip
                # Use the same threshold as detection, or the configured one
                threshold = self.megadetector.confidence_threshold if self.megadetector else 0.1
                candidates = self.bioclip.predict_list(crop_pil, threshold=threshold)
                
                if candidates:
                    # Format: "Lion 0.95, Tiger 0.40"
                    species_label = ", ".join([f"{s.title()} {c:.2f}" for s, c in candidates])
                    species_data = [{'species': s.title(), 'confidence': float(c)} for s, c in candidates]
                    top_species, top_conf = candidates[0]
                else:
                    # Fallback if nothing above threshold but it was an animal
                    top_candidates = self.bioclip.predict_list(crop_pil, threshold=0.0, top_k=1)
                    if top_candidates:
                         top_species, top_conf = top_candidates[0]
                         species_label = f"{top_species.title()} {top_conf:.2f} (Low Conf)"
                         species_data = [{'species': top_species.title(), 'confidence': float(top_conf)}]
                    else:
                         top_species = "Unknown"
                         top_conf = 0.0
                         species_label = "Unknown"
                         species_data = []
                
                # Update result
                result['primary_label'] = 'Animal'
                result['species_label'] = species_label
                result['species_data'] = species_data
                result['detected_animal'] = top_species.title()
                result['detection_confidence'] = top_conf 
                result['method'] = 'MDv5a + BioClip'
                result['secondary_method'] = 'BioClip'
                
                final_results.append(result)
                
            except Exception as e:
                print(f"Error in BioClip step for {label}: {e}")
                final_results.append(result)
                
        return final_results if final_results else [base_result]

# Compatibility alias
EnsembleDetector = AnimalDetector
