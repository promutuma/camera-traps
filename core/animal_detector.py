"""
Enhanced Animal Detection Module
Integrates MegaDetector V6 with MobileNetV2 (via PyTorch) for improved wildlife detection.
Supports ensemble voting for higher accuracy.
"""

import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, Optional, Dict, List

# Try to import MegaDetector. 
# Depending on the installed version, paths might vary.
try:
    from megadetector.detection.run_detector import load_detector
    MD_AVAILABLE = True
except ImportError:
    MD_AVAILABLE = False
    print("MegaDetector module not found in python path.")

class MegaDetectorWrapper:
    """Wrapper for MegaDetector V6 model."""
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize MegaDetector.
        
        Args:
            confidence_threshold: Minimum confidence score to accept detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.load_error = None
        self._load_model()
    
    def _load_model(self):
        """Load MegaDetector V6 model."""
        if not MD_AVAILABLE:
            self.load_error = "megadetector package not installed"
            return

        try:
            # Try loading MDV6b first
            self.model = load_detector('MDV6b')
            print("MegaDetector V6b loaded successfully")
        except Exception as e:
            print(f"Error loading MDV6b: {str(e)}")
            try:
                print("Attempting to load MDV5a as fallback...")
                self.model = load_detector('MDV5a')
                print("MegaDetector V5a loaded successfully")
            except Exception as e2:
                print(f"Error loading MDV5a: {str(e2)}")
                self.load_error = f"MDV6b failed ({str(e)}), MDV5a failed ({str(e2)})"
                self.model = None

    def get_status(self) -> Dict:
        """Get model loading status."""
        return {
            'loaded': self.model is not None,
            'error': self.load_error
        }

    def detect(self, image_path: str) -> Dict:
        """
        Detect animals in image using MegaDetector.
        """
        if self.model is None:
            return {
                'category': 'empty',
                'confidence': 0.0,
                'bbox': None
            }
        
        try:
            # Run detection directly
            img = Image.open(image_path)
            # generate_detections_one_image API depends on MD version
            # Assuming standard MD API
            result = self.model.generate_detections_one_image(img, image_path, detection_threshold=0.0)
            detections = result.get('detections', [])
            
            if not detections:
                return {
                    'category': 'empty',
                    'confidence': 0.0,
                    'bbox': None
                }
            
            # Debug: Print all detections
            # print(f"MegaDetector raw detections for {image_path}:")
            # for d in detections:
            #     print(f"  - Cat: {d.get('category')}, Conf: {d.get('conf')}, Box: {d.get('bbox')}")
            
            # Get highest confidence detection
            best_detection = max(detections, key=lambda x: x.get('conf', 0))
            confidence = best_detection.get('conf', 0.0)
            
            # MegaDetector categories: 1=animal, 2=person, 3=vehicle
            category_map = {
                '1': 'animal',
                '2': 'person',
                '3': 'vehicle'
            }
            
            raw_category = str(best_detection.get('category', '0'))
            category = category_map.get(raw_category, 'empty')
            bbox = best_detection.get('bbox', None)
            
            if confidence < self.confidence_threshold:
                category = 'empty'
            
            return {
                'category': category,
                'confidence': float(confidence),
                'bbox': bbox,
                'raw_category': raw_category
            }
            
        except Exception as e:
            print(f"Error in MegaDetector detection: {str(e)}")
            return {
                'category': 'empty',
                'confidence': 0.0,
                'bbox': None
            }

    def detect_all(self, image_path: str) -> List[Dict]:
        """Return all raw detections without filtering."""
        if self.model is None:
            return []
        try:
            img = Image.open(image_path)
            result = self.model.generate_detections_one_image(img, image_path, detection_threshold=0.0)
            return result.get('detections', [])
        except:
            return []


class AnimalDetector:
    """Handles animal detection using PyTorch MobileNetV2."""
    
    # Wildlife-related ImageNet classes (expanded for East Africa)
    WILDLIFE_CLASSES = {
        'zebra', 'elephant', 'lion', 'tiger', 'leopard', 'cheetah',
        'giraffe', 'buffalo', 'hyena', 'gazelle', 'impala', 'kudu',
        'warthog', 'baboon', 'monkey', 'gorilla', 'chimpanzee',
        'rhinoceros', 'hippopotamus', 'crocodile', 'snake',
        'eagle', 'vulture', 'ostrich', 'flamingo',
        'fox', 'wolf', 'coyote', 'bear', 'deer', 'antelope',
        'wild_boar', 'wildebeest', 'hartebeest', 'oryx', 'eland',
        'jackal', 'serval', 'mongoose', 'porcupine', 'pangolin'
    }
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize the animal detector.
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.categories = []
        self._load_model()
        
        # Define transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self):
        """Load the pre-trained MobileNetV2 model using Torchvision."""
        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            self.model = models.mobilenet_v2(weights=weights)
            self.model.eval()
            self.categories = weights.meta["categories"]
            print("MobileNetV2 (PyTorch) model loaded successfully")
        except Exception as e:
            print(f"Error loading MobileNetV2: {str(e)}")
            self.model = None
    
    def is_wildlife(self, class_name: str) -> bool:
        """Check if the predicted class is wildlife-related."""
        class_name_lower = class_name.lower()
        return any(wildlife in class_name_lower for wildlife in self.WILDLIFE_CLASSES)
    
    def detect(self, image_input, bbox: Optional[List[float]] = None) -> Tuple[str, float]:
        """
        Detect animal in the image.
        Returns Tuple of (animal_name, confidence_score)
        """
        if self.model is None:
            return ("Unidentified", 0.0)
        
        try:
            # Prepare PIL Image
            if isinstance(image_input, str):
                img = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                img = Image.fromarray(image_input).convert('RGB')
            else:
                img = image_input

            # Crop if bbox provided
            if bbox is not None:
                w, h = img.size
                x, y, box_w, box_h = bbox
                x1 = int(x * w)
                y1 = int(y * h)
                x2 = int((x + box_w) * w)
                y2 = int((y + box_h) * h)
                
                if x2 > x1 and y2 > y1:
                    img = img.crop((x1, y1, x2, y2))
            
            # Preprocess
            input_tensor = self.preprocess(img)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
            
            # Inference
            with torch.no_grad():
                output = self.model(input_batch)
            
            # Probabilities
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Top 5
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            # Check for wildlife
            for i in range(5):
                score = top5_prob[i].item()
                cat_id = top5_catid[i].item()
                class_name = self.categories[cat_id]
                
                if self.is_wildlife(class_name) and score >= self.confidence_threshold:
                    clean_name = class_name.replace('_', ' ').title()
                    return (clean_name, score)
            
            # If no wildlife found, return top deduction if high enough? 
            # Or just Unidentified. 
            # Original code said: if no wildlife, return Unidentified but showing score of top prediction.
            top_score = top5_prob[0].item()
            top_name = self.categories[top5_catid[0].item()]
            
            # If it's something like "cliff dwelling" or "jeep" we probably want to return it if detection mode is mobilenet only
            # The original logic was complex. Let's simplify:
            # Return "Unidentified" unless we are confident it's an animal.
            # But the caller might handle "Unidentified" by falling back.
            
            return ("Unidentified", top_score)
                
        except Exception as e:
            print(f"Error detecting animal (PyTorch): {str(e)}")
            import traceback
            traceback.print_exc()
            return ("Unidentified", 0.0)
            
    def swap_model(self, new_model):
        self.model = new_model
        print("Model swapped successfully")


class EnsembleDetector:
    """Ensemble detector combining MegaDetector and MobileNetV2."""
    
    def __init__(self, confidence_threshold: float = 0.3, mode: str = 'ensemble'):
        self.confidence_threshold = confidence_threshold
        self.mode = mode
        
        self.megadetector = None
        self.mobilenet = None
        
        # Initialize necessary detectors
        if mode in ['ensemble', 'megadetector']:
            self.megadetector = MegaDetectorWrapper(confidence_threshold)
        
        if mode in ['ensemble', 'mobilenet']:
            self.mobilenet = AnimalDetector(confidence_threshold)
            
        # Ensure MobileNet is available for fallback in ensemble mode
        if mode == 'ensemble' and self.mobilenet is None:
            self.mobilenet = AnimalDetector(confidence_threshold)
    
    def detect(self, image_path: str) -> Dict:
        """
        Detect and classify animals using ensemble approach.
        """
        result = {
            'detected_animal': 'Unidentified',
            'detection_confidence': 0.0,
            'bbox': None,
            'megadetector_result': None,
            'mobilenet_result': None,
            'method': 'Unknown'
        }
        
        try:
            if self.mode == 'mobilenet':
                # MobileNetV2 only
                animal_name, confidence = self.mobilenet.detect(image_path)
                result['detected_animal'] = animal_name
                result['detection_confidence'] = confidence
                result['mobilenet_result'] = (animal_name, confidence)
                result['method'] = 'MobileNetV2'
                
            elif self.mode == 'megadetector':
                # MegaDetector only
                md_result = self.megadetector.detect(image_path)
                result['bbox'] = md_result['bbox']
                result['megadetector_result'] = md_result
                result['method'] = 'MegaDetector'
                
                if md_result['category'] != 'empty':
                    result['detected_animal'] = md_result['category'].title()
                    result['detection_confidence'] = md_result['confidence']
                
            else:  # ensemble mode
                # Step 1: Run MegaDetector
                md_result = self.megadetector.detect(image_path)
                result['bbox'] = md_result['bbox']
                result['megadetector_result'] = md_result
                
                # Step 2: If animal detected, run MobileNetV2 on cropped region
                if md_result['category'] == 'animal' and md_result['bbox'] is not None:
                    animal_name, mn_confidence = self.mobilenet.detect(image_path, md_result['bbox'])
                    result['mobilenet_result'] = (animal_name, mn_confidence)
                    
                    # Step 3: Ensemble voting (weighted)
                    if animal_name != 'Unidentified':
                        ensemble_confidence = (md_result['confidence'] * 0.6) + (mn_confidence * 0.4)
                        result['detected_animal'] = animal_name
                        result['detection_confidence'] = ensemble_confidence
                        result['method'] = 'Ensemble (Strong)'
                    else:
                        result['detected_animal'] = 'Animal'
                        result['detection_confidence'] = md_result['confidence']
                        result['method'] = 'MegaDetector Only'
                
                elif md_result['category'] in ['person', 'vehicle']:
                    result['detected_animal'] = md_result['category'].title()
                    result['detection_confidence'] = md_result['confidence']
                    result['method'] = 'MegaDetector'
                
                else:
                    # MegaDetector found nothing (or confidence too low)
                    # Fallback logic
                    # print(f"MegaDetector found nothing. Attempting fallback.")
                    animal_name, mn_confidence = self.mobilenet.detect(image_path)
                    
                    fallback_threshold = max(0.2, self.confidence_threshold)
                    
                    if animal_name != 'Unidentified' and mn_confidence >= fallback_threshold:
                        result['detected_animal'] = animal_name
                        result['detection_confidence'] = mn_confidence
                        result['mobilenet_result'] = (animal_name, mn_confidence)
                        result['method'] = 'MobileNetV2 Fallback'
                        # print(f"Fallback successful: {animal_name}")
        
        except Exception as e:
            print(f"Error in ensemble detection: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return result


def detect_animal(image_path: str, confidence_threshold: float = 0.3, mode: str = 'ensemble') -> Tuple[str, float, Optional[List[float]]]:
    """
    Convenience function to detect animal in an image.
    """
    detector = EnsembleDetector(confidence_threshold=confidence_threshold, mode=mode)
    result = detector.detect(image_path)
    return (result['detected_animal'], result['detection_confidence'], result['bbox'])
