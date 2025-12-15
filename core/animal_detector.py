"""
Enhanced Animal Detection Module
Integrates MegaDetector V6 with MobileNetV2 for improved wildlife detection.
Supports ensemble voting for higher accuracy.
"""

import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
from typing import Tuple, Optional, Dict, List
from PIL import Image


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
        try:
            from megadetector.detection.run_detector import load_detector
            self.model = load_detector('MDV6b')
            print("MegaDetector V6b loaded successfully")
        except Exception as e:
            print(f"Error loading MDV6b: {str(e)}")
            try:
                print("Attempting to load MDV5a as fallback...")
                from megadetector.detection.run_detector import load_detector
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
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with detection results including bounding boxes
        """

        if self.model is None:
            return {
                'category': 'empty',
                'confidence': 0.0,
                'bbox': None
            }
        
        try:
            from megadetector.detection.run_detector import load_and_run_detector_batch
            
            # Run detection
            results = load_and_run_detector_batch(self.model, [image_path])
            
            if not results or len(results) == 0:
                return {
                    'category': 'empty',
                    'confidence': 0.0,
                    'bbox': None
                }
            
            # Get first result
            result = results[0]
            detections = result.get('detections', [])
            
            if not detections:
                return {
                    'category': 'empty',
                    'confidence': 0.0,
                    'bbox': None
                }
            
            # Debug: Print all detections
            print(f"MegaDetector raw detections for {image_path}:")
            for d in detections:
                print(f"  - Cat: {d.get('category')}, Conf: {d.get('conf')}, Box: {d.get('bbox')}")
            
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
            from megadetector.detection.run_detector import load_and_run_detector_batch
            results = load_and_run_detector_batch(self.model, [image_path])
            if results and len(results) > 0:
                return results[0].get('detections', [])
            return []
        except:
            return []


class AnimalDetector:
    """Handles animal detection using pre-trained models."""
    
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
        
        Args:
            confidence_threshold: Minimum confidence score to accept predictions (0-1)
        """

        self.confidence_threshold = confidence_threshold
        # Load pre-trained MobileNetV2 model
        try:
            self.model = MobileNetV2(weights='imagenet')
        except Exception as e:
            print(f"Error loading MobileNetV2: {e}")
            self.model = None

    def get_raw_classifications(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top K raw classifications from MobileNetV2."""
        if self.model is None:
            return []
            
        try:
            # Load and preprocess image
            img = keras_image.load_img(image_path, target_size=(224, 224))
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Predict
            preds = self.model.predict(x)
            decoded = decode_predictions(preds, top=top_k)[0]
            
            return [(label, float(score)) for (_, label, score) in decoded]
        except Exception as e:
            print(f"Error in raw classification: {e}")
            return []
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained MobileNetV2 model."""
        try:
            self.model = MobileNetV2(weights='imagenet', include_top=True)
            print("MobileNetV2 model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
    
    def preprocess_image(self, image_input, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image for model inference.
        """
        if isinstance(image_input, str):
            img = keras_image.load_img(image_input, target_size=target_size)
        else:
            # Convert numpy array to PIL Image and resize
            img = Image.fromarray(image_input)
            img = img.resize(target_size)
        
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
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
            # If bbox provided and image_input is a path, crop the image
            if bbox is not None and isinstance(image_input, str):
                img = cv2.imread(image_input)
                h, w = img.shape[:2]
                
                # Convert normalized bbox to pixel coordinates
                x, y, box_w, box_h = bbox
                x1 = int(x * w)
                y1 = int(y * h)
                x2 = int((x + box_w) * w)
                y2 = int((y + box_h) * h)
                
                # Crop image
                if x2 > x1 and y2 > y1:
                    cropped = img[y1:y2, x1:x2]
                    if cropped.size > 0:
                        # Convert BGR to RGB
                        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        processed_image = self.preprocess_image(cropped)
                    else:
                        processed_image = self.preprocess_image(image_input)
                else:
                    processed_image = self.preprocess_image(image_input)
            else:
                # Process full image
                processed_image = self.preprocess_image(image_input)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Decode predictions (top 5)
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            
            # Get top prediction
            top_class_id, top_class_name, top_confidence = decoded_predictions[0]
            
            # Check if it's wildlife and confidence is above threshold
            if self.is_wildlife(top_class_name) and top_confidence >= self.confidence_threshold:
                clean_name = top_class_name.replace('_', ' ').title()
                return (clean_name, float(top_confidence))
            else:
                # Check if any of the top 5 predictions are wildlife
                for class_id, class_name, confidence in decoded_predictions:
                    if self.is_wildlife(class_name) and confidence >= self.confidence_threshold:
                        clean_name = class_name.replace('_', ' ').title()
                        return (clean_name, float(confidence))
                
                # No wildlife detected with sufficient confidence
                return ("Unidentified", float(top_confidence))
                
        except Exception as e:
            print(f"Error detecting animal: {str(e)}")
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
                    # Use Fallback: Check MobileNetV2 on full image
                    # Only accept if HIGH confidence (> 0.4) to avoid noise
                    print(f"MegaDetector found nothing. Attempting fallback on {image_path}")
                    animal_name, mn_confidence = self.mobilenet.detect(image_path)
                    
                    # Higher threshold for fallback to reduce false positives
                    # Lowered to 0.2 to catch difficult cases (like the baboon sample)
                    fallback_threshold = max(0.2, self.confidence_threshold)
                    
                    if animal_name != 'Unidentified' and mn_confidence >= fallback_threshold:
                        result['detected_animal'] = animal_name
                        result['detection_confidence'] = mn_confidence
                        result['mobilenet_result'] = (animal_name, mn_confidence)
                        result['method'] = 'MobileNetV2 Fallback'
                        print(f"Fallback successful: {animal_name}")
        
        except Exception as e:
            print(f"Error in ensemble detection: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return result


def detect_animal(image_path: str, confidence_threshold: float = 0.3, mode: str = 'ensemble') -> Tuple[str, float, Optional[List[float]]]:
    """
    Convenience function to detect animal in an image.
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence score to accept predictions
        mode: Detection mode - 'ensemble', 'megadetector', or 'mobilenet'
        
    Returns:
        Tuple of (animal_name, confidence_score, bbox)
    """
    detector = EnsembleDetector(confidence_threshold=confidence_threshold, mode=mode)
    result = detector.detect(image_path)
    return (result['detected_animal'], result['detection_confidence'], result['bbox'])
