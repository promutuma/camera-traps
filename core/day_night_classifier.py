"""
Day/Night Classification Module
Classifies images as day or night based on brightness analysis.
"""

import cv2
import numpy as np
from typing import Tuple


class DayNightClassifier:
    """Classifies images as day or night based on pixel brightness."""
    
    def __init__(self, brightness_threshold: int = 100):
        """
        Initialize the classifier.
        
        Args:
            brightness_threshold: Threshold value (0-255) to distinguish day from night.
                                 Values above threshold are classified as day.
                                 Default is 100 (mid-range).
        """
        self.brightness_threshold = brightness_threshold
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        """
        Calculate the average brightness of an image.
        
        Args:
            image: Input image as numpy array (BGR or grayscale)
            
        Returns:
            Average brightness value (0-255)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate mean brightness
        mean_brightness = np.mean(gray)
        return float(mean_brightness)
    
    def detect_night_vision(self, image: np.ndarray) -> bool:
        """
        Detect if image is from night vision/infrared camera.
        Night vision images are typically grayscale or have very low color saturation.
        
        Args:
            image: Input image as numpy array (BGR)
            
        Returns:
            True if likely night vision, False otherwise
        """
        if len(image.shape) == 2:
            # Already grayscale
            return True
        
        # Convert to HSV to check saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # Calculate mean saturation
        mean_saturation = np.mean(saturation)
        
        # Night vision images have very low saturation (< 20 typically)
        return mean_saturation < 30
    
    def classify(self, image_path: str) -> Tuple[str, float]:
        """
        Classify an image as day or night.
        Enhanced to handle night vision/infrared images.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (classification, brightness_value)
            classification: "Day" or "Night"
            brightness_value: Average brightness (0-255)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Detect if night vision
            is_night_vision = self.detect_night_vision(image)
            
            # Calculate brightness
            brightness = self.calculate_brightness(image)
            
            # Adjust classification for night vision images
            if is_night_vision:
                # Night vision images are always "Night" regardless of brightness
                # They use infrared illumination which appears bright but is nighttime
                classification = "Night"
            else:
                # Normal color images - use brightness threshold
                classification = "Day" if brightness >= self.brightness_threshold else "Night"
            
            return (classification, brightness)
            
        except Exception as e:
            print(f"Error classifying image {image_path}: {str(e)}")
            return ("Unknown", 0.0)
    
    def classify_with_confidence(self, image_path: str) -> Tuple[str, float, float]:
        """
        Classify an image with a confidence score.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (classification, brightness_value, confidence)
            confidence: How far the brightness is from the threshold (0-1)
        """
        classification, brightness = self.classify(image_path)
        
        # Calculate confidence based on distance from threshold
        # Normalized to 0-1 range
        distance_from_threshold = abs(brightness - self.brightness_threshold)
        max_distance = 255 - self.brightness_threshold if brightness >= self.brightness_threshold else self.brightness_threshold
        confidence = min(distance_from_threshold / max_distance, 1.0) if max_distance > 0 else 0.5
        
        return (classification, brightness, confidence)


def classify_day_night(image_path: str, brightness_threshold: int = 100) -> Tuple[str, float]:
    """
    Convenience function to classify an image as day or night.
    
    Args:
        image_path: Path to the image file
        brightness_threshold: Threshold value to distinguish day from night
        
    Returns:
        Tuple of (classification, brightness_value)
    """
    classifier = DayNightClassifier(brightness_threshold=brightness_threshold)
    return classifier.classify(image_path)
