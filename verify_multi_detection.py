
import os
import sys
import logging

# Add current dir to path
sys.path.append(os.getcwd())

from core.animal_detector import AnimalDetector, MegaDetectorWrapper
from core.image_processor import ImageProcessor
from core.bioclip_classifier import BioClipClassifier
from core.day_night_classifier import DayNightClassifier
from core.ocr_processor import OCRProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline():
    logger.info("Initializing models...")
    
    # Mock or simple init
    # We don't want to load full weights if possible, but for verification we might need to.
    # Let's assume MD is installed.
    
    try:
        md = MegaDetectorWrapper(confidence_threshold=0.1)
        bio = BioClipClassifier()
        ocr = OCRProcessor()
        dn = DayNightClassifier()
        
        detector = AnimalDetector(md, bio)
        processor = ImageProcessor(ocr, detector, dn)
        
        test_img = "test_md.jpg"
        if not os.path.exists(test_img):
            logger.error(f"{test_img} not found. Creating a dummy one.")
            import cv2
            import numpy as np
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.imwrite(test_img, dummy)
            
        logger.info("Testing AnimalDetector.detect()...")
        detections = detector.detect(test_img)
        logger.info(f"Detections type: {type(detections)}")
        logger.info(f"Detections content: {detections}")
        
        if not isinstance(detections, list):
            logger.error("FAILURE: detector.detect() must return a list")
            return
            
        logger.info("Testing ImageProcessor.process_single_image()...")
        results = processor.process_single_image(test_img)
        logger.info(f"Results type: {type(results)}")
        logger.info(f"Results content: {results}")
        
        if not isinstance(results, list):
            logger.error("FAILURE: processor.process_single_image() must return a list")
            return
            
        logger.info("SUCCESS: All return types correct.")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
