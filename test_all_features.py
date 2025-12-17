import os
import sys
import logging
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Add current dir to path
sys.path.append(os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    logger.info("--- Testing Imports ---")
    try:
        from core.animal_detector import AnimalDetector, MegaDetectorWrapper
        from core.image_processor import ImageProcessor
        from core.bioclip_classifier import BioClipClassifier
        from core.day_night_classifier import DayNightClassifier
        from core.ocr_processor import OCRProcessor
        logger.info("SUCCESS: All core modules imported.")
        return True
    except ImportError as e:
        logger.error(f"FAILURE: Import failed: {e}")
        return False

def test_ocr(ocr_processor):
    logger.info("--- Testing OCR ---")
    # Increase image size to 500x500
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    # Bottom 10% is y=450-500.
    # Text at y=480.
    cv2.putText(img, "2023-10-27 14:30:00 25C", (50, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite("test_ocr.jpg", img)
    
    try:
        metadata = ocr_processor.process_image("test_ocr.jpg")
        logger.info(f"OCR Output: {metadata}")
        if metadata.get('date') == '2023-10-27':
             logger.info("SUCCESS: OCR detected date.")
        elif "2023-10-27" in metadata.get('raw_text', ''):
             logger.info("SUCCESS: OCR detected date in raw text.")
        else:
            logger.error(f"FAILURE: OCR did not detect expected text. Got: {metadata}")
    except Exception as e:
        logger.error(f"FAILURE: OCR test crashed: {e}")
    finally:
        if os.path.exists("test_ocr.jpg"):
            os.remove("test_ocr.jpg")

def test_day_night(dn_classifier):
    logger.info("--- Testing Day/Night Classification ---")
    # Day image must be Bright AND Colored (to pass Night Vision check)
    # Cyan (255, 255, 0) BGR. High Saturation, High Brightness.
    day_img = np.zeros((100, 100, 3), dtype=np.uint8)
    day_img[:] = (255, 255, 0) 
    cv2.imwrite("test_day.jpg", day_img)
    
    # Night image (black)
    night_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite("test_night.jpg", night_img)
    
    try:
        is_day_day, _ = dn_classifier.classify("test_day.jpg")
        is_day_night, _ = dn_classifier.classify("test_night.jpg")
        
        if is_day_day == 'Day' and is_day_night == 'Night':
            logger.info("SUCCESS: Day/Night classification correct.")
        else:
            logger.error(f"FAILURE: Day/Night classification incorrect. Day->{is_day_day}, Night->{is_day_night}")
    except Exception as e:
        logger.error(f"FAILURE: Day/Night test crashed: {e}")
    finally:
        if os.path.exists("test_day.jpg"): os.remove("test_day.jpg")
        if os.path.exists("test_night.jpg"): os.remove("test_night.jpg")

def test_models_and_pipeline():
    logger.info("--- Testing Models & Full Pipeline ---")
    
    # Check for test image
    test_img_path = "test_md.jpg"
    if not os.path.exists(test_img_path):
        logger.warning(f"{test_img_path} not found. Creating a dummy image for pipeline structure test (detections will vary).")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        # Draw a "box" approx animal like
        cv2.rectangle(dummy, (100, 100), (300, 300), (100, 100, 100), -1)
        cv2.imwrite(test_img_path, dummy)

    try:
        from core.animal_detector import AnimalDetector, MegaDetectorWrapper
        from core.image_processor import ImageProcessor
        from core.bioclip_classifier import BioClipClassifier
        from core.day_night_classifier import DayNightClassifier
        from core.ocr_processor import OCRProcessor

        logger.info("Loading OCR...")
        ocr = OCRProcessor()
        
        logger.info("Loading Day/Night...")
        dn = DayNightClassifier()
        
        logger.info("Loading MegaDetector V5a...")
        md = MegaDetectorWrapper(confidence_threshold=0.1)
        
        if not md.get_status()['loaded']:
            logger.error(f"FAILURE: MegaDetector not loaded. Error: {md.get_status()['error']}")
            return

        logger.info("Loading BioClip...")
        bio = BioClipClassifier()
        
        logger.info("Initializing AnimalDetector...")
        detector = AnimalDetector(md, bio)
        
        logger.info("Initializing ImageProcessor...")
        processor = ImageProcessor(ocr, detector, dn)
        
        # Test 1: Detect
        logger.info("Running detection on test image...")
        detections = detector.detect(test_img_path)
        logger.info(f"Detections: {detections}")
        if isinstance(detections, list) and len(detections) > 0:
            logger.info("SUCCESS: Detection returned a list.")
        else:
            logger.error(f"FAILURE: Detection returned unexpected format: {type(detections)}")

        # Test 2: Full Process
        logger.info("Running full image processing...")
        results = processor.process_single_image(test_img_path)
        logger.info(f"Process Results: {results}")
        
        if isinstance(results, list) and len(results) > 0:
            logger.info("SUCCESS: Full processing returned a list.")
            first_res = results[0]
            required_keys = ['filename', 'detected_animal', 'primary_label', 'detection_confidence']
            missing = [k for k in required_keys if k not in first_res]
            if not missing:
                logger.info("SUCCESS: Result dictionary has required keys.")
            else:
                logger.error(f"FAILURE: Missing keys in result: {missing}")
        else:
             logger.error(f"FAILURE: Full processing returned unexpected format or empty list: {results}")

        # Run specific module tests with loaded instances
        test_ocr(ocr)
        test_day_night(dn)

    except Exception as e:
        logger.error(f"FAILURE: Pipeline test crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if test_imports():
        test_models_and_pipeline()
    else:
        logger.error("Skipping further tests due to import failure.")
