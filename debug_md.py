
import sys
import os
import megadetector
print(f"MegaDetector version: {getattr(megadetector, '__version__', 'unknown')}")
print(f"MegaDetector file: {megadetector.__file__}")

try:
    from megadetector.detection.run_detector import load_detector
    print("Imported load_detector successfully.")
except ImportError as e:
    print(f"Failed to import load_detector: {e}")

# Try to list available models if possible or check mappings
# Usually they are hardcoded in run_detector.py or a constants file.

try:
    print("Attempting to load MDV6b...")
    model = load_detector('MDV6b')
    print("SUCCESS: Loaded MDV6b")
except Exception as e:
    print(f"FAILED MDV6b: {e}")

try:
    print("Attempting to load MDV5a...")
    model = load_detector('MDV5a')
    print("SUCCESS: Loaded MDV5a")
except Exception as e:
    print(f"FAILED MDV5a: {e}")
    # Check sys.path
    print("sys.path:")
    for p in sys.path:
        print(f"  {p}")

# Output where megadetector thinks yolo is
import importlib.util
spec = importlib.util.find_spec("yolov5")
print(f"YOLOv5 spec: {spec}")
