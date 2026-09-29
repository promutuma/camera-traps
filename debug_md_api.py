
from megadetector.detection import run_detector
from PIL import Image
import os

print("Imports successful.")

try:
    # Load default model (MDv5a)
    print("Loading model...")
    model = run_detector.load_detector("MDV5a")
    print(f"Model loaded: {type(model)}")
    print(f"Attributes: {dir(model)}")
    
    # Create dummy image
    img = Image.new('RGB', (640, 640), color='green')
    
    # Run detection
    print("Running detection...")
    result = model.generate_detections_one_image(img)
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
