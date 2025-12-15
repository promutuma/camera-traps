
import sys
print(f"Python Executable: {sys.executable}")
print(f"Python Path: {sys.path}")

try:
    import torch
    print(f"Torch Version: {torch.__version__}")
except ImportError as e:
    print(f"Torch Import Failed: {e}")

try:
    import PytorchWildlife
    print("PytorchWildlife imported successfully!")
    try:
        print(f"Version: {PytorchWildlife.__version__}")
    except:
        print("Version attribute missing")
        
    from PytorchWildlife.models import detection as pw_detection
    print("PytorchWildlife.models.detection imported.")
    
except ImportError as e:
    print(f"PytorchWildlife Import Error: {e}")
except Exception as e:
    print(f"PytorchWildlife General Error: {e}")
    import traceback
    traceback.print_exc()
