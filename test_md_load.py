
import sys
import os
try:
    from megadetector.detection.run_detector import load_detector
    print("Attempting to load MDV5a...")
    model = load_detector('MDV5a')
    print("SUCCESS: Loaded MDV5a")
except Exception as e:
    print(f"FAILED MDV5a: {e}")
    # Print sys.path and checking utils
    import sys
    print("sys.path:", sys.path)
    try:
        import utils
        print("Imported utils from:", utils.__file__)
        try:
            import utils.general
            print("Imported utils.general from:", utils.general.__file__)
        except ImportError:
            print("Could not import utils.general")
    except ImportError:
        print("Could not import utils")
