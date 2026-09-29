import sys
sys.path.append("/home/mutuma/Code/camera-traps")
from core.animal_detector import MegaDetectorWrapper

md = MegaDetectorWrapper(model_version="MDv5a")
res = md.detect_all_candidates("test_md.jpg")
print("MDv5a result:")
print(res)

md2 = MegaDetectorWrapper(model_version="MD1000-redwood")
res2 = md2.detect_all_candidates("test_md.jpg")
print("MD1000 result:")
print(res2)
