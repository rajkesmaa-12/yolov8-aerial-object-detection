"""Report mAP / precision / recall on the validation split."""
import argparse
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)
p.add_argument("--data", default="data/data.yaml")
a = p.parse_args()

m = YOLO(a.weights).val(data=a.data)
print(f"mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  P={m.box.mp:.3f}  R={m.box.mr:.3f}")
