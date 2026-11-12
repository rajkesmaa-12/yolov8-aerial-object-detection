"""Run inference on an image, folder or video."""
import argparse
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)
p.add_argument("--source", required=True)
p.add_argument("--conf", type=float, default=0.25)
a = p.parse_args()

YOLO(a.weights).predict(source=a.source, conf=a.conf, save=True)
