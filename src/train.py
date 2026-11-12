"""Train a YOLOv8 model on the aerial dataset."""
import argparse
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--model", default="yolov8n.pt")
p.add_argument("--data", default="data/data.yaml")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=16)
a = p.parse_args()

YOLO(a.model).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=42)
