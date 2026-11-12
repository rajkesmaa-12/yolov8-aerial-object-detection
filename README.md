# Low-Altitude Aerial Object Detection with YOLOv8

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-111F68) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

Detecting **humans, bicycles and vehicles** in low-altitude drone imagery with YOLOv8.
Project for the *Machine Vision* course, University of Passau (2026).

## Highlights
- Custom dataset: **1,013 drone images, 2,044 bounding-box annotations**, 3 classes (Human, Bicycle, Vehicle)
- Full pipeline: collection → annotation in Roboflow → cleaning → training → evaluation
- Reproducible training and inference scripts

## Results
<!-- TODO: replace with your real numbers and add a sample prediction image to results/ -->
| Model | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| YOLOv8n | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## Project structure
```
├── data/data.yaml        # dataset config (class names)
├── src/train.py          # training
├── src/evaluate.py       # validation metrics
├── src/predict.py        # inference on images/videos
├── results/              # plots, sample predictions
└── requirements.txt
```

## Quick start
```bash
pip install -r requirements.txt
# put the Roboflow export (YOLOv8 format) into data/  → data/train, data/valid, data/test
python src/train.py --model yolov8n.pt --epochs 100
python src/evaluate.py --weights runs/detect/train/weights/best.pt
python src/predict.py --weights runs/detect/train/weights/best.pt --source path/to/image.jpg
```

## Tech
Python · Ultralytics YOLOv8 · Roboflow · PyTorch
