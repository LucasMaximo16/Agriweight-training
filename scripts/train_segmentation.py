#!/usr/bin/env python3
"""
Fine-tunes YOLOv8s-seg on the cattle segmentation dataset produced by
prepare_dataset.py, initializing from the stock COCO checkpoint (transfer
learning) but training a single class ('boi') so the model isolates only the
animal — not the 79 other COCO classes.

Usage:
    python scripts/train_segmentation.py [--epochs 100] [--imgsz 640] [--batch 16]

Requires: pip install -r requirements.txt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = REPO_ROOT / "data" / "segmentation" / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base-weights", default="yolov8s-seg.pt")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--name", default="cattle_seg")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found — run scripts/prepare_dataset.py first")

    model = YOLO(args.base_weights)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    model.val(data=str(DATA_YAML), split="test")


if __name__ == "__main__":
    main()
