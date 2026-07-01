#!/usr/bin/env python3
"""
Exports a trained YOLOv8-seg checkpoint (best.pt from train_segmentation.py)
to Core ML, ready to bundle into the AgriWeight-app iOS project.

Usage:
    python scripts/export_coreml.py --weights runs/cattle_seg/weights/best.pt

Requires: pip install -r requirements.txt
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", default=True, help="Export Float16 (default on)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "models"))
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    exported_path = model.export(format="coreml", imgsz=args.imgsz, half=args.half, nms=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / Path(exported_path).name
    shutil.move(exported_path, destination)
    print(f"CoreML model written to {destination}")
    print("Copy this .mlpackage into ios/AgriWeight/ in AgriWeight-app and add it as a build resource.")


if __name__ == "__main__":
    main()
