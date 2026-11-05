#!/usr/bin/env python3
"""
Train YOLO Stage 1 Cervical Spine ROI Detector.

Trains an Ultralytics YOLO model on dataset/yolo/cvm_roi.yaml to predict
the bounding box of cervical vertebrae C2-C4.
"""

import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


def train_yolo_stage1(
    data_yaml: str = "dataset/yolo/cvm_roi.yaml",
    model_name: str = "yolo26s.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "",
    project: str = "runs/yolo_stage1",
    name: str = "exp",
    output_weights: str = "artifacts/yolo_stage1.pt",
) -> str:
    """
    Trains Ultralytics YOLO model and copies best checkpoint to output_weights.
    """
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"Dataset config not found at '{data_yaml}'. Please run src/data/preprocessing/generate_yolo_labels.py first."
        )

    print(f"🚀 Initializing YOLO model: {model_name}")
    model = YOLO(model_name)

    os.makedirs(os.path.dirname(output_weights) or ".", exist_ok=True)

    print(f"🏋️ Starting YOLO training for {epochs} epochs on {data_yaml}...")
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "project": project,
        "name": name,
        "exist_ok": True,
        "verbose": True,
    }
    if device:
        train_args["device"] = device

    results = model.train(**train_args)

    best_pt = Path(project) / name / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, output_weights)
        print(f"✅ Saved best YOLO weights to: {output_weights}")
    else:
        print(f"⚠️ Warning: Best weights not found at {best_pt}")

    return output_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 1 YOLO Cervical Spine ROI Detector")
    parser.add_argument("--data-yaml", default="dataset/yolo/cvm_roi.yaml", help="Path to cvm_roi.yaml")
    parser.add_argument("--model", default="yolo26s.pt", help="Pretrained YOLO base model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", default="", help="Device (e.g. 0, cpu, mps)")
    parser.add_argument("--output-weights", default="artifacts/yolo_stage1.pt", help="Path to save best weights")

    args = parser.parse_args()

    train_yolo_stage1(
        data_yaml=args.data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        output_weights=args.output_weights,
    )
