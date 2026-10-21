#!/usr/bin/env python3
"""
Landmark Detection & Inference CLI for CephalometricSwinGCN.

Performs neural landmark detection on lateral cephalometric radiographs:
- Predicts 13 cervical vertebral landmarks (C2, C3, C4).
- Supports single image or batch processing on entire folders.
- Automatically leverages MPS (Apple Silicon GPU), CUDA, or CPU.
- Seamlessly integrates with symcvm's CVM geometric calculator for clinical staging (CS1-CS6).
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure project root is first in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Also add symcvm to sys.path if present (appended, not prepended, so local src takes precedence)
SYMCVM_DIR = ROOT_DIR.parent / "symcvm"
if SYMCVM_DIR.is_dir() and str(SYMCVM_DIR) not in sys.path:
    sys.path.append(str(SYMCVM_DIR))

from src.data.constants import LANDMARK_CLASSES, NUM_LANDMARKS
from src.models.model import CephalometricSwinGCN

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def natural_sort_key(p: Path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", p.name)]


def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_model(weights_path: str, device: torch.device, img_size: int = 640) -> CephalometricSwinGCN:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}")

    model = CephalometricSwinGCN(num_landmarks=NUM_LANDMARKS, pretrained=False, img_size=img_size)
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        model = checkpoint
        model.to(device)
        model.eval()
        return model

    cleaned = {}
    for k, v in state_dict.items():
        if k.endswith("grid_x") or k.endswith("grid_y"):
            continue
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess_image(pil_img: Image.Image, img_size: int = 640) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    orig_w, orig_h = pil_img.size
    scale = img_size / max(orig_w, orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    pad_left = (img_size - new_w) // 2
    pad_top = (img_size - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float()

    return tensor, (orig_w, orig_h, pad_left, pad_top)


def predict_landmarks_single(
    model: CephalometricSwinGCN,
    pil_img: Image.Image,
    device: torch.device,
    img_size: int = 640,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    tensor, (orig_w, orig_h, pad_left, pad_top) = preprocess_image(pil_img, img_size)
    tensor = tensor.to(device)

    with torch.no_grad():
        _, pred_coords = model(tensor)  # [1, 13, 2] in normalized [0, 1] canvas space

    coords_canvas = pred_coords.squeeze(0).cpu().numpy()  # [13, 2]
    coords_canvas_px = coords_canvas * img_size

    scale = img_size / max(orig_w, orig_h)
    pixel_coords = []
    norm_coords = []

    for x_c, y_c in coords_canvas_px:
        orig_x = max(0.0, min(float(orig_w), (x_c - pad_left) / scale))
        orig_y = max(0.0, min(float(orig_h), (y_c - pad_top) / scale))
        pixel_coords.append((orig_x, orig_y))
        norm_coords.append((orig_x / orig_w, orig_y / orig_h))

    return pixel_coords, norm_coords


def predict_landmarks_two_stage(
    stage1_model: CephalometricSwinGCN,
    stage2_model: CephalometricSwinGCN,
    pil_img: Image.Image,
    device: torch.device,
    stage1_img_size: int = 640,
    stage2_img_size: int = 448,
    margin_ratio: float = 0.18,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    from src.utils.roi_utils import (
        compute_roi_bbox,
        crop_roi_with_padding,
        inverse_transform_coords,
    )

    tensor_s1, (orig_w, orig_h, pad_left, pad_top) = preprocess_image(pil_img, stage1_img_size)
    tensor_s1 = tensor_s1.to(device)

    # 1. Stage 1 Forward Pass
    with torch.no_grad():
        _, pred_coords_s1 = stage1_model(tensor_s1)

    s1_canvas_norm = pred_coords_s1.squeeze(0).cpu().numpy()  # [13, 2] in [0, 1]

    # 2. Extract strictly square ROI bounding box
    bbox = compute_roi_bbox(
        coords=s1_canvas_norm,
        img_w=stage1_img_size,
        img_h=stage1_img_size,
        margin_ratio=margin_ratio,
    )

    # Reconstruct 640x640 canvas for cropping
    scale = stage1_img_size / max(orig_w, orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    resized_canvas = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (stage1_img_size, stage1_img_size), (0, 0, 0))
    canvas.paste(resized_canvas, (pad_left, pad_top))
    canvas_np = np.array(canvas, dtype=np.uint8)

    # 3. Crop ROI with boundary padding
    roi_crop_np, _ = crop_roi_with_padding(
        image=canvas_np,
        bbox=bbox,
        target_size=stage2_img_size,
        interpolation=cv2.INTER_LINEAR,
    )

    # Preprocess ROI crop for Stage 2
    arr = roi_crop_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    tensor_s2 = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 4. Stage 2 Forward Pass
    with torch.no_grad():
        _, pred_coords_s2 = stage2_model(tensor_s2)

    s2_roi_norm = pred_coords_s2.squeeze(0).cpu().numpy()

    # 5. Inversely map coordinates back to canvas space with half-pixel alignment
    if bbox.is_fallback:
        final_canvas_norm = s1_canvas_norm
    else:
        final_canvas_norm = inverse_transform_coords(
            roi_coords=s2_roi_norm,
            bbox=bbox,
            target_size=stage2_img_size,
            output_normalized=True,
        )

    # 6. Un-letterbox to native pixel space
    pixel_coords = []
    norm_coords = []
    for x_c_norm, y_c_norm in final_canvas_norm:
        x_c_px = x_c_norm * stage1_img_size
        y_c_px = y_c_norm * stage1_img_size
        orig_x = max(0.0, min(float(orig_w), (x_c_px - pad_left) / scale))
        orig_y = max(0.0, min(float(orig_h), (y_c_px - pad_top) / scale))
        pixel_coords.append((orig_x, orig_y))
        norm_coords.append((orig_x / orig_w, orig_y / orig_h))

    return pixel_coords, norm_coords


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cephalometric Swin-GCN Landmark & CVM Inference CLI")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to a single X-ray image")
    parser.add_argument("--images-dir", "-d", type=str, default=None, help="Path to directory of X-ray images")
    parser.add_argument("--output-dir", "-o", type=str, default="inference_results", help="Directory for saving outputs")
    parser.add_argument(
        "--weights", "-w", type=str, default="artifacts/best.pth", help="Path to Stage 1 (or standalone) weights checkpoint"
    )
    parser.add_argument(
        "--stage2-weights", type=str, default=None, help="Optional Stage 2 fine refiner weights for Two-Stage inference"
    )
    parser.add_argument("--stage2-size", type=int, default=448, help="Stage 2 ROI resolution (default: 448)")
    parser.add_argument("--margin-ratio", type=float, default=0.18, help="ROI margin ratio (default: 0.18)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--compute-cvm", action="store_true", default=True, help="Compute clinical CVM stage (CS1-6)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images processed")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.image and not args.images_dir:
        parser.print_help()
        sys.exit(1)

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = args.weights
    if not os.path.exists(weights_path):
        # Fallback to symcvm weights if available
        alt_weights = Path(ROOT_DIR).parent / "symcvm" / "model" / "weights.pth"
        if alt_weights.exists():
            weights_path = str(alt_weights)
        else:
            print(f"❌ Weights not found at '{args.weights}'", file=sys.stderr)
            sys.exit(1)

    print(f"🧠 Loading Stage 1 CephalometricSwinGCN on {device}...")
    model = load_model(weights_path, device)
    print(f"✅ Stage 1 model successfully loaded from '{weights_path}'.")

    stage2_model = None
    stage2_weights_cand = args.stage2_weights or ("artifacts/stage2_best.pth" if os.path.exists("artifacts/stage2_best.pth") else None)
    if stage2_weights_cand:
        if os.path.exists(stage2_weights_cand):
            print(f"🧠 Loading Stage 2 Fine CephalometricSwinGCN from '{stage2_weights_cand}'...")
            stage2_model = load_model(stage2_weights_cand, device, img_size=args.stage2_size)
            print(f"✅ Stage 2 model successfully loaded ({args.stage2_size}x{args.stage2_size}).")
        else:
            print(f"⚠️ Warning: Stage 2 weights not found at '{stage2_weights_cand}'; falling back to single-stage.")

    # Gather images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        in_dir = Path(args.images_dir)
        image_paths = [p for p in in_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS and not p.name.startswith(".")]
        image_paths.sort(key=natural_sort_key)

    if args.limit:
        image_paths = image_paths[:args.limit]

    print(f"📋 Processing {len(image_paths)} image(s)...")

    # Optional CVM calculator hook
    cvm_calc = None
    if args.compute_cvm:
        try:
            from src.utils.cvm_calculator import evaluate_landmarks_dict
            cvm_calc = lambda lm_dict: evaluate_landmarks_dict(lm_dict)
            print("🩺 Clinical CVM Calculator (CS1-CS6) activated.")
        except ImportError:
            print("ℹ️ CVM calculator not available; skipping CVM staging.")

    results_json = {}
    csv_rows = []

    for p in image_paths:
        with Image.open(p) as img:
            pil_img = img.convert("RGB")

        if stage2_model is not None:
            px_coords, norm_coords = predict_landmarks_two_stage(
                stage1_model=model,
                stage2_model=stage2_model,
                pil_img=pil_img,
                device=device,
                stage1_img_size=640,
                stage2_img_size=args.stage2_size,
                margin_ratio=args.margin_ratio,
            )
        else:
            px_coords, norm_coords = predict_landmarks_single(model, pil_img, device)

        lm_dict = {
            name: {"x": px_coords[idx][0], "y": px_coords[idx][1]}
            for idx, name in enumerate(LANDMARK_CLASSES)
        }

        stage_str = "N/A"
        if cvm_calc:
            stage_res = cvm_calc(lm_dict)
            stage_str = stage_res["stage"]

        results_json[p.name] = {
            "stage": stage_str,
            "landmarks": {name: list(px_coords[idx]) for idx, name in enumerate(LANDMARK_CLASSES)},
            "normalized_landmarks": {name: list(norm_coords[idx]) for idx, name in enumerate(LANDMARK_CLASSES)},
        }

        csv_rows.append({"filename": p.name, "stage": stage_str, **{f"{k}_x": v[0] for k, v in zip(LANDMARK_CLASSES, px_coords)}, **{f"{k}_y": v[1] for k, v in zip(LANDMARK_CLASSES, px_coords)}})

    # Save outputs
    json_path = output_dir / "landmarks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    csv_path = output_dir / "landmarks.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\n🎉 Finished! Saved results to:\n - {json_path}\n - {csv_path}")


if __name__ == "__main__":
    main()
