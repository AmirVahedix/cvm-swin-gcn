#!/usr/bin/env python3
"""
Batch Inference Script for Cephalometric Swin-GCN Landmark Detection.

Performs neural landmark detection on lateral cephalometric radiographs:
1. Loads CephalometricSwinGCN model weights.
2. Processes a directory of unlabelled X-ray images (PNG, JPG, TIFF, etc.).
3. Runs CUDA-accelerated batch inference (with optional TTA).
4. Un-pads coordinates back to original image resolution.
5. Saves a JSON file with image IDs (filename without extension) as top-level keys.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Automatically re-exec with .venv if torch is missing
try:
    import torch
except ImportError:
    script_dir = Path(__file__).resolve().parent
    venv_candidates = [
        script_dir.parent / ".venv" / "bin" / "python",
        script_dir / ".venv" / "bin" / "python",
        Path.cwd() / ".venv" / "bin" / "python",
    ]
    for venv_py in venv_candidates:
        if venv_py.exists():
            os.execv(str(venv_py), [str(venv_py)] + sys.argv)
    import torch  # If neither exists, let standard ImportError raise

import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", unit="", **kwargs):
        items = list(iterable)
        total = len(items)
        for idx, item in enumerate(items, 1):
            if idx % 20 == 0 or idx == total:
                print(f"--> {desc}: {idx}/{total} {unit}")
            yield item

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.constants import LANDMARK_CLASSES, NUM_LANDMARKS
from src.models.model import CephalometricSwinGCN

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def natural_sort_key(p: Path):
    """Sort filenames in natural human order: 1.png, 2.png, ..., 10.png."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", p.name)]


def load_model(weights_path: str, device: torch.device, img_size: int = 640) -> torch.nn.Module:
    """Loads CephalometricSwinGCN model and cleans checkpoint keys safely."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: '{weights_path}'")

    print(f"🧠 Loading CephalometricSwinGCN weights from: {weights_path}")
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

    # Strip compile and parallel prefixes, ignore coordinate buffers
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.endswith("grid_x") or k.endswith("grid_y"):
            continue
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    matched = len([k for k in cleaned_state_dict.keys() if k not in unexpected_keys])
    print(f"✅ Loaded {matched}/{len(model.state_dict())} parameter tensors successfully.")

    model.to(device)
    model.eval()
    return model


def preprocess_image(
    image_path: Path,
    target_size: int = 640,
) -> Tuple[torch.Tensor, Tuple[int, int], float, int, int]:
    """
    Reads image, applies letterbox padding to preserve aspect ratio,
    and applies standard ImageNet normalization.

    Returns:
        tensor: [3, target_size, target_size] float32 tensor
        (orig_w, orig_h): Original image dimensions
        scale: Scale factor applied
        pad_x: Left padding on canvas
        pad_y: Top padding on canvas
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image file: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    # ImageNet normalization
    arr = canvas.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std

    tensor = torch.from_numpy(norm).permute(2, 0, 1).float()
    return tensor, (orig_w, orig_h), scale, pad_x, pad_y


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Inference CLI: Cephalometric Swin-GCN Landmark Detection"
    )
    parser.add_argument(
        "--images-dir",
        "-i",
        type=str,
        required=True,
        help="Path to folder containing cephalometric images (e.g. /workspace/500_images)",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        type=str,
        default="predictions.json",
        help="Path to output JSON file (default: predictions.json)",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default="artifacts/best.pth",
        help="Path to trained model weights checkpoint (default: artifacts/best.pth)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu", "auto"],
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size for model inference (default: 8)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Model input image resolution (default: 640)",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        default=False,
        help="Also export a summary CSV alongside the JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N images (useful for quick verification)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Resolve Device
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device_str = args.device

    if device_str == "cuda" and not torch.cuda.is_available():
        print("⚠️ Warning: CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        device_str = "cpu"

    device = torch.device(device_str)

    # 2. Check input path
    images_dir = Path(args.images_dir).resolve()
    if not images_dir.is_dir():
        print(f"❌ Error: Images directory not found at '{images_dir}'", file=sys.stderr)
        sys.exit(1)

    # 3. Find and sort images
    image_paths = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS and not p.name.startswith(".")
    ]
    image_paths.sort(key=natural_sort_key)

    if not image_paths:
        print(f"❌ Error: No valid image files found in '{images_dir}'", file=sys.stderr)
        sys.exit(1)

    if args.limit and args.limit > 0:
        image_paths = image_paths[:args.limit]

    output_json_path = Path(args.output_json).resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      CEPHALOMETRIC SWIN-GCN BATCH LANDMARK INFERENCE PIPELINE       ")
    print("=" * 80)
    print(f"📁 Input Directory:   {images_dir}")
    print(f"💾 Output JSON:       {output_json_path}")
    print(f"🖼️ Images Found:      {len(image_paths)}")
    print(f"⚡ Device:            {device} {f'({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''}")
    print(f"📦 Batch Size:        {args.batch_size}")
    print(f"⚖️ Model Weights:     {args.weights}")
    print("-" * 80)

    # 4. Load Model
    start_load = time.time()
    model = load_model(args.weights, device=device, img_size=args.img_size)
    print(f"Model ready in {time.time() - start_load:.2f}s.\n")

    # 5. Process in batches
    results_dict: Dict[str, Any] = {}
    failed_images: List[str] = []
    total_images = len(image_paths)

    start_infer = time.time()
    batch_size = max(1, args.batch_size)

    with torch.no_grad():
        for start_idx in tqdm(range(0, total_images, batch_size), desc="Inferring Batches", unit="batch"):
            batch_slice = image_paths[start_idx : start_idx + batch_size]
            batch_tensors = []
            batch_meta = []

            for img_p in batch_slice:
                try:
                    tensor, (orig_w, orig_h), scale, pad_x, pad_y = preprocess_image(
                        img_p, target_size=args.img_size
                    )
                    batch_tensors.append(tensor)
                    batch_meta.append((img_p, orig_w, orig_h, scale, pad_x, pad_y))
                except Exception as err:
                    print(f"\n⚠️ Skipping corrupted image '{img_p.name}': {err}", file=sys.stderr)
                    failed_images.append(img_p.name)

            if not batch_tensors:
                continue

            input_batch = torch.stack(batch_tensors, dim=0).to(device)

            # Model forward pass: outputs (heatmaps, normalized_coords)
            # coords shape: [B, 13, 2] in normalized canvas coordinates [0.0, 1.0]
            _, pred_coords = model(input_batch)
            pred_coords = pred_coords.cpu().numpy()

            for b_i, (img_p, orig_w, orig_h, scale, pad_x, pad_y) in enumerate(batch_meta):
                # Preserves image name without extension as ID
                img_id = img_p.stem
                filename = img_p.name

                coords_canvas = pred_coords[b_i] * float(args.img_size)  # [13, 2] in canvas px

                landmarks_px: Dict[str, List[float]] = {}
                landmarks_norm: Dict[str, List[float]] = {}
                points_list: List[Dict[str, Any]] = []

                for lm_idx, lm_name in enumerate(LANDMARK_CLASSES):
                    cx = float(coords_canvas[lm_idx, 0])
                    cy = float(coords_canvas[lm_idx, 1])

                    # Un-pad and invert letterbox scale back to original image coordinate frame
                    orig_x = (cx - float(pad_x)) / float(scale)
                    orig_y = (cy - float(pad_y)) / float(scale)

                    # Clamp to image boundaries
                    orig_x = max(0.0, min(float(orig_w), orig_x))
                    orig_y = max(0.0, min(float(orig_h), orig_y))

                    norm_x = orig_x / float(orig_w) if orig_w > 0 else 0.0
                    norm_y = orig_y / float(orig_h) if orig_h > 0 else 0.0

                    px_rounded = [round(orig_x, 2), round(orig_y, 2)]
                    norm_rounded = [round(norm_x, 5), round(norm_y, 5)]

                    landmarks_px[lm_name] = px_rounded
                    landmarks_norm[lm_name] = norm_rounded

                    points_list.append({
                        "id": lm_idx,
                        "name": lm_name,
                        "x": px_rounded[0],
                        "y": px_rounded[1],
                        "x_norm": norm_rounded[0],
                        "y_norm": norm_rounded[1],
                    })

                results_dict[img_id] = {
                    "id": img_id,
                    "filename": filename,
                    "image_size": {"width": orig_w, "height": orig_h},
                    "landmarks": landmarks_px,
                    "normalized_landmarks": landmarks_norm,
                    "points": points_list,
                }

    elapsed = time.time() - start_infer
    fps = total_images / elapsed if elapsed > 0 else 0.0

    # 6. Save JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    # 7. Optional CSV
    if args.save_csv:
        import csv
        csv_path = output_json_path.with_suffix(".csv")
        csv_rows = []
        for img_id, data in results_dict.items():
            row = {
                "id": img_id,
                "filename": data["filename"],
                "width": data["image_size"]["width"],
                "height": data["image_size"]["height"],
            }
            for lm_name, coords in data["landmarks"].items():
                row[f"{lm_name}_x"] = coords[0]
                row[f"{lm_name}_y"] = coords[1]
            csv_rows.append(row)

        if csv_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"📊 Summary CSV saved to: {csv_path}")

    print("\n" + "=" * 80)
    print("                    BATCH INFERENCE COMPLETED                       ")
    print("=" * 80)
    print(f"⏱️ Total Time:         {elapsed:.2f} s ({fps:.2f} images/s)")
    print(f"✅ Successful:         {len(results_dict)} / {total_images}")
    if failed_images:
        print(f"❌ Failed:             {len(failed_images)}")
    print(f"💾 Output JSON:        {output_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
