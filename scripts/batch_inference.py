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
from typing import Any, Dict, List, Optional, Tuple, Union

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
from src.utils.roi_utils import (
    apply_anatomical_height_prior,
    compute_roi_bbox,
    crop_roi_with_padding,
    extract_subpixel_peaks,
    inverse_transform_coords,
    map_bbox_to_native,
)
from src.data.preprocessing.split_dataset import (
    load_fixed_id_set,
    load_label_studio_mapping,
)
from sklearn.model_selection import train_test_split

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
) -> Tuple[torch.Tensor, Tuple[int, int], float, int, int, np.ndarray]:
    """
    Reads image, applies letterbox padding to preserve aspect ratio,
    and applies standard ImageNet normalization.

    Returns:
        tensor: [3, target_size, target_size] float32 tensor
        (orig_w, orig_h): Original image dimensions
        scale: Scale factor applied
        pad_x: Left padding on canvas
        pad_y: Top padding on canvas
        canvas: [target_size, target_size, 3] uint8 canvas
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
    return tensor, (orig_w, orig_h), scale, pad_x, pad_y, canvas


def build_split_mapping(
    image_paths: List[Path],
    dataset_dir: Optional[Path] = None,
    test_ids_path: Optional[Path] = None,
    val_ids_path: Optional[Path] = None,
    export_json_path: Optional[Path] = None,
    seed: int = 42,
    val_ratio: float = 0.15,
) -> Dict[str, str]:
    """
    Builds a deterministic mapping from image stem/id to split ('train', 'val', 'test').
    Priority:
    1. Direct folder membership in dataset/<split>/images
    2. Fixed ID files (test_image_ids.json, val_image_ids.json)
    3. Label Studio / CVM export task mappings
    4. Deterministic ratio split fallback for remaining unassigned samples
    """
    split_map: Dict[str, str] = {}

    test_stems = set()
    val_stems = set()
    train_stems = set()

    if dataset_dir:
        d_path = Path(dataset_dir)
        for split_name, stem_set in [("test", test_stems), ("val", val_stems), ("train", train_stems)]:
            split_img_dir = d_path / split_name / "images"
            if split_img_dir.is_dir():
                for p in split_img_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                        stem_set.add(p.stem)

    fixed_test_ids = load_fixed_id_set(test_ids_path) if test_ids_path else set()
    fixed_val_ids = load_fixed_id_set(val_ids_path) if val_ids_path else set()

    ls_mapping = {}
    if export_json_path and Path(export_json_path).exists():
        try:
            ls_mapping = load_label_studio_mapping(export_json_path)
        except Exception:
            ls_mapping = {}

    unassigned = []

    for img_p in image_paths:
        stem = img_p.stem
        task_id = ls_mapping.get(stem)
        task_id_alt = int(stem) if stem.isdigit() else stem

        # Test check
        is_test = (
            stem in test_stems
            or (task_id is not None and task_id in fixed_test_ids)
            or (task_id_alt in fixed_test_ids)
            or (stem in fixed_test_ids)
        )
        if is_test:
            split_map[stem] = "test"
            continue

        # Val check
        is_val = (
            stem in val_stems
            or (task_id is not None and task_id in fixed_val_ids)
            or (task_id_alt in fixed_val_ids)
            or (stem in fixed_val_ids)
        )
        if is_val:
            split_map[stem] = "val"
            continue

        # Train check
        if stem in train_stems:
            split_map[stem] = "train"
            continue

        unassigned.append(stem)

    if unassigned:
        if len(unassigned) > 1 and val_ratio > 0:
            try:
                train_sub, val_sub = train_test_split(
                    unassigned, test_size=val_ratio, random_state=seed
                )
                for s in train_sub:
                    split_map[s] = "train"
                for s in val_sub:
                    split_map[s] = "val"
            except Exception:
                for s in unassigned:
                    split_map[s] = "train"
        else:
            for s in unassigned:
                split_map[s] = "train"

    return split_map


def export_lightweight_predictions(
    results_dict: Dict[str, Any],
    light_output_dir: Path,
) -> Dict[str, Any]:
    """
    Exports clean, ultra-lightweight JSON and CSV files for visualization in external environments.
    Generates:
      ├── individual/<img_id>.json
      ├── train/<img_id>.json
      ├── val/<img_id>.json
      ├── test/<img_id>.json
      ├── by_landmark/<landmark_name>.json
      ├── all_landmarks.json
      ├── train_landmarks.json
      ├── val_landmarks.json
      ├── test_landmarks.json
      └── predictions_light.csv
    """
    light_output_dir = Path(light_output_dir)
    individual_dir = light_output_dir / "individual"
    by_landmark_dir = light_output_dir / "by_landmark"

    individual_dir.mkdir(parents=True, exist_ok=True)
    by_landmark_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (light_output_dir / split).mkdir(parents=True, exist_ok=True)

    all_landmarks_compact: Dict[str, Any] = {}
    split_landmarks_compact: Dict[str, Dict[str, Any]] = {"train": {}, "val": {}, "test": {}}
    by_landmark_data: Dict[str, Dict[str, Any]] = {lm: {} for lm in LANDMARK_CLASSES}
    csv_rows: List[Dict[str, Any]] = []

    for img_id, item in results_dict.items():
        split = item.get("split", "train")
        landmarks = item.get("landmarks", {})
        norm_landmarks = item.get("normalized_landmarks", {})
        img_size = item.get("image_size", {})
        w = img_size.get("width", 0)
        h = img_size.get("height", 0)

        # 1. Individual lightweight JSON per image
        light_item = {
            "id": img_id,
            "filename": item.get("filename", f"{img_id}.png"),
            "split": split,
            "image_size": {"width": w, "height": h},
            "landmarks": landmarks,
            "normalized_landmarks": norm_landmarks,
        }

        indiv_path = individual_dir / f"{img_id}.json"
        with open(indiv_path, "w", encoding="utf-8") as f:
            json.dump(light_item, f, indent=2)

        split_indiv_path = light_output_dir / split / f"{img_id}.json"
        with open(split_indiv_path, "w", encoding="utf-8") as f:
            json.dump(light_item, f, indent=2)

        # 2. Aggregations
        all_landmarks_compact[img_id] = {
            "split": split,
            "image_size": [w, h],
            "landmarks": landmarks,
        }
        if split in split_landmarks_compact:
            split_landmarks_compact[split][img_id] = {
                "image_size": [w, h],
                "landmarks": landmarks,
            }

        # 3. By Landmark
        for lm_name, coords in landmarks.items():
            if lm_name in by_landmark_data:
                by_landmark_data[lm_name][img_id] = {
                    "split": split,
                    "x": coords[0],
                    "y": coords[1],
                }

        # 4. CSV rows
        for lm_name, coords in landmarks.items():
            norm_c = norm_landmarks.get(lm_name, [0.0, 0.0])
            csv_rows.append({
                "id": img_id,
                "split": split,
                "width": w,
                "height": h,
                "landmark": lm_name,
                "x": coords[0],
                "y": coords[1],
                "norm_x": norm_c[0],
                "norm_y": norm_c[1],
            })

    # Save aggregated files
    with open(light_output_dir / "all_landmarks.json", "w", encoding="utf-8") as f:
        json.dump(all_landmarks_compact, f, indent=2)

    for s_name, s_data in split_landmarks_compact.items():
        with open(light_output_dir / f"{s_name}_landmarks.json", "w", encoding="utf-8") as f:
            json.dump(s_data, f, indent=2)

    for lm_name, lm_dict in by_landmark_data.items():
        with open(by_landmark_dir / f"{lm_name}.json", "w", encoding="utf-8") as f:
            json.dump(lm_dict, f, indent=2)

    # Save summary CSV
    if csv_rows:
        import csv
        csv_file = light_output_dir / "predictions_light.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"✨ Lightweight visualization predictions successfully written to: {light_output_dir}")
    print(f"   • Individual files: {len(results_dict)} in {individual_dir}")
    print(f"   • By landmark files: {len(LANDMARK_CLASSES)} in {by_landmark_dir}")
    print(f"   • Combined JSON: {light_output_dir / 'all_landmarks.json'}")
    return {
        "individual_dir": str(individual_dir),
        "by_landmark_dir": str(by_landmark_dir),
        "all_landmarks_json": str(light_output_dir / "all_landmarks.json"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Inference CLI: Cephalometric Swin-GCN Landmark Detection"
    )
    parser.add_argument(
        "--images-dir",
        "-i",
        type=str,
        required=True,
        help="Path to folder containing cephalometric images (e.g. data/images)",
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
        default="auto",
        choices=["cuda", "mps", "cpu", "auto"],
        help="Compute device (default: auto)",
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
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Path to dataset directory containing train/val/test splits (default: dataset)",
    )
    parser.add_argument(
        "--test-ids-file",
        type=str,
        default="dataset/test_image_ids.json",
        help="Path to JSON file with fixed test image IDs (default: dataset/test_image_ids.json)",
    )
    parser.add_argument(
        "--val-ids-file",
        type=str,
        default="dataset/val_image_ids.json",
        help="Path to JSON file with fixed validation image IDs (default: dataset/val_image_ids.json)",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default="data/exports/export.json",
        help="Path to Label Studio or CVM export JSON mapping task IDs (default: data/exports/export.json)",
    )
    parser.add_argument(
        "--light-output-dir",
        type=str,
        default="evaluation/predictions_light",
        help="Output directory to store lightweight predicted landmark location files (default: evaluation/predictions_light)",
    )
    parser.add_argument(
        "--export-light",
        dest="export_light",
        action="store_true",
        default=True,
        help="Generate clean ultra-lightweight landmark location files for external visualization (default: True)",
    )
    parser.add_argument(
        "--no-export-light",
        dest="export_light",
        action="store_false",
        help="Disable generating lightweight landmark files",
    )
    parser.add_argument(
        "--compute-metrics",
        dest="compute_metrics",
        action="store_true",
        default=True,
        help="Compute full comprehensive metrics (metrics_val.json, metrics_train.json) on splits (default: True)",
    )
    parser.add_argument(
        "--no-compute-metrics",
        dest="compute_metrics",
        action="store_false",
        help="Disable computing full split metrics",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="data/labels",
        help="Directory containing ground truth .npz label files (default: data/labels)",
    )
    parser.add_argument(
        "--metrics-splits",
        type=str,
        default="val,train,test",
        help="Comma-separated splits to compute full metrics for (default: 'val,train,test')",
    )
    parser.add_argument(
        "--metrics-output-dir",
        type=str,
        default=None,
        help="Directory to save metrics_<split>.json files (default: parent of output-json or evaluation)",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="log_to_mlflow",
        action="store_false",
        default=True,
        help="Disable logging metrics to MLflow",
    )
    parser.add_argument(
        "--stage1-weights",
        type=str,
        default=None,
        help="Stage 1 coarse weights path (overrides --weights if given)",
    )
    parser.add_argument(
        "--stage2-weights",
        type=str,
        default=None,
        help="Stage 2 fine ROI weights path for Two-Stage cascaded batch inference",
    )
    parser.add_argument(
        "--stage2-size",
        type=int,
        default=448,
        help="Stage 2 ROI input resolution (default: 448)",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.18,
        help="Vertebra ROI margin ratio (default: 0.18)",
    )
    parser.add_argument(
        "--no-anatomical-prior",
        dest="use_anatomical_prior",
        action="store_false",
        default=True,
        help="Disable Anatomical Vertebra Height Prior safeguard",
    )
    return parser.parse_args()


def run_batch_inference(
    images_dir: str | Path,
    output_json: Optional[str | Path] = "predictions.json",
    weights: str | Path = "artifacts/best.pth",
    stage1_weights: Optional[Union[str, Path]] = None,
    stage2_weights: Optional[Union[str, Path]] = None,
    stage2_size: int = 448,
    margin_ratio: float = 0.18,
    bottom_margin_ratio: Optional[float] = 0.28,
    use_peaks: bool = True,
    use_anatomical_prior: bool = True,
    device: str = "auto",
    batch_size: int = 8,
    img_size: int = 640,
    save_csv: bool = False,
    limit: Optional[int] = None,
    dataset_dir: Optional[str | Path] = "dataset",
    test_ids_file: Optional[str | Path] = "dataset/test_image_ids.json",
    val_ids_file: Optional[str | Path] = "dataset/val_image_ids.json",
    export_json: Optional[str | Path] = "data/exports/export.json",
    light_output_dir: Optional[str | Path] = "evaluation/predictions_light",
    export_light: bool = True,
    compute_metrics: bool = True,
    labels_dir: Union[str, Path] = "data/labels",
    metrics_splits: Union[str, List[str]] = "val,train,test",
    metrics_output_dir: Optional[Union[str, Path]] = None,
    log_to_mlflow: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Executes Cephalometric Swin-GCN batch landmark inference over an image directory.
    Identifies train/val/test split for each image and exports lightweight landmark location files.
    """
    # 1. Resolve Device
    if device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device_str = device

    if device_str == "cuda" and not torch.cuda.is_available():
        print("⚠️ Warning: CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        device_str = "cpu"

    dev = torch.device(device_str)

    # 2. Check input path
    images_path = Path(images_dir).resolve()
    if not images_path.is_dir():
        raise FileNotFoundError(f"Images directory not found at '{images_path}'")

    # 3. Find and sort images
    image_paths = [
        p for p in images_path.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS and not p.name.startswith(".")
    ]
    image_paths.sort(key=natural_sort_key)

    if not image_paths:
        raise ValueError(f"No valid image files found in '{images_path}'")

    if limit and limit > 0:
        image_paths = image_paths[:limit]

    # 4. Resolve train/val/test splits
    split_map = build_split_mapping(
        image_paths=image_paths,
        dataset_dir=Path(dataset_dir) if dataset_dir else None,
        test_ids_path=Path(test_ids_file) if test_ids_file else None,
        val_ids_path=Path(val_ids_file) if val_ids_file else None,
        export_json_path=Path(export_json) if export_json else None,
    )

    counts = {"train": 0, "val": 0, "test": 0}
    for s in split_map.values():
        if s in counts:
            counts[s] += 1

    print("=" * 80)
    print("      CEPHALOMETRIC SWIN-GCN BATCH LANDMARK INFERENCE PIPELINE       ")
    print("=" * 80)
    print(f"📁 Input Directory:   {images_path}")
    print(f"🖼️ Images Found:      {len(image_paths)}")
    print(f"📊 Dataset Splits:    Train={counts['train']} | Val={counts['val']} | Test={counts['test']}")
    print(f"⚡ Device:            {dev} {f'({torch.cuda.get_device_name(0)})' if dev.type == 'cuda' else ''}")
    print(f"📦 Batch Size:        {batch_size}")
    print(f"⚖️ Model Weights:     {weights}")
    if output_json:
        print(f"💾 Output Full JSON:  {output_json}")
    if export_light and light_output_dir:
        print(f"✨ Lightweight Files: {light_output_dir}")
    print("-" * 80)

    # 5. Load Model(s)
    start_load = time.time()
    s1_weights_path = stage1_weights or weights
    model_s1 = load_model(str(s1_weights_path), device=dev, img_size=img_size)

    model_s2 = None
    if stage2_weights and os.path.exists(str(stage2_weights)):
        print(f"🔬 Loading Stage 2 Fine ROI model from: {stage2_weights}")
        model_s2 = load_model(str(stage2_weights), device=dev, img_size=stage2_size)
        print(f"--> Two-Stage Cascaded batch inference active ({stage2_size}x{stage2_size} ROI).")

    print(f"Model ready in {time.time() - start_load:.2f}s.\n")

    # 6. Process in batches
    results_dict: Dict[str, Any] = {}
    failed_images: List[str] = []
    total_images = len(image_paths)

    start_infer = time.time()
    bs = max(1, batch_size)

    with torch.no_grad():
        for start_idx in tqdm(range(0, total_images, bs), desc="Inferring Batches", unit="batch"):
            batch_slice = image_paths[start_idx : start_idx + bs]
            batch_tensors = []
            batch_canvases = []
            batch_meta = []

            for img_p in batch_slice:
                try:
                    tensor, (orig_w, orig_h), scale, pad_x, pad_y, canvas = preprocess_image(
                        img_p, target_size=img_size
                    )
                    batch_tensors.append(tensor)
                    batch_canvases.append(canvas)
                    batch_meta.append((img_p, orig_w, orig_h, scale, pad_x, pad_y))
                except Exception as err:
                    print(f"\n⚠️ Skipping corrupted image '{img_p.name}': {err}", file=sys.stderr)
                    failed_images.append(img_p.name)

            if not batch_tensors:
                continue

            input_batch = torch.stack(batch_tensors, dim=0).to(dev)

            # Model forward pass: outputs (heatmaps, normalized_coords)
            # coords shape: [B, 13, 2] in normalized canvas coordinates [0.0, 1.0]
            pred_hm_s1, pred_coords_s1 = model_s1(input_batch)
            if use_peaks:
                pred_coords_s1 = extract_subpixel_peaks(pred_hm_s1)
            pred_coords_s1_np = pred_coords_s1.cpu().numpy()

            if model_s2 is not None:
                final_coords_np = np.zeros_like(pred_coords_s1_np)
                for b_i in range(len(batch_meta)):
                    s1_coords = pred_coords_s1_np[b_i]
                    bbox = compute_roi_bbox(
                        coords=s1_coords,
                        img_w=img_size,
                        img_h=img_size,
                        margin_ratio=margin_ratio,
                        bottom_margin_ratio=bottom_margin_ratio,
                    )
                    if bbox.is_fallback:
                        final_coords_np[b_i] = s1_coords
                    else:
                        img_p_cur, orig_w_cur, orig_h_cur, _, _, _ = batch_meta[b_i]
                        # Use high-res native crop if available and higher resolution than canvas
                        if max(orig_w_cur, orig_h_cur) > img_size:
                            raw_bgr = cv2.imread(str(img_p_cur))
                            if raw_bgr is not None:
                                raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                                native_bbox = map_bbox_to_native(bbox, orig_w=orig_w_cur, orig_h=orig_h_cur, canvas_size=img_size)
                                roi_patch, _ = crop_roi_with_padding(
                                    image=raw_rgb,
                                    bbox=native_bbox,
                                    target_size=stage2_size,
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            else:
                                canvas = batch_canvases[b_i]
                                roi_patch, _ = crop_roi_with_padding(
                                    image=canvas,
                                    bbox=bbox,
                                    target_size=stage2_size,
                                    interpolation=cv2.INTER_LINEAR,
                                )
                        else:
                            canvas = batch_canvases[b_i]
                            roi_patch, _ = crop_roi_with_padding(
                                image=canvas,
                                bbox=bbox,
                                target_size=stage2_size,
                                interpolation=cv2.INTER_LINEAR,
                            )
                        arr = roi_patch.astype(np.float32) / 255.0
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        norm_roi = (arr - mean) / std
                        roi_in = torch.from_numpy(norm_roi).permute(2, 0, 1).unsqueeze(0).float().to(dev)

                        pred_hm_roi, pred_roi = model_s2(roi_in)
                        if use_peaks:
                            pred_roi = extract_subpixel_peaks(pred_hm_roi)
                        pred_roi_np = pred_roi.squeeze(0).cpu().numpy()
                        reprojected = inverse_transform_coords(
                            roi_coords=pred_roi_np,
                            bbox=bbox,
                            target_size=stage2_size,
                            output_normalized=True,
                        )
                        final_coords_np[b_i] = reprojected
                pred_coords = final_coords_np
            else:
                pred_coords = pred_coords_s1_np

            # Apply Anatomical Vertebra Height Prior safeguard
            if use_anatomical_prior:
                for b_i in range(len(pred_coords)):
                    pred_coords[b_i], _ = apply_anatomical_height_prior(
                        pred_coords[b_i], img_w=img_size, img_h=img_size
                    )

            for b_i, (img_p, orig_w, orig_h, scale, pad_x, pad_y) in enumerate(batch_meta):
                img_id = img_p.stem
                filename = img_p.name
                split_val = split_map.get(img_id, "train")

                coords_canvas = pred_coords[b_i] * float(img_size)  # [13, 2] in canvas px

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
                    "split": split_val,
                    "image_size": {"width": orig_w, "height": orig_h},
                    "landmarks": landmarks_px,
                    "normalized_landmarks": landmarks_norm,
                    "points": points_list,
                }

    elapsed = time.time() - start_infer
    fps = total_images / elapsed if elapsed > 0 else 0.0

    # 7. Save Output Full JSON
    if output_json:
        out_p = Path(output_json).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)

    # 8. Optional Summary CSV
    if save_csv and output_json:
        import csv
        csv_path = Path(output_json).resolve().with_suffix(".csv")
        csv_rows = []
        for img_id, data in results_dict.items():
            row = {
                "id": img_id,
                "filename": data["filename"],
                "split": data.get("split", "train"),
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

    # 9. Export clean lightweight visualization files
    if export_light and light_output_dir:
        export_lightweight_predictions(results_dict, light_output_dir=Path(light_output_dir))

    # 10. Compute full comprehensive metrics for val and train sets
    if compute_metrics:
        splits_list = (
            [s.strip() for s in metrics_splits.split(",") if s.strip()]
            if isinstance(metrics_splits, str)
            else metrics_splits
        )
        metrics_out = Path(
            metrics_output_dir
            or (Path(output_json).parent if output_json else "evaluation")
        )
        try:
            from src.eval import compute_and_log_split_metrics

            compute_and_log_split_metrics(
                results_dict=results_dict,
                labels_dir=labels_dir,
                output_dir=metrics_out,
                export_json_path=export_json,
                img_size=img_size,
                log_to_mlflow=log_to_mlflow,
                splits_to_eval=splits_list,
            )
        except Exception as m_err:
            print(f"⚠️ Warning: Could not compute split metrics: {m_err}")

    print("\n" + "=" * 80)
    print("                    BATCH INFERENCE COMPLETED                       ")
    print("=" * 80)
    print(f"⏱️ Total Time:         {elapsed:.2f} s ({fps:.2f} images/s)")
    print(f"✅ Successful:         {len(results_dict)} / {total_images}")
    if failed_images:
        print(f"❌ Failed:             {len(failed_images)}")
    if output_json:
        print(f"💾 Output Full JSON:   {output_json}")
    if export_light and light_output_dir:
        print(f"✨ Lightweight Files:  {light_output_dir}")
    print("=" * 80)

    return results_dict


def main():
    args = parse_args()
    run_batch_inference(**vars(args))
