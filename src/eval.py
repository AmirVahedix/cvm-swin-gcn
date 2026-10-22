import os
import sys
import json
import argparse
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import math
import time
import requests
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import mlflow

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import LANDMARK_CLASSES, NUM_LANDMARKS, get_test_dataloader
from src.models.model import CephalometricSwinGCN
from src.utils.metrics import (
    compute_test_comprehensive_metrics,
    format_summary_report,
    compute_inferior_border_projections,
    compute_reference_length,
    C3_PI_IDX,
    C3_AI_IDX,
    VERTEBRA_INFERIOR_LANDMARKS,
)
from src.utils.cvm_calculator import evaluate_landmarks_dict


LANDMARK_COLORS_BGR = [
    (0, 0, 255),      # 0: C2_PI - Red
    (0, 140, 255),    # 1: C2_IC - Orange
    (0, 215, 255),    # 2: C2_AI - Yellow
    (0, 255, 128),    # 3: C3_PS - Spring Green
    (0, 255, 0),      # 4: C3_AS - Green
    (128, 255, 0),    # 5: C3_PI - Lime
    (255, 255, 0),    # 6: C3_IC - Cyan
    (255, 191, 0),    # 7: C3_AI - Azure
    (255, 0, 0),      # 8: C4_PS - Blue
    (255, 0, 128),    # 9: C4_AS - Violet
    (255, 0, 255),    # 10: C4_PI - Magenta
    (128, 0, 255),    # 11: C4_IC - Purple
    (255, 255, 255),  # 12: C4_AI - White
]


def draw_legend_box(vis_img: np.ndarray, num_landmarks: int):
    """
    Draws a clean, rectangular guide box in the top-right section of the image,
    displaying each landmark's distinct color alongside its corresponding label.
    """
    h, w = vis_img.shape[:2]
    total_landmarks = min(num_landmarks, len(LANDMARK_CLASSES))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness = 1
    row_height = 18
    padding_x = 10
    padding_y = 8
    dot_radius = 4

    # Calculate max text width for tight guide box sizing
    max_text_w = 0
    for i in range(total_landmarks):
        label_text = f"{i}:{LANDMARK_CLASSES[i]}"
        (text_w, _), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        if text_w > max_text_w:
            max_text_w = text_w

    box_w = padding_x * 2 + dot_radius * 2 + 8 + max_text_w
    box_h = padding_y * 2 + total_landmarks * row_height

    margin_right = 12
    margin_top = 12

    box_x2 = w - margin_right
    box_x1 = max(0, box_x2 - box_w)
    box_y1 = margin_top
    box_y2 = min(h, box_y1 + box_h)

    # Semi-transparent dark background for the guide box
    sub_img = vis_img[box_y1:box_y2, box_x1:box_x2]
    if sub_img.size > 0:
        dark_rect = np.zeros_like(sub_img, dtype=np.uint8)
        blended = cv2.addWeighted(sub_img, 0.2, dark_rect, 0.8, 0)
        vis_img[box_y1:box_y2, box_x1:box_x2] = blended

    # White outline border for the guide box
    cv2.rectangle(vis_img, (box_x1, box_y1), (box_x2, box_y2), (220, 220, 220), 1)

    # Render entries inside the guide box
    for i in range(total_landmarks):
        color_bgr = LANDMARK_COLORS_BGR[i % len(LANDMARK_COLORS_BGR)]
        label_text = f"{i}:{LANDMARK_CLASSES[i]}"

        center_y = box_y1 + padding_y + (i * row_height) + (row_height // 2)
        dot_x = box_x1 + padding_x + dot_radius
        text_x = dot_x + dot_radius + 6

        # Draw colored dot
        cv2.circle(vis_img, (dot_x, center_y), dot_radius, color_bgr, -1)
        cv2.circle(vis_img, (dot_x, center_y), dot_radius + 1, (0, 0, 0), 1)

        # Draw corresponding text label
        cv2.putText(
            vis_img,
            label_text,
            (text_x, center_y + 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def load_cvm_stage_mapping(
    export_json_path: str | None = None,
    api_url: str | None = None,
    api_token: str | None = None,
) -> dict[str, int]:
    """
    Loads ground-truth CVM stages (1-6) from CVM Backend API, local export JSON, or SQLite database.
    Maps filenames, stem identifiers, and task IDs to integer stage levels.
    """
    stage_map: dict[str, int] = {}

    def _register(key, stage_val):
        if key is None or stage_val is None:
            return
        try:
            istg = int(stage_val)
            if 1 <= istg <= 6:
                k_str = str(key).strip()
                stage_map[k_str] = istg
                stem = Path(k_str).stem
                stage_map[stem] = istg
                if k_str.isdigit():
                    stage_map[f"{int(k_str)}.jpg"] = istg
                    stage_map[f"{int(k_str)}.png"] = istg
                    stage_map[f"{int(k_str):04d}"] = istg
                    stage_map[f"{int(k_str):04d}.jpg"] = istg
                    stage_map[f"{int(k_str):04d}.png"] = istg
        except (ValueError, TypeError):
            pass

    # 1. Custom or explicitly provided export JSON
    if export_json_path and os.path.exists(export_json_path):
        try:
            with open(export_json_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            tasks = raw_data.get("data", raw_data) if isinstance(raw_data, dict) else raw_data
            if isinstance(tasks, list):
                for t in tasks:
                    if not isinstance(t, dict):
                        continue
                    stg = (
                        t.get("finalStage")
                        or t.get("final_stage")
                        or t.get("cvmStage")
                        or t.get("cvm_stage")
                        or t.get("stage")
                    )
                    fn = t.get("filename") or t.get("file_upload") or t.get("data", {}).get("img")
                    tid = t.get("taskId") or t.get("task_id") or t.get("id")
                    if fn:
                        _register(os.path.basename(str(fn).split("?")[0]), stg)
                    if tid is not None:
                        _register(tid, stg)
            print(f"--> Loaded stage mapping from export file: {export_json_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not parse export file '{export_json_path}': {e}")

    # 2. CVM Backend API
    chosen_url = (api_url or os.getenv("CVM_API_URL") or "https://cvm.amirlogedixyz.ir").rstrip("/")
    chosen_token = api_token or os.getenv("CVM_API_TOKEN") or "O0s66Vi.r8e8WBGfY9I3x4neIm"
    if chosen_url:
        try:
            headers = {"Authorization": f"Bearer {chosen_token}"} if chosen_token else {}
            resp = requests.get(
                f"{chosen_url}/api/export?exclude_review_needed=false&exclude_excluded=false",
                headers=headers,
                timeout=12,
            )
            if resp.status_code == 200:
                tasks = resp.json().get("data", [])
                for t in tasks:
                    stg = (
                        t.get("finalStage")
                        or t.get("final_stage")
                        or t.get("cvmStage")
                        or t.get("cvm_stage")
                    )
                    fn = t.get("filename") or t.get("file_upload") or t.get("imageUrl") or t.get("data", {}).get("img")
                    tid = t.get("taskId") or t.get("task_id") or t.get("id")
                    if fn:
                        _register(os.path.basename(str(fn).split("?")[0]), stg)
                    if tid is not None:
                        _register(tid, stg)
                print(f"--> Loaded stage mapping from CVM Backend API at {chosen_url} ({len(stage_map)} entries).")
        except Exception as e:
            print(f"ℹ️ Note: Live CVM API fetch skipped/failed ({e}). Checking local database fallbacks...")

    # 3. Local SQLite database fallbacks
    db_candidates = [
        Path("/Users/macbookair/cvm-project/cvm-backend/data/annotations.db"),
        Path("data/annotations.db"),
        Path("../cvm-backend/data/annotations.db"),
    ]
    for db_path in db_candidates:
        if db_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute("SELECT task_id, filename, final_stage, cvm_stage FROM annotations")
                for tid, fn, fstg, cstg in c.fetchall():
                    stg = fstg if fstg is not None else cstg
                    if fn:
                        _register(os.path.basename(str(fn)), stg)
                    if tid is not None:
                        _register(tid, stg)
                conn.close()
                print(f"--> Supplemented stage mapping from local database: {db_path}")
                break
            except Exception as e:
                print(f"⚠️ Warning: Could not read local SQLite DB '{db_path}': {e}")

    # 4. Local export JSON candidates
    if len(stage_map) == 0:
        local_candidates = [
            Path("data/exports/export.json"),
            Path("/Users/macbookair/cvm-project/cvm-backend/data/dataset_backup.json"),
        ]
        for lc in local_candidates:
            if lc.exists():
                try:
                    with open(lc, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    tasks = raw_data.get("data", raw_data) if isinstance(raw_data, dict) else raw_data
                    if isinstance(tasks, list):
                        for t in tasks:
                            if not isinstance(t, dict):
                                continue
                            stg = t.get("finalStage") or t.get("final_stage") or t.get("cvmStage") or t.get("cvm_stage")
                            fn = t.get("filename") or t.get("file_upload") or t.get("data", {}).get("img")
                            tid = t.get("taskId") or t.get("id")
                            if fn:
                                _register(os.path.basename(str(fn).split("?")[0]), stg)
                            if tid is not None:
                                _register(tid, stg)
                    print(f"--> Supplemented stage mapping from local JSON: {lc}")
                    break
                except Exception:
                    pass

    return stage_map


def load_model(weights_path: str, device: torch.device, img_size: int = 640) -> torch.nn.Module:
    """
    Instantiates CephalometricSwinGCN and loads checkpoint weights safely,
    supporting checkpoints trained with torch.compile (_orig_mod.) and DataParallel (module.).
    """
    print(f"Loading checkpoint weights from: {weights_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at '{weights_path}'")

    model = CephalometricSwinGCN(num_landmarks=NUM_LANDMARKS, pretrained=False, img_size=img_size)
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        # Full saved model object
        model = checkpoint
        model.to(device)
        model.eval()
        return model

    # Clean state dict keys:
    # 1. Strip static coordinate grid buffers
    # 2. Strip PyTorch 2.0+ torch.compile '_orig_mod.' prefix
    # 3. Strip DataParallel/DistributedDataParallel 'module.' prefix
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
    matched_keys = [k for k in cleaned_state_dict.keys() if k not in unexpected_keys]

    if len(matched_keys) == 0:
        raise RuntimeError(
            f"FATAL: 0 weights matched when loading '{weights_path}'! Model would remain randomly initialized."
        )

    if missing_keys:
        critical_missing = [k for k in missing_keys if not k.endswith("adj_matrix")]
        if critical_missing:
            print(f"⚠️ Warning: Missing keys when loading checkpoint ({len(missing_keys)} total): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"⚠️ Warning: Unexpected keys in checkpoint ({len(unexpected_keys)} total): {unexpected_keys[:5]}...")

    print(f"--> Successfully loaded {len(matched_keys)}/{len(model.state_dict())} parameter tensors from '{weights_path}'.")

    model.to(device)
    model.eval()
    return model


def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    use_tta: bool = False,
    tta_scales: list[float] | tuple[float, ...] = (0.95, 1.0, 1.05),
    tta_shifts: list[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inference helper for a batch of images with optional Test-Time Augmentation (TTA).

    Args:
        model: CephalometricSwinGCN model.
        images: Input tensor of shape [B, 3, H, W] normalized.
        use_tta: Whether to apply multi-scale / shift TTA.
        tta_scales: Scale factors to evaluate (e.g. 0.95, 1.0, 1.05).
        tta_shifts: List of (dx, dy) pixel shift offsets (e.g. [(-4, -4), (4, 4)]).

    Returns:
        consensus_heatmaps: Heatmaps of shape [B, 13, H, W].
        consensus_coords: Landmark coordinates of shape [B, 13, 2] in normalized [0, 1] range.
    """
    if not use_tta:
        return model(images)

    B, C, H, W = images.shape
    all_coords = []
    all_heatmaps = []

    # 1. Multi-scale TTA
    if tta_scales:
        for s in tta_scales:
            if abs(s - 1.0) < 1e-4:
                hm, cd = model(images)
                all_coords.append(cd)
                all_heatmaps.append(hm)
            elif s < 1.0:
                # Zoom out: resize smaller and pad to (H, W)
                Hs = max(1, int(round(H * s)))
                Ws = max(1, int(round(W * s)))
                scaled_img = F.interpolate(images, size=(Hs, Ws), mode="bilinear", align_corners=False)
                pad_top = (H - Hs) // 2
                pad_bottom = H - Hs - pad_top
                pad_left = (W - Ws) // 2
                pad_right = W - Ws - pad_left
                padded_img = F.pad(scaled_img, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

                hm_aug, cd_aug = model(padded_img)

                # Invert coordinates from padded+scaled space back to original [0, 1]
                cd_x_px = cd_aug[..., 0] * W
                cd_y_px = cd_aug[..., 1] * H
                orig_cd_x = (cd_x_px - pad_left) / float(Ws)
                orig_cd_y = (cd_y_px - pad_top) / float(Hs)
                orig_cd = torch.stack([orig_cd_x, orig_cd_y], dim=-1).clamp(0.0, 1.0)
                all_coords.append(orig_cd)

                # Invert heatmaps: slice padded region and resize back to (H, W)
                hm_unpad = hm_aug[:, :, pad_top:pad_top + Hs, pad_left:pad_left + Ws]
                hm_orig = F.interpolate(hm_unpad, size=(H, W), mode="bilinear", align_corners=False)
                all_heatmaps.append(hm_orig)
            else:
                # Zoom in: resize larger and center crop to (H, W)
                Hs = int(round(H * s))
                Ws = int(round(W * s))
                scaled_img = F.interpolate(images, size=(Hs, Ws), mode="bilinear", align_corners=False)
                crop_top = (Hs - H) // 2
                crop_left = (Ws - W) // 2
                cropped_img = scaled_img[:, :, crop_top:crop_top + H, crop_left:crop_left + W]

                hm_aug, cd_aug = model(cropped_img)

                # Invert coordinates from cropped+scaled space back to original [0, 1]
                cd_x_px = cd_aug[..., 0] * W
                cd_y_px = cd_aug[..., 1] * H
                orig_cd_x = (cd_x_px + crop_left) / float(Ws)
                orig_cd_y = (cd_y_px + crop_top) / float(Hs)
                orig_cd = torch.stack([orig_cd_x, orig_cd_y], dim=-1).clamp(0.0, 1.0)
                all_coords.append(orig_cd)

                # Invert heatmaps: pad to (Hs, Ws) then resize down to (H, W)
                pad_top = crop_top
                pad_bottom = Hs - H - crop_top
                pad_left = crop_left
                pad_right = Ws - W - crop_left
                hm_pad = F.pad(hm_aug, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
                hm_orig = F.interpolate(hm_pad, size=(H, W), mode="bilinear", align_corners=False)
                all_heatmaps.append(hm_orig)

    # 2. Translation Shifts (if specified)
    if tta_shifts:
        for dx, dy in tta_shifts:
            if dx == 0 and dy == 0:
                continue
            pad_l = max(0, dx)
            pad_r = max(0, -dx)
            pad_t = max(0, dy)
            pad_b = max(0, -dy)
            shifted = F.pad(images, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
            shifted = shifted[:, :, pad_b:pad_b + H, pad_r:pad_r + W]

            hm_aug, cd_aug = model(shifted)

            orig_cd_x = cd_aug[..., 0] - (dx / float(W))
            orig_cd_y = cd_aug[..., 1] - (dy / float(H))
            orig_cd = torch.stack([orig_cd_x, orig_cd_y], dim=-1).clamp(0.0, 1.0)
            all_coords.append(orig_cd)

    # Consensus ensemble average
    consensus_coords = torch.mean(torch.stack(all_coords, dim=0), dim=0)
    if len(all_heatmaps) > 0:
        consensus_heatmaps = torch.mean(torch.stack(all_heatmaps, dim=0), dim=0)
    else:
        consensus_heatmaps, _ = model(images)

    return consensus_heatmaps, consensus_coords


def compute_metrics(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    img_size: int = 640,
    threshold_px: float = 2.5,
) -> tuple[dict, list[dict], np.ndarray]:
    """
    Computes overall summary metrics and per-landmark metrics breakdown in both normalized percentage of C3 reference length and canvas pixels.

    Args:
        pred_coords: Array of shape (N, 13, 2) in normalized [0, 1] range.
        gt_coords: Array of shape (N, 13, 2) in normalized [0, 1] range (-1.0 for missing).
        img_size: Image dimension in pixels (default 640).
        threshold_px: Tolerance threshold in pixels for binary detection metrics.

    Returns:
        summary_metrics: Dict of overall dataset evaluation metrics.
        landmark_metrics: List of dicts with metrics per landmark.
        valid_radial_errors: Array of radial errors in pixels for valid landmarks.
    """
    # Convert normalized coordinates to absolute pixel scale
    pred_px = pred_coords * img_size
    gt_px = gt_coords * img_size

    # Valid mask for GT coordinates (excluding missing landmarks marked as -1.0)
    valid_mask = (gt_coords[:, :, 0] >= 0) & (gt_coords[:, :, 1] >= 0)  # Shape: (N, 13)

    dx = pred_px[:, :, 0] - gt_px[:, :, 0]
    dy = pred_px[:, :, 1] - gt_px[:, :, 1]
    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    radial_errors = np.sqrt(dx**2 + dy**2)  # Shape: (N, 13) in pixels

    # Reference length per sample (C3 inferior endplate width)
    ref_lengths = compute_reference_length(gt_coords, img_size=img_size)
    ref_lengths_2d = ref_lengths[:, None]  # shape (N, 1)
    norm_errors = (radial_errors / ref_lengths_2d) * 100.0  # % of ref length

    valid_radial_errors = radial_errors[valid_mask]
    valid_norm_errors = norm_errors[valid_mask]
    valid_abs_dx = abs_dx[valid_mask]
    valid_abs_dy = abs_dy[valid_mask]

    if len(valid_radial_errors) == 0:
        raise ValueError("No valid ground truth landmarks found for metric calculation.")

    # 1. Normalized Metrics (% of C3 Inferior Endplate Width)
    mean_norm = float(np.mean(valid_norm_errors))
    median_norm = float(np.median(valid_norm_errors))
    sd_norm = float(np.std(valid_norm_errors))
    rmse_norm = float(np.sqrt(np.mean(valid_norm_errors ** 2)))
    min_norm = float(np.min(valid_norm_errors))
    max_norm = float(np.max(valid_norm_errors))

    norm_sdr_thresholds = [1.0, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0]
    norm_sdr_dict = {}
    for th in norm_sdr_thresholds:
        clean_th = str(th).replace(".", "_")
        norm_sdr_dict[f"share_within_{clean_th}pct"] = float(np.mean(valid_norm_errors <= th) * 100.0)

    # 2. Pixel Scale Metrics
    mae_x = float(np.mean(valid_abs_dx))
    mae_y = float(np.mean(valid_abs_dy))
    mae = float(np.mean((valid_abs_dx + valid_abs_dy) / 2.0))
    rmse = float(np.sqrt(np.mean(dx[valid_mask] ** 2 + dy[valid_mask] ** 2)))
    mre = float(np.mean(valid_radial_errors))
    medre = float(np.median(valid_radial_errors))
    sdre = float(np.std(valid_radial_errors))
    min_err = float(np.min(valid_radial_errors))
    max_err = float(np.max(valid_radial_errors))

    sdr_thresholds = [2.0, 2.5, 3.0, 4.0, 5.0, 10.0]
    sdr_dict = {}
    for th in sdr_thresholds:
        sdr_val = float(np.mean(valid_radial_errors <= th) * 100.0)
        sdr_dict[f"sdr_{th}px"] = sdr_val

    total_valid = len(valid_radial_errors)

    summary_metrics = {
        "total_samples": int(pred_coords.shape[0]),
        "total_valid_landmarks": total_valid,
        "img_size": img_size,
        # Normalized Percentage Metrics (Primary Clinical Benchmark)
        "mean_norm_err": mean_norm,
        "median_norm_err": median_norm,
        "sd_norm_err": sd_norm,
        "rmse_norm_err": rmse_norm,
        "min_norm_err": min_norm,
        "max_norm_err": max_norm,
        "share_within_5pct": norm_sdr_dict.get("share_within_5_0pct", 0.0),
        "share_within_10pct": norm_sdr_dict.get("share_within_10_0pct", 0.0),
        **norm_sdr_dict,
        # Canvas Pixel Metrics
        "mae_pixels": mae,
        "mae_x_pixels": mae_x,
        "mae_y_pixels": mae_y,
        "rmse_pixels": rmse,
        "mre_pixels": mre,
        "medre_pixels": medre,
        "sdre_pixels": sdre,
        "min_error_pixels": min_err,
        "max_error_pixels": max_err,
        **sdr_dict,
    }

    # 3. Per-landmark Breakdown
    landmark_metrics = []
    for i, name in enumerate(LANDMARK_CLASSES):
        l_mask = valid_mask[:, i]
        if not np.any(l_mask):
            continue

        vert = name.split("_")[0] if "_" in name else ""
        lm_type = name.split("_")[1] if "_" in name else ""
        l_radial = radial_errors[:, i][l_mask]
        l_norm = norm_errors[:, i][l_mask]
        l_dx = dx[:, i][l_mask]
        l_dy = dy[:, i][l_mask]
        l_abs_dx = abs_dx[:, i][l_mask]
        l_abs_dy = abs_dy[:, i][l_mask]

        l_mae = float(np.mean((l_abs_dx + l_abs_dy) / 2.0))
        l_rmse = float(np.sqrt(np.mean(l_dx**2 + l_dy**2)))
        l_mre = float(np.mean(l_radial))
        l_medre = float(np.median(l_radial))
        l_sdre = float(np.std(l_radial)) if len(l_radial) > 1 else 0.0

        l_mean_norm = float(np.mean(l_norm))
        l_med_norm = float(np.median(l_norm))
        l_sd_norm = float(np.std(l_norm)) if len(l_norm) > 1 else 0.0
        l_share_2_5 = float(np.mean(l_norm <= 2.5) * 100.0)
        l_share_5 = float(np.mean(l_norm <= 5.0) * 100.0)
        l_share_7_5 = float(np.mean(l_norm <= 7.5) * 100.0)
        l_share_10 = float(np.mean(l_norm <= 10.0) * 100.0)

        # Pixels
        l_sdr2_0 = float(np.mean(l_radial <= 2.0) * 100.0)
        l_sdr2_5 = float(np.mean(l_radial <= 2.5) * 100.0)
        l_sdr3_0 = float(np.mean(l_radial <= 3.0) * 100.0)
        l_sdr4_0 = float(np.mean(l_radial <= 4.0) * 100.0)

        entry = {
            "id": i,
            "name": name,
            "vertebra": vert,
            "type": lm_type,
            "count": int(np.sum(l_mask)),
            # Normalized
            "mean_norm_err": l_mean_norm,
            "median_norm_err": l_med_norm,
            "sd_norm_err": l_sd_norm,
            "share_within_2_5pct": l_share_2_5,
            "share_within_5pct": l_share_5,
            "share_within_7_5pct": l_share_7_5,
            "share_within_10pct": l_share_10,
            # Pixels
            "mae_px": l_mae,
            "rmse_px": l_rmse,
            "mre_px": l_mre,
            "medre_px": l_medre,
            "sdre_px": l_sdre,
            "sdr_2.0px": l_sdr2_0,
            "sdr_2.5px": l_sdr2_5,
            "sdr_3.0px": l_sdr3_0,
            "sdr_4.0px": l_sdr4_0,
        }

        # Inferior border projections
        if vert in ("C2", "C3", "C4") and lm_type in ("PI", "IC", "AI"):
            try:
                a_px, p_px, a_pct, p_pct = compute_inferior_border_projections(
                    pred_coords,
                    gt_coords,
                    vertebra=vert,
                    landmark_key=lm_type,
                    img_size=img_size,
                    ref_lengths=ref_lengths,
                    return_pct=True,
                )
                if len(a_px) > 0:
                    entry["mean_along_px"] = float(np.mean(a_px))
                    entry["mean_perp_px"] = float(np.mean(p_px))
                    entry["mean_along_pct"] = float(np.mean(a_pct))
                    entry["mean_perp_pct"] = float(np.mean(p_pct))
            except Exception:
                pass

        landmark_metrics.append(entry)

    return summary_metrics, landmark_metrics, valid_radial_errors


def format_metrics_table(
    summary: dict,
    landmarks: list[dict],
    run_name: str,
    weights_path: str,
) -> str:
    """
    Formats the evaluation results into publication-ready ASCII terminal tables.
    """
    lines = []
    lines.append("=" * 96)
    lines.append(f" SWIN-GCN CLINICAL MODEL EVALUATION REPORT | RUN: {run_name}")
    lines.append("=" * 96)
    tta_str = f"Enabled (scales={summary.get('tta_scales')})" if summary.get("use_tta") else "Disabled"
    lines.append(f"Checkpoint Weights: {weights_path} | TTA: {tta_str}")
    lines.append(f"Image Size: {summary.get('img_size', 640)}x{summary.get('img_size', 640)}")
    lines.append(f"Total Test Images: {summary['total_samples']} | Total Valid Landmarks: {summary['total_valid_landmarks']}")
    lines.append("-" * 96)

    # 1. Normalized Landmark Metrics Table (Primary Benchmark)
    lines.append("\n" + "+" + "-" * 50 + "+" + "-" * 43 + "+")
    lines.append(f"| {'NORMALIZED METRIC (% OF C3 REF LENGTH)':<48} | {'VALUE':<41} |")
    lines.append("+" + "-" * 50 + "+" + "-" * 43 + "+")

    norm_rows = [
        ("Mean Normalized Error", f"{summary.get('mean_norm_err', 0.0):.2f} %"),
        ("Median Normalized Error", f"{summary.get('median_norm_err', 0.0):.2f} %"),
        ("RMSE Normalized Error", f"{summary.get('rmse_norm_err', 0.0):.2f} %"),
        ("SD Normalized Error", f"{summary.get('sd_norm_err', 0.0):.2f} %"),
        ("Min Normalized Error", f"{summary.get('min_norm_err', 0.0):.2f} %"),
        ("Max Normalized Error", f"{summary.get('max_norm_err', 0.0):.2f} %"),
        ("Share ≤ 2.5% (High Precision)", f"{summary.get('share_within_2_5pct', 0.0):.2f} %"),
        ("Share ≤ 5.0% (Primary Clinical Benchmark)", f"{summary.get('share_within_5pct', 0.0):.2f} %"),
        ("Share ≤ 7.5%", f"{summary.get('share_within_7_5pct', 0.0):.2f} %"),
        ("Share ≤ 10.0% (Coarse Localization)", f"{summary.get('share_within_10_0pct', 0.0):.2f} %"),
    ]
    for label, val in norm_rows:
        lines.append(f"| {label:<48} | {val:<41} |")
    lines.append("+" + "-" * 50 + "+" + "-" * 43 + "+")

    # 2. Pixel Scale Metrics Table (Secondary Reference)
    lines.append("\n" + "+" + "-" * 50 + "+" + "-" * 43 + "+")
    lines.append(f"| {'CANVAS METRIC (PIXELS)':<48} | {'VALUE':<41} |")
    lines.append("+" + "-" * 50 + "+" + "-" * 43 + "+")
    px_rows = [
        ("MAE (Pixels)", f"{summary['mae_pixels']:.2f} px"),
        ("RMSE (Pixels)", f"{summary['rmse_pixels']:.2f} px"),
        ("MRE (Pixels)", f"{summary['mre_pixels']:.2f} px"),
        ("SDR @ 2.0 px", f"{summary.get('sdr_2.0px', 0.0):.2f} %"),
        ("SDR @ 2.5 px", f"{summary.get('sdr_2.5px', 0.0):.2f} %"),
        ("SDR @ 4.0 px", f"{summary.get('sdr_4.0px', 0.0):.2f} %"),
        ("SDR @ 10.0 px", f"{summary.get('sdr_10.0px', 0.0):.2f} %"),
    ]
    for label, val in px_rows:
        lines.append(f"| {label:<48} | {val:<41} |")
    lines.append("+" + "-" * 50 + "+" + "-" * 43 + "+")

    # 3. Per-Landmark Breakdown Table
    lines.append("\n" + "+" + "-" * 96 + "+")
    lines.append(f"| {'PER-LANDMARK NORMALIZED PERFORMANCE BREAKDOWN':^94} |")
    lines.append("+" + "-" * 4 + "+" + "-" * 11 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 11 + "+")
    lines.append(f"| {'ID':<2} | {'Landmark':<9} | {'Mean Norm(%)':<12} | {'Med Norm(%)':<12} | {'Share≤2.5%':<12} | {'Share≤5.0%':<12} | {'MAE(px)':<10} | {'SDR@2.5px':<11} |")
    lines.append("+" + "-" * 4 + "+" + "-" * 11 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 11 + "+")

    for lm in landmarks:
        lines.append(
            f"| {lm['id']:<2} | {lm['name']:<9} | {lm.get('mean_norm_err', 0.0):<12.2f} | {lm.get('median_norm_err', 0.0):<12.2f} | "
            f"{lm.get('share_within_2_5pct', 0.0):<11.1f}% | {lm.get('share_within_5pct', 0.0):<11.1f}% | "
            f"{lm['mae_px']:<10.2f} | {lm.get('sdr_2.5px', 0.0):<10.1f}% |"
        )
    lines.append("+" + "-" * 4 + "+" + "-" * 11 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 11 + "+")

    return "\n".join(lines)


def generate_evaluation_charts(
    summary_metrics: dict,
    landmark_metrics: list[dict],
    valid_radial_errors: np.ndarray,
    save_dir: Path | str | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, str]:
    """
    Generates PNG diagram charts for all normalized and canvas pixel evaluation metrics
    and logs them as artifacts to MLflow.
    """
    use_temp = save_dir is None
    temp_dir_obj = None
    if use_temp:
        temp_dir_obj = tempfile.TemporaryDirectory()
        target_dir = Path(temp_dir_obj.name)
    else:
        target_dir = Path(save_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    generated_charts = {}

    try:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.family": "sans-serif"})

        lm_names = [lm["name"] for lm in landmark_metrics]
        lm_indices = np.arange(len(landmark_metrics))
        width = 0.35

        # 1. Landmark Errors Bar Chart (Mean Norm % vs Median Norm %)
        fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
        mean_norm_vals = [lm.get("mean_norm_err", 0.0) for lm in landmark_metrics]
        med_norm_vals = [lm.get("median_norm_err", 0.0) for lm in landmark_metrics]

        ax1.bar(lm_indices - width / 2, mean_norm_vals, width, label="Mean Norm Error (%)", color="#e76f51", alpha=0.9)
        ax1.bar(lm_indices + width / 2, med_norm_vals, width, label="Median Norm Error (%)", color="#457b9d", alpha=0.9)

        ax1.set_xticks(lm_indices)
        ax1.set_xticklabels(lm_names, rotation=45, ha="right")
        ax1.set_ylabel("Normalized Error (% of C3 ref width)")
        ax1.set_title("Per-Landmark Normalized Errors Breakdown (Mean vs Median %)", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p1 = target_dir / "chart_landmark_errors.png"
        fig1.savefig(p1, bbox_inches="tight")
        plt.close(fig1)
        generated_charts["landmark_errors"] = str(p1)

        # 2. Per-Landmark Tolerance Bar Chart (Share ≤ 2.5% vs Share ≤ 5.0% vs Share ≤ 10.0%)
        fig2, ax2 = plt.subplots(figsize=(11, 5), dpi=300)
        w3 = 0.25
        sdr2_5 = [lm.get("share_within_2_5pct", 0.0) for lm in landmark_metrics]
        sdr5_0 = [lm.get("share_within_5pct", 0.0) for lm in landmark_metrics]
        sdr10_0 = [lm.get("share_within_10pct", 0.0) for lm in landmark_metrics]

        ax2.bar(lm_indices - w3, sdr2_5, w3, label="Share ≤ 2.5% (High Precision)", color="#2a9d8f", alpha=0.9)
        ax2.bar(lm_indices, sdr5_0, w3, label="Share ≤ 5.0% (Primary Benchmark)", color="#3a86ff", alpha=0.9)
        ax2.bar(lm_indices + w3, sdr10_0, w3, label="Share ≤ 10.0% (Coarse)", color="#8338ec", alpha=0.9)

        ax2.set_xticks(lm_indices)
        ax2.set_xticklabels(lm_names, rotation=45, ha="right")
        ax2.set_ylabel("Share (%)")
        ax2.set_ylim(0, 105)
        ax2.set_title("Per-Landmark Normalized Detection Rates (≤2.5% / ≤5.0% / ≤10.0%)", fontsize=12, fontweight="bold")
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p2 = target_dir / "chart_landmark_sdr.png"
        fig2.savefig(p2, bbox_inches="tight")
        plt.close(fig2)
        generated_charts["landmark_sdr"] = str(p2)

        # 3. SDR Threshold Curve (Cumulative Detection Rate across Normalized Thresholds)
        fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=300)
        th_norm = [1.0, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0, 15.0]
        sdr_values = [
            summary_metrics.get(f"share_within_{str(th).replace('.', '_')}pct", 0.0)
            for th in th_norm
        ]

        ax3.plot(th_norm, sdr_values, marker="o", linewidth=2.5, color="#ff006e", markersize=8)
        for th, val in zip(th_norm, sdr_values):
            ax3.annotate(f"{val:.1f}%", (th, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

        ax3.set_xlabel("Error Threshold (% of C3 Inferior Endplate Width)")
        ax3.set_ylabel("Successful Detection Rate (%)")
        ax3.set_ylim(0, 108)
        ax3.set_title("Cumulative SDR Curve Across Normalized Distance Thresholds (%)", fontsize=12, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p3 = target_dir / "chart_sdr_thresholds.png"
        fig3.savefig(p3, bbox_inches="tight")
        plt.close(fig3)
        generated_charts["sdr_thresholds"] = str(p3)

        # 4. Error Distribution Histogram (Normalized % or Canvas px)
        fig4, ax4 = plt.subplots(figsize=(8, 4.5), dpi=300)
        mean_err_norm = summary_metrics.get("mean_norm_err", 0.0)
        med_err_norm = summary_metrics.get("median_norm_err", 0.0)

        # Plot pixel errors histogram
        ax4.hist(valid_radial_errors, bins=30, color="#fb8500", edgecolor="black", alpha=0.7, density=True, label="Radial Error (px)")
        mre_px = summary_metrics.get("mre_pixels", float(np.mean(valid_radial_errors)))
        med_px = summary_metrics.get("medre_pixels", float(np.median(valid_radial_errors)))
        ax4.axvline(mre_px, color="red", linestyle="--", linewidth=2, label=f"Mean (MRE): {mre_px:.2f} px")
        ax4.axvline(med_px, color="green", linestyle="-.", linewidth=2, label=f"Median: {med_px:.2f} px")

        ax4.set_xlabel("Radial Error (Pixels)")
        ax4.set_ylabel("Density")
        ax4.set_title(f"Radial Error Distribution (Mean Norm: {mean_err_norm:.2f}%, Med Norm: {med_err_norm:.2f}%)", fontsize=12, fontweight="bold")
        ax4.legend(loc="upper right")
        ax4.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p4 = target_dir / "chart_radial_error_distribution.png"
        fig4.savefig(p4, bbox_inches="tight")
        plt.close(fig4)
        generated_charts["radial_error_distribution"] = str(p4)

        # 5. Evaluation Dashboard (2x2 Grid)
        fig5, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

        # Subplot (0,0): Landmark Normalized Errors
        axes[0, 0].bar(lm_indices - width / 2, mean_norm_vals, width, label="Mean Norm (%)", color="#e76f51")
        axes[0, 0].bar(lm_indices + width / 2, med_norm_vals, width, label="Med Norm (%)", color="#457b9d")
        axes[0, 0].set_xticks(lm_indices)
        axes[0, 0].set_xticklabels(lm_names, rotation=45, ha="right", fontsize=8)
        axes[0, 0].set_ylabel("Error (% C3 width)")
        axes[0, 0].set_title("Landmark Normalized Errors (% C3 width)", fontsize=10, fontweight="bold")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

        # Subplot (0,1): Per-Landmark SDR
        axes[0, 1].bar(lm_indices - w3, sdr2_5, w3, label="≤2.5%", color="#2a9d8f")
        axes[0, 1].bar(lm_indices, sdr5_0, w3, label="≤5.0%", color="#3a86ff")
        axes[0, 1].bar(lm_indices + w3, sdr10_0, w3, label="≤10.0%", color="#8338ec")
        axes[0, 1].set_xticks(lm_indices)
        axes[0, 1].set_xticklabels(lm_names, rotation=45, ha="right", fontsize=8)
        axes[0, 1].set_ylabel("Share (%)")
        axes[0, 1].set_title("Per-Landmark Normalized Tolerances", fontsize=10, fontweight="bold")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        # Subplot (1,0): SDR Curve
        axes[1, 0].plot(th_norm, sdr_values, marker="o", color="#ff006e", linewidth=2)
        axes[1, 0].set_title("Cumulative SDR Curve (% of C3 width)", fontsize=10, fontweight="bold")
        axes[1, 0].set_xlabel("Threshold (%)", fontsize=9)
        axes[1, 0].set_ylabel("SDR (%)", fontsize=9)
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

        # Subplot (1,1): Error Distribution
        axes[1, 1].hist(valid_radial_errors, bins=25, color="#fb8500", alpha=0.7, density=True)
        axes[1, 1].axvline(mre_px, color="red", linestyle="--", label=f"MRE: {mre_px:.2f}px")
        axes[1, 1].set_title("Radial Error Distribution (px)", fontsize=10, fontweight="bold")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("SWIN-GCN NORMALIZED EVALUATION METRICS DASHBOARD", fontsize=14, fontweight="bold")
        plt.tight_layout()
        p5 = target_dir / "chart_evaluation_dashboard.png"
        fig5.savefig(p5, bbox_inches="tight")
        plt.close(fig5)
        generated_charts["evaluation_dashboard"] = str(p5)

        # Log artifacts to MLflow if tracking is active
        if log_to_mlflow:
            try:
                active_run = mlflow.active_run()
                if active_run is not None:
                    mlflow.log_artifacts(str(target_dir), artifact_path="evaluation/charts")
                    print(f"--> Successfully logged evaluation metric PNG charts to MLflow artifact path 'evaluation/charts'")
            except Exception as ml_err:
                print(f"⚠️ Warning: Could not log evaluation charts to MLflow: {ml_err}")

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

    return generated_charts


VERTEBRA_CONTOURS = [
    ("C2", [0, 1, 2], False),
    ("C3", [3, 4, 7, 6, 5], True),
    ("C4", [8, 9, 12, 11, 10], True),
]


def compute_roi_bbox(
    gt_coords: np.ndarray | None,
    pred_coords: np.ndarray | None,
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.35,
    min_size: int = 180,
) -> tuple[int, int, int, int]:
    """
    Computes a square bounding box centered around valid cervical landmarks with padding.
    Coordinates are expected to be normalized [0, 1].
    """
    valid_pts = []
    if gt_coords is not None:
        for pt in gt_coords:
            if pt[0] >= 0 and pt[1] >= 0 and not np.isnan(pt[0]) and not np.isnan(pt[1]):
                valid_pts.append(pt)
    if pred_coords is not None:
        for pt in pred_coords:
            if pt[0] >= 0 and pt[1] >= 0 and not np.isnan(pt[0]) and not np.isnan(pt[1]):
                valid_pts.append(pt)

    if not valid_pts:
        half = min(img_w, img_h) // 4
        cx, cy = img_w // 2, img_h // 2
        return max(0, cx - half), max(0, cy - half), min(img_w, cx + half), min(img_h, cy + half)

    pts = np.array(valid_pts)
    xs = pts[:, 0] * img_w
    ys = pts[:, 1] * img_h

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    box_w = max_x - min_x
    box_h = max_y - min_y
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    dim = max(box_w, box_h) * (1.0 + 2.0 * margin_ratio)
    dim = max(dim, min_size)
    half_dim = dim / 2.0

    x1 = int(round(cx - half_dim))
    x2 = int(round(cx + half_dim))
    y1 = int(round(cy - half_dim))
    y2 = int(round(cy + half_dim))

    if x1 < 0:
        x2 = min(img_w, x2 - x1)
        x1 = 0
    if x2 > img_w:
        x1 = max(0, x1 - (x2 - img_w))
        x2 = img_w

    if y1 < 0:
        y2 = min(img_h, y2 - y1)
        y1 = 0
    if y2 > img_h:
        y1 = max(0, y1 - (y2 - img_h))
        y2 = img_h

    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def render_roi_panel(
    roi_rgb: np.ndarray,
    coords: np.ndarray,
    img_w: int,
    img_h: int,
    x1: int,
    y1: int,
    roi_w: int,
    roi_h: int,
    panel_size: int = 560,
    title: str = "Ground Truth",
    title_color: tuple[int, int, int] = (76, 175, 80),
) -> np.ndarray:
    """
    Renders one high-resolution ROI panel (Ground Truth or Prediction)
    with landmark dots, number badges, and anatomical vertebra contours.
    """
    panel = cv2.resize(roi_rgb, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)

    # Convert coordinates to panel pixel space
    pts_dict = {}
    for i, pt in enumerate(coords):
        if pt[0] < 0 or pt[1] < 0 or np.isnan(pt[0]) or np.isnan(pt[1]):
            continue
        px_full = pt[0] * img_w
        py_full = pt[1] * img_h
        px_p = int(round((px_full - x1) / float(roi_w) * panel_size))
        py_p = int(round((py_full - y1) / float(roi_h) * panel_size))
        pts_dict[i] = (px_p, py_p)

    # Draw landmark points (clean circles with contrast border and pinpoint center)
    for i, _ in enumerate(coords):
        if i not in pts_dict:
            continue
        px, py = pts_dict[i]
        color_bgr = LANDMARK_COLORS_BGR[i % len(LANDMARK_COLORS_BGR)]
        # Convert BGR constant to RGB for panel
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        # Outer black ring for contrast against bright bone
        cv2.circle(panel, (px, py), 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        # Colored landmark dot
        cv2.circle(panel, (px, py), 5, color_rgb, -1, lineType=cv2.LINE_AA)
        # Inner white center point for precise pinpointing
        cv2.circle(panel, (px, py), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # 3. Add panel title banner at top
    banner_h = 36
    banner = np.zeros((banner_h, panel_size, 3), dtype=np.uint8)
    banner[:] = (25, 28, 32)
    cv2.line(banner, (0, banner_h - 1), (panel_size, banner_h - 1), title_color, 2)

    (ttw, tth), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(
        banner,
        title,
        ((panel_size - ttw) // 2, (banner_h + tth) // 2 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        title_color,
        2,
        lineType=cv2.LINE_AA,
    )

    panel_with_header = np.vstack([banner, panel])
    return panel_with_header


def create_side_by_side_roi_visualization(
    image_rgb: np.ndarray,
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    sample_name: str = "",
    panel_size: int = 560,
) -> np.ndarray:
    """
    Creates an interpretable side-by-side ROI comparison image showing only the cervical vertebrae:
    - Left panel: Ground Truth
    - Right panel: Model Prediction
    - Top banner: Sample name and Mean Radial Error (pixels & normalized %)
    - Bottom banner: Clean 3-group landmark guide (C2, C3, C4)
    """
    img_h, img_w = image_rgb.shape[:2]

    # Compute ROI bounding box
    x1, y1, x2, y2 = compute_roi_bbox(
        gt_coords, pred_coords, img_w, img_h, margin_ratio=0.35, min_size=180
    )
    roi_rgb = image_rgb[y1:y2, x1:x2]
    roi_w = x2 - x1
    roi_h = y2 - y1

    # Compute sample-level error
    valid_mask = (gt_coords[:, 0] >= 0) & (gt_coords[:, 1] >= 0)
    if np.any(valid_mask):
        gt_px = gt_coords[valid_mask] * np.array([img_w, img_h])
        pred_px = pred_coords[valid_mask] * np.array([img_w, img_h])
        radial_errors_px = np.sqrt(np.sum((gt_px - pred_px) ** 2, axis=1))
        mre_px = float(np.mean(radial_errors_px))
        try:
            ref_len = compute_reference_length(gt_coords[None, ...], img_size=max(img_w, img_h))[0]
            norm_pct = (mre_px / ref_len) * 100.0 if ref_len > 1e-4 else 0.0
            metric_str = f"Mean Radial Error: {mre_px:.2f} px ({norm_pct:.1f}% C3 width)"
        except Exception:
            metric_str = f"Mean Radial Error: {mre_px:.2f} px"
    else:
        metric_str = "Mean Radial Error: N/A"

    # Render Left Panel (Ground Truth)
    panel_gt = render_roi_panel(
        roi_rgb=roi_rgb,
        coords=gt_coords,
        img_w=img_w,
        img_h=img_h,
        x1=x1,
        y1=y1,
        roi_w=roi_w,
        roi_h=roi_h,
        panel_size=panel_size,
        title="GROUND TRUTH",
        title_color=(76, 175, 80),
    )

    # Render Right Panel (Prediction)
    panel_pred = render_roi_panel(
        roi_rgb=roi_rgb,
        coords=pred_coords,
        img_w=img_w,
        img_h=img_h,
        x1=x1,
        y1=y1,
        roi_w=roi_w,
        roi_h=roi_h,
        panel_size=panel_size,
        title="MODEL PREDICTION",
        title_color=(33, 150, 243),
    )

    # Divider bar
    divider_w = 6
    divider = np.full((panel_gt.shape[0], divider_w, 3), 40, dtype=np.uint8)

    # Combine panels side by side
    panels_combined = np.hstack([panel_gt, divider, panel_pred])
    total_w = panels_combined.shape[1]

    # Top Header
    header_h = 44
    header = np.zeros((header_h, total_w, 3), dtype=np.uint8)
    header[:] = (18, 20, 24)

    header_left = f"Sample: {sample_name}" if sample_name else "CVM Vertebrae Evaluation"
    cv2.putText(
        header,
        header_left,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        1,
        lineType=cv2.LINE_AA,
    )

    (mw, _), _ = cv2.getTextSize(metric_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(
        header,
        metric_str,
        (total_w - mw - 16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 200, 100),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.line(header, (0, header_h - 1), (total_w, header_h - 1), (50, 55, 65), 1)

    # Bottom Legend (split by vertebra: C2, C3, C4)
    legend_h = 80
    legend = np.zeros((legend_h, total_w, 3), dtype=np.uint8)
    legend[:] = (18, 20, 24)
    cv2.line(legend, (0, 0), (total_w, 0), (50, 55, 65), 1)

    groups = [
        ("C2", [0, 1, 2]),
        ("C3", [3, 4, 5, 6, 7]),
        ("C4", [8, 9, 10, 11, 12]),
    ]

    col_w = total_w // 3
    for col_idx, (grp_name, indices) in enumerate(groups):
        start_x = col_idx * col_w + 16
        cv2.putText(
            legend,
            f"[{grp_name}]",
            (start_x, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 200),
            1,
            lineType=cv2.LINE_AA,
        )

        for row_i, idx in enumerate(indices):
            color_bgr = LANDMARK_COLORS_BGR[idx % len(LANDMARK_COLORS_BGR)]
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            name = LANDMARK_CLASSES[idx]

            if len(indices) <= 3:
                lx = start_x + 60 + (row_i * 90)
                ly = 20
            else:
                if row_i < 3:
                    lx = start_x + 50 + (row_i * 90)
                    ly = 20
                else:
                    lx = start_x + 50 + ((row_i - 3) * 90)
                    ly = 50

            cv2.circle(legend, (lx, ly - 4), 5, color_rgb, -1, lineType=cv2.LINE_AA)
            cv2.circle(legend, (lx, ly - 4), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(
                legend,
                f"{idx:02d}:{name.split('_')[-1]}",
                (lx + 9, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (230, 230, 230),
                1,
                lineType=cv2.LINE_AA,
            )

    final_image = np.vstack([header, panels_combined, legend])
    return final_image


def draw_landmarks_on_image(
    image: np.ndarray,
    pred_coords: np.ndarray,
    gt_coords: np.ndarray | None = None,
    img_size: int = 640,
) -> np.ndarray:
    """
    Renders predicted landmarks (and optionally ground truth) onto an RGB image.
    Preserved for backward-compatibility with full-image visualizations.
    """
    vis_img = image.copy()
    h, w = vis_img.shape[:2]

    # Draw Ground Truth landmarks first (Yellow circle markers)
    if gt_coords is not None:
        for i in range(len(gt_coords)):
            x_norm, y_norm = gt_coords[i]
            if x_norm < 0 or y_norm < 0:
                continue
            abs_x = int(round(x_norm * w))
            abs_y = int(round(y_norm * h))
            cv2.circle(vis_img, (abs_x, abs_y), 6, (255, 255, 0), 1)
            cv2.drawMarker(vis_img, (abs_x, abs_y), (255, 255, 0), cv2.MARKER_CROSS, 8, 1)

    # Draw Predicted landmarks with distinct colors
    for i in range(len(pred_coords)):
        x_norm, y_norm = pred_coords[i]
        if x_norm < 0 or y_norm < 0:
            continue
        abs_x = int(round(x_norm * w))
        abs_y = int(round(y_norm * h))

        color_bgr = LANDMARK_COLORS_BGR[i % len(LANDMARK_COLORS_BGR)]
        cv2.circle(vis_img, (abs_x, abs_y), 4, color_bgr, -1)
        cv2.circle(vis_img, (abs_x, abs_y), 5, (0, 0, 0), 1)

    # Draw Top-Right Landmark Legend Guide Box
    legend_x = w - 145
    legend_y = 12
    legend_w = 135
    legend_h = len(LANDMARK_CLASSES) * 16 + 10

    overlay = vis_img.copy()
    cv2.rectangle(
        overlay,
        (legend_x, legend_y),
        (legend_x + legend_w, legend_y + legend_h),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.75, vis_img, 0.25, 0, vis_img)
    cv2.rectangle(
        vis_img,
        (legend_x, legend_y),
        (legend_x + legend_w, legend_y + legend_h),
        (180, 180, 180),
        1,
    )

    for i, name in enumerate(LANDMARK_CLASSES):
        item_y = legend_y + 14 + (i * 16)
        color_bgr = LANDMARK_COLORS_BGR[i % len(LANDMARK_COLORS_BGR)]
        cv2.circle(vis_img, (legend_x + 10, item_y - 4), 4, color_bgr, -1)
        cv2.circle(vis_img, (legend_x + 10, item_y - 4), 5, (255, 255, 255), 1)
        cv2.putText(
            vis_img,
            f"{i:02d}: {name}",
            (legend_x + 20, item_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis_img


def visualize_batch_and_save(
    test_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    save_dir: Path,
    num_samples: int = 8,
    img_size: int = 640,
    test_img_dir: str = "dataset/test/images",
    use_tta: bool = False,
    tta_scales: list[float] | tuple[float, ...] = (0.95, 1.0, 1.05),
    tta_shifts: list[tuple[int, int]] | None = None,
    roi_only: bool = True,
    panel_size: int = 560,
) -> None:
    """
    Renders visual ground truth vs. predicted landmark comparisons and exports high-resolution PNGs.
    By default (roi_only=True), produces side-by-side ROI crops showing only the cervical vertebrae
    with ground truth on the left and model prediction on the right.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    with torch.no_grad():
        for batch in test_loader:
            if saved_count >= num_samples:
                break

            images = batch["image"].to(device)
            gt_coords = batch["coords"].cpu().numpy()
            filenames = batch["filename"]

            _, pred_coords = predict_batch(
                model=model,
                images=images,
                use_tta=use_tta,
                tta_scales=tta_scales,
                tta_shifts=tta_shifts,
            )
            pred_coords = pred_coords.cpu().numpy()

            for i in range(len(filenames)):
                if saved_count >= num_samples:
                    break

                fname = filenames[i]
                orig_img_path = Path(test_img_dir) / fname

                if orig_img_path.exists():
                    raw_bgr = cv2.imread(str(orig_img_path))
                    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                else:
                    tensor_np = images[i].cpu().numpy().transpose(1, 2, 0)
                    raw_rgb = (np.clip(tensor_np, 0, 1) * 255).astype(np.uint8)

                if roi_only:
                    vis_rgb = create_side_by_side_roi_visualization(
                        image_rgb=raw_rgb,
                        gt_coords=gt_coords[i],
                        pred_coords=pred_coords[i],
                        sample_name=fname,
                        panel_size=panel_size,
                    )
                else:
                    canvas_img = cv2.resize(raw_rgb, (img_size, img_size))
                    vis_rgb = draw_landmarks_on_image(
                        image=canvas_img,
                        pred_coords=pred_coords[i],
                        gt_coords=gt_coords[i],
                        img_size=img_size,
                    )

                out_path = save_dir / f"vis_{saved_count + 1:02d}_{Path(fname).stem}.png"
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), vis_bgr)
                saved_count += 1


def run_evaluation(
    weights_path: str = "./artifacts/best.pth",
    test_img_dir: str = "dataset/test/images",
    test_npz_dir: str = "dataset/test/labels",
    output_dir: str = "evaluation",
    run_name: str | None = None,
    batch_size: int = 8,
    img_size: int = 640,
    num_samples: int = 8,
    threshold_px: float = 2.5,
    device_str: str | None = None,
    log_to_mlflow: bool = True,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    tracking_username: str | None = None,
    tracking_password: str | None = None,
    mlflow_run_name: str | None = None,
    use_tta: bool = False,
    tta_scales: list[float] | tuple[float, ...] = (0.95, 1.0, 1.05),
    tta_shifts: list[tuple[int, int]] | None = None,
    roi_only: bool = True,
    export_json: str | None = None,
    api_url: str | None = None,
    api_token: str | None = None,
) -> tuple[dict, str]:
    """
    Main programmatic evaluation routine.

    Returns:
        summary_metrics: Dictionary of metric values.
        run_folder: Path string of the run subfolder.
    """
    load_dotenv()

    # Configure MLflow Authentication & URI if provided
    user = tracking_username or os.getenv("MLFLOW_TRACKING_USERNAME")
    pwd = tracking_password or os.getenv("MLFLOW_TRACKING_PASSWORD")
    if user:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
    if pwd:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)

    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n" + "=" * 80)
    tta_banner = f" | TTA: ENABLED (scales={list(tta_scales)})" if use_tta else " | TTA: DISABLED"
    print(f"STARTING MODEL EVALUATION | Device: {device}{tta_banner}")
    print(f"=" * 80)

    # 1. Setup output run folder
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    run_folder = Path(output_dir) / run_name
    run_folder.mkdir(parents=True, exist_ok=True)
    vis_folder = run_folder / "visualizations"

    # 2. Load model & test dataset
    model = load_model(weights_path, device, img_size=img_size)
    test_loader = get_test_dataloader(
        test_img_dir=test_img_dir,
        test_npz_dir=test_npz_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=0,  # Safety for cross-platform evaluation
    )

    # Load CVM stage ground truth mapping (API priority with database/file fallbacks)
    stage_map = load_cvm_stage_mapping(
        export_json_path=export_json,
        api_url=api_url,
        api_token=api_token,
    )

    # 3. Aggregate all predictions and ground truth targets
    all_preds = []
    all_gts = []
    all_filenames = []
    latencies_ms = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            gt_coords = batch["coords"].cpu().numpy()
            all_filenames.extend(batch["filename"])

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()

            _, pred_coords = predict_batch(
                model=model,
                images=images,
                use_tta=use_tta,
                tta_scales=tta_scales,
                tta_shifts=tta_shifts,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            batch_sz = images.shape[0]
            latencies_ms.extend([(t1 - t0) * 1000.0 / batch_sz] * batch_sz)

            all_preds.append(pred_coords.cpu().numpy())
            all_gts.append(gt_coords)

    pred_array = np.concatenate(all_preds, axis=0)  # (N, 13, 2)
    gt_array = np.concatenate(all_gts, axis=0)      # (N, 13, 2)

    # 4. Un-Letterboxing to Native Scanner Dimensions & Per-Landmark CSV Rows
    dims_map = {}
    dims_file = ROOT_DIR / "data" / "image_dimensions.json"
    if dims_file.exists():
        try:
            with open(dims_file, "r") as f:
                dims_map = json.load(f)
        except Exception:
            dims_map = {}

    n_samples = len(all_filenames)
    all_orig_dims = []
    unletterboxed_preds_list = []
    unletterboxed_gts_list = []
    pred_stages_list = []
    gt_stages_list = []
    pred_cvm_list = []
    gt_cvm_list = []
    csv_rows = []

    for i in range(n_samples):
        fn = all_filenames[i]
        orig_w, orig_h = None, None
        stem = Path(fn).stem
        if fn in dims_map:
            orig_w = dims_map[fn].get("orig_w")
            orig_h = dims_map[fn].get("orig_h")
        elif f"{stem}.jpg" in dims_map:
            orig_w = dims_map[f"{stem}.jpg"].get("orig_w")
            orig_h = dims_map[f"{stem}.jpg"].get("orig_h")
        elif f"{stem}.png" in dims_map:
            orig_w = dims_map[f"{stem}.png"].get("orig_w")
            orig_h = dims_map[f"{stem}.png"].get("orig_h")

        if orig_w is None or orig_h is None:
            local_p = Path(test_img_dir) / fn
            if local_p.exists():
                try:
                    img_read = cv2.imread(str(local_p))
                    if img_read is not None:
                        orig_h, orig_w = img_read.shape[:2]
                except Exception:
                    pass
        if orig_w is None or orig_h is None:
            orig_w, orig_h = img_size, img_size

        orig_w_int = int(orig_w)
        orig_h_int = int(orig_h)
        all_orig_dims.append((orig_w_int, orig_h_int))

        scale = float(img_size) / max(orig_w_int, orig_h_int)
        new_w = int(round(orig_w_int * scale))
        new_h = int(round(orig_h_int * scale))
        pad_x = (img_size - new_w) // 2
        pad_y = (img_size - new_h) // 2
        scale_x = float(new_w) / float(orig_w_int) if orig_w_int > 0 else 1.0
        scale_y = float(new_h) / float(orig_h_int) if orig_h_int > 0 else 1.0

        p_orig = np.zeros((NUM_LANDMARKS, 2), dtype=float)
        g_orig = np.zeros((NUM_LANDMARKS, 2), dtype=float)

        for k in range(NUM_LANDMARKS):
            p_c_x = pred_array[i, k, 0] * img_size
            p_c_y = pred_array[i, k, 1] * img_size
            p_orig[k, 0] = max(0.0, min(float(orig_w_int), (p_c_x - pad_x) / scale_x))
            p_orig[k, 1] = max(0.0, min(float(orig_h_int), (p_c_y - pad_y) / scale_y))

            if gt_array[i, k, 0] >= 0 and gt_array[i, k, 1] >= 0:
                g_c_x = gt_array[i, k, 0] * img_size
                g_c_y = gt_array[i, k, 1] * img_size
                g_orig[k, 0] = (g_c_x - pad_x) / scale_x
                g_orig[k, 1] = (g_c_y - pad_y) / scale_y
            else:
                g_orig[k, 0] = -1.0
                g_orig[k, 1] = -1.0

        unletterboxed_preds_list.append(p_orig)
        unletterboxed_gts_list.append(g_orig)

        # Anatomical Reference Length (C3 inferior endplate width) in original pixel space
        if g_orig[C3_AI_IDX, 0] >= 0 and g_orig[C3_PI_IDX, 0] >= 0:
            ref_length = math.hypot(
                g_orig[C3_AI_IDX, 0] - g_orig[C3_PI_IDX, 0],
                g_orig[C3_AI_IDX, 1] - g_orig[C3_PI_IDX, 1],
            )
        else:
            ref_length = float(orig_w_int)
        if ref_length <= 1e-4:
            ref_length = float(orig_w_int)

        # Clinical morphometrics & stage evaluation via PercentCVMEngine
        gt_dict = {LANDMARK_CLASSES[k]: (g_orig[k, 0], g_orig[k, 1]) for k in range(NUM_LANDMARKS)}
        pred_dict = {LANDMARK_CLASSES[k]: (p_orig[k, 0], p_orig[k, 1]) for k in range(NUM_LANDMARKS)}

        try:
            gt_cvm = evaluate_landmarks_dict(gt_dict, img_width=orig_w_int, img_height=orig_h_int)
        except Exception:
            gt_cvm = {}

        try:
            pred_cvm = evaluate_landmarks_dict(pred_dict, img_width=orig_w_int, img_height=orig_h_int)
        except Exception:
            pred_cvm = {}

        # Determine predicted stage (from PercentCVMEngine using latest JS rules)
        predicted_stage = pred_cvm.get("stage_number", -1)
        if predicted_stage == -1 and "stage" in pred_cvm:
            try:
                predicted_stage = int(str(pred_cvm["stage"]).replace("CS", ""))
            except (ValueError, TypeError):
                predicted_stage = -1

        # Determine ground-truth clinical stage (API export data priority, fallback to GT geometry)
        true_stage = -1
        fn_stem = Path(fn).stem
        for cand in (fn, fn_stem, str(int(fn_stem)) if fn_stem.isdigit() else None):
            if cand and cand in stage_map:
                true_stage = stage_map[cand]
                break
        if true_stage == -1 or not (1 <= true_stage <= 6):
            true_stage = gt_cvm.get("stage_number", -1)
            if true_stage == -1 and "stage" in gt_cvm:
                try:
                    true_stage = int(str(gt_cvm["stage"]).replace("CS", ""))
                except (ValueError, TypeError):
                    true_stage = -1

        pred_stages_list.append(predicted_stage)
        gt_stages_list.append(true_stage)
        pred_cvm_list.append(pred_cvm)
        gt_cvm_list.append(gt_cvm)

        split = "test"

        # Calculate inferior border projections for this sample (along tangent & perpendicular normal)
        projections = {}
        for vert, lm_map in VERTEBRA_INFERIOR_LANDMARKS.items():
            pi_k = lm_map["PI"]
            ai_k = lm_map["AI"]
            if g_orig[ai_k, 0] >= 0 and g_orig[pi_k, 0] >= 0:
                base_dx = g_orig[ai_k, 0] - g_orig[pi_k, 0]
                base_dy = g_orig[ai_k, 1] - g_orig[pi_k, 1]
                base_len = math.hypot(base_dx, base_dy)
                base_len_safe = base_len if base_len > 1e-4 else 1.0
                u_along_x = base_dx / base_len_safe
                u_along_y = base_dy / base_len_safe
                s = 1.0 if base_dx >= 0 else -1.0
                norm_x = s * base_dy
                norm_y = s * (-base_dx)
                norm_len = math.hypot(norm_x, norm_y)
                norm_len_safe = norm_len if norm_len > 1e-4 else 1.0
                u_perp_x = norm_x / norm_len_safe
                u_perp_y = norm_y / norm_len_safe

                for lm_type, lm_idx in lm_map.items():
                    err_dx = p_orig[lm_idx, 0] - g_orig[lm_idx, 0]
                    err_dy = p_orig[lm_idx, 1] - g_orig[lm_idx, 1]
                    e_along_px = err_dx * u_along_x + err_dy * u_along_y
                    e_perp_px = err_dx * u_perp_x + err_dy * u_perp_y
                    e_along_pct = (e_along_px / ref_length) * 100.0
                    e_perp_pct = (e_perp_px / ref_length) * 100.0
                    projections[lm_idx] = (e_along_px, e_along_pct, e_perp_px, e_perp_pct)

        for k in range(NUM_LANDMARKS):
            lm_name = LANDMARK_CLASSES[k]
            gt_x = g_orig[k, 0]
            gt_y = g_orig[k, 1]
            pred_x = p_orig[k, 0]
            pred_y = p_orig[k, 1]
            dx = pred_x - gt_x
            dy = pred_y - gt_y
            rad_px = math.hypot(dx, dy)
            norm_pct = (rad_px / ref_length) * 100.0
            dx_pct = (dx / ref_length) * 100.0
            dy_pct = (dy / ref_length) * 100.0

            if k in projections:
                e_al_px, e_al_pct, e_pr_px, e_pr_pct = projections[k]
                along_px_str = f"{e_al_px:.4f}"
                along_pct_str = f"{e_al_pct:.4f}"
                perp_px_str = f"{e_pr_px:.4f}"
                perp_pct_str = f"{e_pr_pct:.4f}"
            else:
                along_px_str = ""
                along_pct_str = ""
                perp_px_str = ""
                perp_pct_str = ""

            csv_rows.append([
                fn, split, true_stage, predicted_stage,
                orig_w_int, orig_h_int, pad_x, pad_y, f"{scale_x:.6f}", f"{scale_y:.6f}",
                lm_name, f"{gt_x:.4f}", f"{gt_y:.4f}", f"{pred_x:.4f}", f"{pred_y:.4f}",
                f"{ref_length:.4f}", f"{rad_px:.4f}", f"{norm_pct:.4f}",
                f"{dx:.4f}", f"{dy:.4f}", f"{dx_pct:.4f}", f"{dy_pct:.4f}",
                along_px_str, along_pct_str, perp_px_str, perp_pct_str,
            ])

    # Save predictions_landmarks.csv
    csv_path = run_folder / "predictions_landmarks.csv"
    csv_headers = [
        "image_id", "split", "true_stage", "predicted_stage",
        "orig_w", "orig_h", "pad_x", "pad_y", "scale_x", "scale_y",
        "landmark_name", "gt_x", "gt_y", "pred_x", "pred_y", "ref_length",
        "radial_error_px", "normalized_error_pct",
        "signed_dx_px", "signed_dy_px", "signed_dx_pct", "signed_dy_pct",
        "error_along_px", "error_along_pct", "error_perp_px", "error_perp_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(csv_rows)
    print(f"--> Saved audit-ready test landmark CSV ({len(csv_rows)} rows) to '{csv_path}'")

    # 5. Compute Comprehensive Publication-Grade Evaluation Suite
    pred_morphometrics = {
        "concavities": {
            "C2": [c.get("percentages", {}).get("C2", 0.0) for c in pred_cvm_list],
            "C3": [c.get("percentages", {}).get("C3", 0.0) for c in pred_cvm_list],
            "C4": [c.get("percentages", {}).get("C4", 0.0) for c in pred_cvm_list],
        },
        "shape_indices": {
            "C3": [c.get("shapes", {}).get("C3", {}).get("shape_index", 1.0) for c in pred_cvm_list],
            "C4": [c.get("shapes", {}).get("C4", {}).get("shape_index", 1.0) for c in pred_cvm_list],
        },
        "notches": {
            "C2": [c.get("notches", {}).get("C2", False) for c in pred_cvm_list],
            "C3": [c.get("notches", {}).get("C3", False) for c in pred_cvm_list],
            "C4": [c.get("notches", {}).get("C4", False) for c in pred_cvm_list],
        },
        "shapes": {
            "C3": [c.get("shapes", {}).get("C3", {}).get("shape", "Unknown") for c in pred_cvm_list],
            "C4": [c.get("shapes", {}).get("C4", {}).get("shape", "Unknown") for c in pred_cvm_list],
        },
    }
    gt_morphometrics = {
        "concavities": {
            "C2": [c.get("percentages", {}).get("C2", 0.0) for c in gt_cvm_list],
            "C3": [c.get("percentages", {}).get("C3", 0.0) for c in gt_cvm_list],
            "C4": [c.get("percentages", {}).get("C4", 0.0) for c in gt_cvm_list],
        },
        "shape_indices": {
            "C3": [c.get("shapes", {}).get("C3", {}).get("shape_index", 1.0) for c in gt_cvm_list],
            "C4": [c.get("shapes", {}).get("C4", {}).get("shape_index", 1.0) for c in gt_cvm_list],
        },
        "notches": {
            "C2": [c.get("notches", {}).get("C2", False) for c in gt_cvm_list],
            "C3": [c.get("notches", {}).get("C3", False) for c in gt_cvm_list],
            "C4": [c.get("notches", {}).get("C4", False) for c in gt_cvm_list],
        },
        "shapes": {
            "C3": [c.get("shapes", {}).get("C3", {}).get("shape", "Unknown") for c in gt_cvm_list],
            "C4": [c.get("shapes", {}).get("C4", {}).get("shape", "Unknown") for c in gt_cvm_list],
        },
    }

    unletterboxed_preds_arr = np.array(unletterboxed_preds_list)
    unletterboxed_gts_arr = np.array(unletterboxed_gts_list)

    comprehensive_metrics = compute_test_comprehensive_metrics(
        pred_coords=unletterboxed_preds_arr,
        gt_coords=unletterboxed_gts_arr,
        sample_ids=all_filenames,
        orig_dims=all_orig_dims,
        img_size=img_size,
        pred_stages=pred_stages_list,
        gt_stages=gt_stages_list,
        pred_morphometrics=pred_morphometrics,
        gt_morphometrics=gt_morphometrics,
        n_bootstrap=1000,
    )

    # Normalized & Canvas Metrics
    summary_metrics, landmark_metrics, valid_radial_errors = compute_metrics(
        pred_coords=pred_array,
        gt_coords=gt_array,
        img_size=img_size,
        threshold_px=threshold_px,
    )

    # Attach TTA metadata
    summary_metrics["use_tta"] = use_tta
    if use_tta:
        summary_metrics["tta_scales"] = list(tta_scales)
        if tta_shifts:
            summary_metrics["tta_shifts"] = [list(s) for s in tta_shifts]

    # Attach inference latency metrics
    if latencies_ms:
        latency_info = {
            "mean_ms": float(np.mean(latencies_ms)),
            "median_ms": float(np.median(latencies_ms)),
            "p90_ms": float(np.percentile(latencies_ms, 90.0)),
        }
        summary_metrics["latency_ms_per_image"] = latency_info["mean_ms"]
        comprehensive_metrics["latency_ms_per_image"] = latency_info

    # Generate metric PNG charts
    generate_evaluation_charts(
        summary_metrics=summary_metrics,
        landmark_metrics=landmark_metrics,
        valid_radial_errors=valid_radial_errors,
        save_dir=run_folder / "charts",
        log_to_mlflow=False,
    )

    # Format publication-ready summary report
    report_text = format_summary_report(
        comprehensive_metrics,
        title=f"CVM SWIN-GCN EVALUATION REPORT — {run_name}",
    )
    print("\n" + report_text + "\n")

    # Attach staging metrics to summary_metrics for MLflow logging and reporting
    staging_eval = comprehensive_metrics.get("staging", {})
    if staging_eval and staging_eval.get("valid_samples", 0) > 0:
        summary_metrics["staging"] = staging_eval
        if "overall_accuracy" in staging_eval:
            summary_metrics["staging_accuracy"] = float(staging_eval["overall_accuracy"])
        if "adjacent_accuracy" in staging_eval:
            summary_metrics["staging_adjacent_accuracy"] = float(staging_eval["adjacent_accuracy"])
        if "quadratic_weighted_kappa" in staging_eval:
            summary_metrics["staging_qwk"] = float(staging_eval["quadratic_weighted_kappa"])
        if "unweighted_cohen_kappa" in staging_eval:
            summary_metrics["staging_uwk"] = float(staging_eval["unweighted_cohen_kappa"])
        if "mean_absolute_stage_error" in staging_eval:
            summary_metrics["staging_mase"] = float(staging_eval["mean_absolute_stage_error"])
        if "macro_average" in staging_eval:
            summary_metrics["staging_macro_f1"] = float(staging_eval["macro_average"].get("f1_score", 0.0))

    # 6. Save JSON metrics and summary report TXT
    metrics_export = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "weights_path": weights_path,
        "comprehensive": comprehensive_metrics,
        "summary": summary_metrics,
        "landmarks": landmark_metrics,
    }

    json_path = run_folder / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=4)
    print(f"--> Saved evaluation metrics JSON to '{json_path}'")

    txt_path = run_folder / "summary_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"--> Saved summary report text to '{txt_path}'")

    # 7. Render & save visualization PNGs
    visualize_batch_and_save(
        test_loader=test_loader,
        model=model,
        device=device,
        save_dir=vis_folder,
        num_samples=num_samples,
        img_size=img_size,
        test_img_dir=test_img_dir,
        use_tta=use_tta,
        tta_scales=tta_scales,
        tta_shifts=tta_shifts,
        roi_only=roi_only,
    )

    # 8. MLflow Logging (Active Run or Standalone Run)
    if log_to_mlflow:
        clean_metrics = {
            f"test_{k.replace('@', '').replace('.', '_').replace(' ', '_')}": float(v)
            for k, v in summary_metrics.items()
            if isinstance(v, (int, float))
        }

        def _log_eval_artifacts_in_order(run_id_str: str):
            # Step 1: Log scalar metrics and .json/.txt report FIRST
            mlflow.log_metrics(clean_metrics)
            if (run_folder / "predictions_landmarks.csv").exists():
                mlflow.log_artifact(str(run_folder / "predictions_landmarks.csv"), artifact_path="evaluation")
                print(f"--> Successfully logged 'predictions_landmarks.csv' to MLflow artifact path 'evaluation'.")
            if (run_folder / "metrics.json").exists():
                mlflow.log_artifact(str(run_folder / "metrics.json"), artifact_path="evaluation")
                print(f"--> Successfully logged 'metrics.json' to MLflow artifact path 'evaluation'.")
            if (run_folder / "summary_report.txt").exists():
                mlflow.log_artifact(str(run_folder / "summary_report.txt"), artifact_path="evaluation")

            # Step 2: Upload charts and sample visualizations after .json
            if (run_folder / "charts").exists():
                mlflow.log_artifacts(str(run_folder / "charts"), artifact_path="evaluation/charts")
                print(f"--> Successfully logged evaluation charts to MLflow artifact path 'evaluation/charts'.")
            if (run_folder / "visualizations").exists():
                mlflow.log_artifacts(str(run_folder / "visualizations"), artifact_path="evaluation/visualizations")
                print(f"--> Successfully logged evaluation visualization PNGs to MLflow artifact path 'evaluation/visualizations'.")

        active_run = mlflow.active_run()
        if active_run is not None:
            try:
                _log_eval_artifacts_in_order(active_run.info.run_id)
                print(f"--> Successfully logged evaluation metrics and artifacts to active MLflow run {active_run.info.run_id}.")
            except Exception as ml_err:
                print(f"⚠️ Warning: Could not log evaluation to active MLflow run: {ml_err}")
        elif uri or os.getenv("MLFLOW_TRACKING_URI"):
            try:
                exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "cvm-swin-gcn")
                mlflow.set_experiment(exp_name)
                eval_run_title = mlflow_run_name or f"eval_{run_name}"
                with mlflow.start_run(run_name=eval_run_title) as standalone_run:
                    mlflow.log_params({
                        "eval_weights_path": weights_path,
                        "eval_img_size": img_size,
                        "eval_threshold_px": threshold_px,
                        "eval_batch_size": batch_size,
                        "eval_use_tta": use_tta,
                        "eval_tta_scales": str(list(tta_scales)) if use_tta else "none",
                    })
                    _log_eval_artifacts_in_order(standalone_run.info.run_id)
                    print(f"--> Successfully logged standalone evaluation to MLflow experiment '{exp_name}' (Run ID: {standalone_run.info.run_id}).")
            except Exception as ml_err:
                print(f"⚠️ Warning: Could not log standalone evaluation to MLflow: {ml_err}")

    print(f"\nEvaluation run '{run_name}' completed successfully.")
    print(f"All artifacts saved in directory: '{run_folder}'\n" + "=" * 80)

    return summary_metrics, str(run_folder)


def compute_and_log_split_metrics(
    results_dict: Dict[str, Any],
    labels_dir: Union[str, Path] = "data/labels",
    output_dir: Union[str, Path] = "evaluation",
    export_json_path: Optional[Union[str, Path]] = "data/exports/export.json",
    img_size: int = 640,
    threshold_px: float = 2.5,
    log_to_mlflow: bool = True,
    splits_to_eval: Optional[List[str]] = None,
    save_landmarks_csv: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes full comprehensive evaluation metrics (identical structure to evaluation/metrics.json)
    for specified dataset splits (e.g. 'val', 'train') using whole-data batch model predictions.
    
    Generates:
      ├── metrics_<split>.json
      ├── summary_report_<split>.txt
      └── predictions_landmarks_<split>.csv
      
    And logs scalar metrics & JSON/TXT artifacts directly into the active MLflow experiment.
    """
    splits = splits_to_eval if splits_to_eval is not None else ["val", "train"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load ground truth CVM stages mapping from export JSON if available
    stage_map: Dict[str, int] = {}
    if export_json_path and os.path.exists(export_json_path):
        try:
            with open(export_json_path, "r", encoding="utf-8") as f:
                raw_export = json.load(f)
            tasks = raw_export.get("data", raw_export) if isinstance(raw_export, dict) else raw_export
            for t in tasks:
                t_id = t.get("id") or t.get("task_id") or t.get("taskId")
                fn = t.get("filename") or os.path.basename(
                    t.get("file_upload", "") or t.get("data", {}).get("img", "") or ""
                )
                st = t.get("finalStage") or t.get("stage") or t.get("meta", {}).get("finalStage")
                if st is not None:
                    try:
                        st_num = int(str(st).replace("CS", ""))
                        if fn:
                            stage_map[fn] = st_num
                            stage_map[Path(fn).stem] = st_num
                        if t_id is not None:
                            stage_map[str(t_id)] = st_num
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            print(f"⚠️ Warning loading stage map from {export_json_path}: {e}")

    all_split_metrics: Dict[str, Dict[str, Any]] = {}

    for split_name in splits:
        split_items = [
            (img_id, item) for img_id, item in results_dict.items()
            if item.get("split") == split_name
        ]

        if not split_items:
            print(f"ℹ️ No samples found for split '{split_name}'. Skipping metrics calculation.")
            continue

        print("\n" + "=" * 80)
        print(f"      COMPUTING FULL METRICS FOR SPLIT: {split_name.upper()} ({len(split_items)} samples)      ")
        print("=" * 80)

        unletterboxed_preds_list: List[np.ndarray] = []
        unletterboxed_gts_list: List[np.ndarray] = []
        pred_canvas_list: List[np.ndarray] = []
        gt_canvas_list: List[np.ndarray] = []
        sample_ids: List[str] = []
        orig_dims: List[Tuple[int, int]] = []
        pred_stages_list: List[int] = []
        gt_stages_list: List[int] = []
        pred_cvm_list: List[Dict[str, Any]] = []
        gt_cvm_list: List[Dict[str, Any]] = []
        csv_rows: List[List[Any]] = []

        labels_p = Path(labels_dir)

        for img_id, item in split_items:
            # Candidate paths for ground-truth .npz label file
            npz_candidates = [
                labels_p / f"{img_id}.npz",
                Path("data/labels") / f"{img_id}.npz",
                Path("dataset") / split_name / "labels" / f"{img_id}.npz",
                Path("dataset/test/labels") / f"{img_id}.npz",
            ]
            npz_path = next((p for p in npz_candidates if p.is_file()), None)

            if not npz_path:
                continue

            try:
                npz_data = np.load(npz_path)
                gt_coords_canvas = npz_data["coords"]  # shape (13, 2) in normalized [0, 1]
            except Exception as read_err:
                print(f"⚠️ Warning: Could not read GT label file '{npz_path}': {read_err}")
                continue

            orig_w = int(item.get("image_size", {}).get("width", img_size))
            orig_h = int(item.get("image_size", {}).get("height", img_size))
            orig_dims.append((orig_w, orig_h))
            sample_ids.append(img_id)

            scale = float(img_size) / max(orig_w, orig_h)
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            pad_x = (img_size - new_w) // 2
            pad_y = (img_size - new_h) // 2
            scale_x = float(new_w) / float(orig_w) if orig_w > 0 else 1.0
            scale_y = float(new_h) / float(orig_h) if orig_h > 0 else 1.0

            p_orig = np.zeros((NUM_LANDMARKS, 2), dtype=float)
            g_orig = np.zeros((NUM_LANDMARKS, 2), dtype=float)
            p_canvas = np.zeros((NUM_LANDMARKS, 2), dtype=float)
            g_canvas = np.zeros((NUM_LANDMARKS, 2), dtype=float)

            pred_landmarks_dict = item.get("landmarks", {})

            for k in range(NUM_LANDMARKS):
                lm_name = LANDMARK_CLASSES[k]
                p_pt = pred_landmarks_dict.get(lm_name, [0.0, 0.0])
                p_orig[k, 0] = float(p_pt[0])
                p_orig[k, 1] = float(p_pt[1])

                p_canvas[k, 0] = (p_orig[k, 0] * scale_x + pad_x) / float(img_size)
                p_canvas[k, 1] = (p_orig[k, 1] * scale_y + pad_y) / float(img_size)

                if gt_coords_canvas[k, 0] >= 0 and gt_coords_canvas[k, 1] >= 0:
                    g_c_x = gt_coords_canvas[k, 0] * float(img_size)
                    g_c_y = gt_coords_canvas[k, 1] * float(img_size)
                    g_orig[k, 0] = (g_c_x - pad_x) / scale_x
                    g_orig[k, 1] = (g_c_y - pad_y) / scale_y
                    g_canvas[k, 0] = float(gt_coords_canvas[k, 0])
                    g_canvas[k, 1] = float(gt_coords_canvas[k, 1])
                else:
                    g_orig[k, 0] = -1.0
                    g_orig[k, 1] = -1.0
                    g_canvas[k, 0] = -1.0
                    g_canvas[k, 1] = -1.0

            unletterboxed_preds_list.append(p_orig)
            unletterboxed_gts_list.append(g_orig)
            pred_canvas_list.append(p_canvas)
            gt_canvas_list.append(g_canvas)

            # C3 inferior endplate reference length
            if g_orig[C3_AI_IDX, 0] >= 0 and g_orig[C3_PI_IDX, 0] >= 0:
                ref_length = math.hypot(
                    g_orig[C3_AI_IDX, 0] - g_orig[C3_PI_IDX, 0],
                    g_orig[C3_AI_IDX, 1] - g_orig[C3_PI_IDX, 1],
                )
            else:
                ref_length = float(orig_w)
            if ref_length <= 1e-4:
                ref_length = float(orig_w)

            # Clinical morphometrics & stage evaluation
            gt_dict = {LANDMARK_CLASSES[k]: (g_orig[k, 0], g_orig[k, 1]) for k in range(NUM_LANDMARKS)}
            pred_dict = {LANDMARK_CLASSES[k]: (p_orig[k, 0], p_orig[k, 1]) for k in range(NUM_LANDMARKS)}

            try:
                gt_cvm = evaluate_landmarks_dict(gt_dict, img_width=orig_w, img_height=orig_h)
            except Exception:
                gt_cvm = {}

            try:
                pred_cvm = evaluate_landmarks_dict(pred_dict, img_width=orig_w, img_height=orig_h)
            except Exception:
                pred_cvm = {}

            # Determine predicted clinical stage
            predicted_stage = pred_cvm.get("stage_number", -1)
            if predicted_stage == -1 and "stage" in pred_cvm:
                try:
                    predicted_stage = int(str(pred_cvm["stage"]).replace("CS", ""))
                except (ValueError, TypeError):
                    predicted_stage = -1

            # Determine ground-truth clinical stage
            true_stage = -1
            fn_cand = item.get("filename", f"{img_id}.png")
            for cand in (fn_cand, img_id, str(int(img_id)) if img_id.isdigit() else None):
                if cand and cand in stage_map:
                    true_stage = stage_map[cand]
                    break
            if true_stage == -1 or not (1 <= true_stage <= 6):
                true_stage = gt_cvm.get("stage_number", -1)
                if true_stage == -1 and "stage" in gt_cvm:
                    try:
                        true_stage = int(str(gt_cvm["stage"]).replace("CS", ""))
                    except (ValueError, TypeError):
                        true_stage = -1

            pred_stages_list.append(predicted_stage)
            gt_stages_list.append(true_stage)
            pred_cvm_list.append(pred_cvm)
            gt_cvm_list.append(gt_cvm)

            # Calculate inferior border projections for error breakdown
            projections = {}
            for vert, lm_map in VERTEBRA_INFERIOR_LANDMARKS.items():
                pi_k = lm_map["PI"]
                ai_k = lm_map["AI"]
                if g_orig[ai_k, 0] >= 0 and g_orig[pi_k, 0] >= 0:
                    base_dx = g_orig[ai_k, 0] - g_orig[pi_k, 0]
                    base_dy = g_orig[ai_k, 1] - g_orig[pi_k, 1]
                    base_len = math.hypot(base_dx, base_dy)
                    base_len_safe = base_len if base_len > 1e-4 else 1.0
                    u_along_x = base_dx / base_len_safe
                    u_along_y = base_dy / base_len_safe
                    s = 1.0 if base_dx >= 0 else -1.0
                    norm_x = s * base_dy
                    norm_y = s * (-base_dx)
                    norm_len = math.hypot(norm_x, norm_y)
                    norm_len_safe = norm_len if norm_len > 1e-4 else 1.0
                    u_perp_x = norm_x / norm_len_safe
                    u_perp_y = norm_y / norm_len_safe

                    for lm_type, lm_idx in lm_map.items():
                        err_dx = p_orig[lm_idx, 0] - g_orig[lm_idx, 0]
                        err_dy = p_orig[lm_idx, 1] - g_orig[lm_idx, 1]
                        e_along_px = err_dx * u_along_x + err_dy * u_along_y
                        e_perp_px = err_dx * u_perp_x + err_dy * u_perp_y
                        e_along_pct = (e_along_px / ref_length) * 100.0
                        e_perp_pct = (e_perp_px / ref_length) * 100.0
                        projections[lm_idx] = (e_along_px, e_along_pct, e_perp_px, e_perp_pct)

            for k in range(NUM_LANDMARKS):
                lm_name = LANDMARK_CLASSES[k]
                gt_x = g_orig[k, 0]
                gt_y = g_orig[k, 1]
                pred_x = p_orig[k, 0]
                pred_y = p_orig[k, 1]
                dx = pred_x - gt_x
                dy = pred_y - gt_y
                rad_px = math.hypot(dx, dy)
                norm_pct = (rad_px / ref_length) * 100.0
                dx_pct = (dx / ref_length) * 100.0
                dy_pct = (dy / ref_length) * 100.0

                if k in projections:
                    e_al_px, e_al_pct, e_pr_px, e_pr_pct = projections[k]
                    along_px_str = f"{e_al_px:.4f}"
                    along_pct_str = f"{e_al_pct:.4f}"
                    perp_px_str = f"{e_pr_px:.4f}"
                    perp_pct_str = f"{e_pr_pct:.4f}"
                else:
                    along_px_str, along_pct_str, perp_px_str, perp_pct_str = "", "", "", ""

                csv_rows.append([
                    img_id, split_name, true_stage, predicted_stage,
                    orig_w, orig_h, pad_x, pad_y, f"{scale_x:.6f}", f"{scale_y:.6f}",
                    lm_name, f"{gt_x:.4f}", f"{gt_y:.4f}", f"{pred_x:.4f}", f"{pred_y:.4f}",
                    f"{ref_length:.4f}", f"{rad_px:.4f}", f"{norm_pct:.4f}",
                    f"{dx:.4f}", f"{dy:.4f}", f"{dx_pct:.4f}", f"{dy_pct:.4f}",
                    along_px_str, along_pct_str, perp_px_str, perp_pct_str,
                ])

        if not unletterboxed_preds_list:
            print(f"⚠️ No paired GT labels found for split '{split_name}'. Skipping.")
            continue

        # Format morphometrics structures for comprehensive suite
        pred_morphometrics = {
            "concavities": {
                "C2": [c.get("percentages", {}).get("C2", 0.0) for c in pred_cvm_list],
                "C3": [c.get("percentages", {}).get("C3", 0.0) for c in pred_cvm_list],
                "C4": [c.get("percentages", {}).get("C4", 0.0) for c in pred_cvm_list],
            },
            "shape_indices": {
                "C3": [c.get("shapes", {}).get("C3", {}).get("shape_index", 1.0) for c in pred_cvm_list],
                "C4": [c.get("shapes", {}).get("C4", {}).get("shape_index", 1.0) for c in pred_cvm_list],
            },
            "notches": {
                "C2": [c.get("notches", {}).get("C2", False) for c in pred_cvm_list],
                "C3": [c.get("notches", {}).get("C3", False) for c in pred_cvm_list],
                "C4": [c.get("notches", {}).get("C4", False) for c in pred_cvm_list],
            },
            "shapes": {
                "C3": [c.get("shapes", {}).get("C3", {}).get("shape", "Unknown") for c in pred_cvm_list],
                "C4": [c.get("shapes", {}).get("C4", {}).get("shape", "Unknown") for c in pred_cvm_list],
            },
        }
        gt_morphometrics = {
            "concavities": {
                "C2": [c.get("percentages", {}).get("C2", 0.0) for c in gt_cvm_list],
                "C3": [c.get("percentages", {}).get("C3", 0.0) for c in gt_cvm_list],
                "C4": [c.get("percentages", {}).get("C4", 0.0) for c in gt_cvm_list],
            },
            "shape_indices": {
                "C3": [c.get("shapes", {}).get("C3", {}).get("shape_index", 1.0) for c in gt_cvm_list],
                "C4": [c.get("shapes", {}).get("C4", {}).get("shape_index", 1.0) for c in gt_cvm_list],
            },
            "notches": {
                "C2": [c.get("notches", {}).get("C2", False) for c in gt_cvm_list],
                "C3": [c.get("notches", {}).get("C3", False) for c in gt_cvm_list],
                "C4": [c.get("notches", {}).get("C4", False) for c in gt_cvm_list],
            },
            "shapes": {
                "C3": [c.get("shapes", {}).get("C3", {}).get("shape", "Unknown") for c in gt_cvm_list],
                "C4": [c.get("shapes", {}).get("C4", {}).get("shape", "Unknown") for c in gt_cvm_list],
            },
        }

        unletterboxed_preds_arr = np.array(unletterboxed_preds_list)
        unletterboxed_gts_arr = np.array(unletterboxed_gts_list)
        pred_canvas_arr = np.array(pred_canvas_list)
        gt_canvas_arr = np.array(gt_canvas_list)

        # Compute comprehensive metrics suite
        comprehensive_metrics = compute_test_comprehensive_metrics(
            pred_coords=unletterboxed_preds_arr,
            gt_coords=unletterboxed_gts_arr,
            sample_ids=sample_ids,
            orig_dims=orig_dims,
            img_size=img_size,
            pred_stages=pred_stages_list,
            gt_stages=gt_stages_list,
            pred_morphometrics=pred_morphometrics,
            gt_morphometrics=gt_morphometrics,
            n_bootstrap=500,
        )

        # Compute summary & per-landmark metrics
        summary_metrics, landmark_metrics, valid_radial_errors = compute_metrics(
            pred_coords=pred_canvas_arr,
            gt_coords=gt_canvas_arr,
            img_size=img_size,
            threshold_px=threshold_px,
        )

        # Attach staging metrics to summary_metrics
        staging_eval = comprehensive_metrics.get("staging", {})
        if staging_eval and staging_eval.get("valid_samples", 0) > 0:
            summary_metrics["staging"] = staging_eval
            if "overall_accuracy" in staging_eval:
                summary_metrics["staging_accuracy"] = float(staging_eval["overall_accuracy"])
            if "adjacent_accuracy" in staging_eval:
                summary_metrics["staging_adjacent_accuracy"] = float(staging_eval["adjacent_accuracy"])
            if "quadratic_weighted_kappa" in staging_eval:
                summary_metrics["staging_qwk"] = float(staging_eval["quadratic_weighted_kappa"])
            if "unweighted_cohen_kappa" in staging_eval:
                summary_metrics["staging_uwk"] = float(staging_eval["unweighted_cohen_kappa"])
            if "mean_absolute_stage_error" in staging_eval:
                summary_metrics["staging_mase"] = float(staging_eval["mean_absolute_stage_error"])
            if "macro_average" in staging_eval:
                summary_metrics["staging_macro_f1"] = float(staging_eval["macro_average"].get("f1_score", 0.0))

        # Format summary report text
        report_text = format_summary_report(
            comprehensive_metrics,
            title=f"CVM SWIN-GCN FULL METRICS REPORT — {split_name.upper()} SET",
        )
        print("\n" + report_text + "\n")

        # Build full export dictionary matching evaluation/metrics.json 1:1
        metrics_export = {
            "split": split_name,
            "run_name": f"metrics_{split_name}",
            "timestamp": datetime.now().isoformat(),
            "total_evaluated_samples": len(sample_ids),
            "comprehensive": comprehensive_metrics,
            "summary": summary_metrics,
            "landmarks": landmark_metrics,
        }

        # Save JSON metrics file
        json_file = output_path / f"metrics_{split_name}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(metrics_export, f, indent=4)
        print(f"--> Saved {split_name} metrics JSON to '{json_file}'")

        # Save TXT summary report
        txt_file = output_path / f"summary_report_{split_name}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(f"--> Saved {split_name} summary report to '{txt_file}'")

        # Save CSV rows
        if save_landmarks_csv and csv_rows:
            csv_file = output_path / f"predictions_landmarks_{split_name}.csv"
            csv_headers = [
                "image_id", "split", "true_stage", "predicted_stage",
                "orig_w", "orig_h", "pad_x", "pad_y", "scale_x", "scale_y",
                "landmark_name", "gt_x", "gt_y", "pred_x", "pred_y", "ref_length",
                "radial_error_px", "normalized_error_pct",
                "signed_dx_px", "signed_dy_px", "signed_dx_pct", "signed_dy_pct",
                "error_along_px", "error_along_pct", "error_perp_px", "error_perp_pct",
            ]
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)
                writer.writerows(csv_rows)
            print(f"--> Saved {split_name} landmark CSV ({len(csv_rows)} rows) to '{csv_file}'")

        # Log to active MLflow run if available
        if log_to_mlflow:
            try:
                active_run = mlflow.active_run()
                if active_run:
                    mlflow.log_artifact(str(json_file), artifact_path=f"evaluation/{split_name}")
                    mlflow.log_artifact(str(txt_file), artifact_path=f"evaluation/{split_name}")
                    if (output_path / f"predictions_landmarks_{split_name}.csv").exists():
                        mlflow.log_artifact(
                            str(output_path / f"predictions_landmarks_{split_name}.csv"),
                            artifact_path=f"evaluation/{split_name}",
                        )
                    clean_metrics = {
                        f"{split_name}_{k}": v for k, v in summary_metrics.items()
                        if isinstance(v, (int, float))
                    }
                    mlflow.log_metrics(clean_metrics)
                    print(f"--> [MLflow] Successfully logged {len(clean_metrics)} {split_name} metrics and artifacts.")
            except Exception as ml_err:
                print(f"⚠️ Warning: Could not log {split_name} metrics to MLflow: {ml_err}")

        all_split_metrics[split_name] = metrics_export

    return all_split_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Swin-GCN landmark detection model on test dataset."
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default="./artifacts/best.pth",
        help="Path to trained PyTorch model checkpoint (.pth or .pt).",
    )
    parser.add_argument(
        "--test-img-dir",
        type=str,
        default="dataset/test/images",
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--test-npz-dir",
        type=str,
        default="dataset/test/labels",
        help="Directory containing test label .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        help="Base folder to store evaluation run subfolders.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom subfolder name for this evaluation run (defaults to run_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size for test evaluation.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Target image dimension in pixels (default: 640).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of test image visualizations to save as PNG.",
    )
    parser.add_argument(
        "--threshold-px",
        type=float,
        default=2.5,
        help="Radial error tolerance threshold in pixels for detection metrics.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--tracking-username",
        "--mlflow-username",
        type=str,
        default=None,
        help="MLflow tracking username.",
    )
    parser.add_argument(
        "--tracking-password",
        "--mlflow-password",
        type=str,
        default=None,
        help="MLflow tracking password.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="MLflow run name for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on ('mps', 'cuda', 'cpu'). Auto-selects if not specified.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging during evaluation.",
    )
    parser.add_argument(
        "--use-tta",
        "--tta",
        action="store_true",
        help="Enable Test-Time Augmentation (multi-scale inference) for higher accuracy.",
    )
    parser.add_argument(
        "--tta-scales",
        type=str,
        default="0.95,1.0,1.05",
        help="Comma-separated scale factors for TTA (default: '0.95,1.0,1.05').",
    )
    parser.add_argument(
        "--tta-shifts",
        type=str,
        default=None,
        help="Optional comma-separated pixel shifts in format 'dx:dy,dx:dy' (e.g. '-4:-4,4:4').",
    )
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Render full 640x640 cephalometric image instead of cropped vertebrae ROI side-by-side.",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Path to CVM/Label-Studio export JSON file to load ground-truth CVM stages.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="CVM Backend API URL to fetch test stage cohort ground truth (overrides CVM_API_URL).",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help="CVM Backend API Bearer token (overrides CVM_API_TOKEN).",
    )

    args = parser.parse_args()

    # Parse TTA parameters
    parsed_tta_scales = (
        [float(s.strip()) for s in args.tta_scales.split(",") if s.strip()]
        if args.tta_scales
        else [0.95, 1.0, 1.05]
    )
    parsed_tta_shifts = []
    if args.tta_shifts:
        for pair in args.tta_shifts.split(","):
            if ":" in pair:
                dx, dy = pair.strip().split(":")
                parsed_tta_shifts.append((int(dx), int(dy)))

    run_evaluation(
        weights_path=args.weights,
        test_img_dir=args.test_img_dir,
        test_npz_dir=args.test_npz_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_samples=args.num_samples,
        threshold_px=args.threshold_px,
        device_str=args.device,
        log_to_mlflow=not args.no_mlflow,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        tracking_username=args.tracking_username,
        tracking_password=args.tracking_password,
        mlflow_run_name=args.mlflow_run_name,
        use_tta=args.use_tta,
        tta_scales=parsed_tta_scales,
        tta_shifts=parsed_tta_shifts if len(parsed_tta_shifts) > 0 else None,
        roi_only=not args.full_image,
        export_json=args.export_json,
        api_url=args.api_url,
        api_token=args.api_token,
    )


if __name__ == "__main__":
    main()
