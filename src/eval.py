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
from dotenv import load_dotenv
import mlflow

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import LANDMARK_CLASSES, NUM_LANDMARKS, get_test_dataloader
from src.models.model import CephalometricSwinGCN


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
    pixel_spacing: float | np.ndarray = 0.1,
    threshold_px: float = 2.5,
) -> tuple[dict, list[dict], np.ndarray]:
    """
    Computes overall summary metrics and per-landmark metrics breakdown in both millimeters and pixels.

    Args:
        pred_coords: Array of shape (N, 13, 2) in normalized [0, 1] range.
        gt_coords: Array of shape (N, 13, 2) in normalized [0, 1] range (-1.0 for missing).
        img_size: Image dimension in pixels (default 640).
        pixel_spacing: Physical spacing in mm per pixel (float scalar or 1D array of shape (N,)).
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

    valid_radial_errors = radial_errors[valid_mask]
    valid_abs_dx = abs_dx[valid_mask]
    valid_abs_dy = abs_dy[valid_mask]

    if len(valid_radial_errors) == 0:
        raise ValueError("No valid ground truth landmarks found for metric calculation.")

    # 1. Pixel Regression Metrics
    mae_x = float(np.mean(valid_abs_dx))
    mae_y = float(np.mean(valid_abs_dy))
    mae = float(np.mean((valid_abs_dx + valid_abs_dy) / 2.0))
    rmse = float(np.sqrt(np.mean(dx[valid_mask] ** 2 + dy[valid_mask] ** 2)))
    mre = float(np.mean(valid_radial_errors))
    medre = float(np.median(valid_radial_errors))
    sdre = float(np.std(valid_radial_errors))
    min_err = float(np.min(valid_radial_errors))
    max_err = float(np.max(valid_radial_errors))

    # 2. Physical Millimeter Metrics (Sample-level adaptive or uniform)
    if isinstance(pixel_spacing, np.ndarray):
        radial_errors_mm = radial_errors * pixel_spacing[:, None]
        valid_radial_errors_mm = radial_errors_mm[valid_mask]
        spacing_report_val = float(np.mean(pixel_spacing))
        spacing_mode = "adaptive_sample_level"
    else:
        valid_radial_errors_mm = valid_radial_errors * pixel_spacing
        spacing_report_val = float(pixel_spacing)
        spacing_mode = "uniform_fixed"

    mre_mm = float(np.mean(valid_radial_errors_mm))
    rmse_mm = float(np.sqrt(np.mean(valid_radial_errors_mm ** 2)))
    medre_mm = float(np.median(valid_radial_errors_mm))
    sdre_mm = float(np.std(valid_radial_errors_mm))
    min_err_mm = float(np.min(valid_radial_errors_mm))
    max_err_mm = float(np.max(valid_radial_errors_mm))

    # 3. Clinical SDR Thresholds in Millimeters (Orthodontic Standards)
    sdr_mm_thresholds = [2.0, 2.5, 3.0, 4.0]
    sdr_mm_dict = {}
    for th in sdr_mm_thresholds:
        sdr_val = float(np.mean(valid_radial_errors_mm <= th) * 100.0)
        sdr_mm_dict[f"sdr_{th}mm"] = sdr_val

    # 4. SDR at various radial thresholds in pixels
    sdr_thresholds = [2.0, 2.5, 3.0, 4.0, 5.0, 10.0]
    sdr_dict = {}
    for th in sdr_thresholds:
        sdr_val = float(np.mean(valid_radial_errors <= th) * 100.0)
        sdr_dict[f"sdr_{th}px"] = sdr_val

    total_valid = len(valid_radial_errors)

    summary_metrics = {
        "total_samples": int(pred_coords.shape[0]),
        "total_valid_landmarks": total_valid,
        "pixel_spacing_mm_per_px": spacing_report_val,
        "pixel_spacing_mode": spacing_mode,
        "img_size": img_size,
        # Physical Millimeter Metrics (Primary Clinical)
        "mre_mm": mre_mm,
        "rmse_mm": rmse_mm,
        "medre_mm": medre_mm,
        "sdre_mm": sdre_mm,
        "min_error_mm": min_err_mm,
        "max_error_mm": max_err_mm,
        **sdr_mm_dict,
        # Canvas Pixel Metrics (Secondary)
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

    # 5. Per-landmark Breakdown
    landmark_metrics = []
    for i, name in enumerate(LANDMARK_CLASSES):
        l_mask = valid_mask[:, i]
        if not np.any(l_mask):
            continue

        l_radial = radial_errors[:, i][l_mask]
        l_dx = dx[:, i][l_mask]
        l_dy = dy[:, i][l_mask]
        l_abs_dx = abs_dx[:, i][l_mask]
        l_abs_dy = abs_dy[:, i][l_mask]

        l_mae = float(np.mean((l_abs_dx + l_abs_dy) / 2.0))
        l_rmse = float(np.sqrt(np.mean(l_dx**2 + l_dy**2)))
        l_mre = float(np.mean(l_radial))
        l_medre = float(np.median(l_radial))
        l_sdre = float(np.std(l_radial)) if len(l_radial) > 1 else 0.0

        # Millimeters
        if isinstance(pixel_spacing, np.ndarray):
            l_radial_mm = l_radial * pixel_spacing[l_mask]
        else:
            l_radial_mm = l_radial * pixel_spacing
        l_mre_mm = float(np.mean(l_radial_mm))
        l_rmse_mm = float(np.sqrt(np.mean(l_radial_mm ** 2)))
        l_medre_mm = float(np.median(l_radial_mm))
        l_sdre_mm = float(np.std(l_radial_mm)) if len(l_radial_mm) > 1 else 0.0
        l_sdr2_0mm = float(np.mean(l_radial_mm <= 2.0) * 100.0)
        l_sdr2_5mm = float(np.mean(l_radial_mm <= 2.5) * 100.0)
        l_sdr3_0mm = float(np.mean(l_radial_mm <= 3.0) * 100.0)
        l_sdr4_0mm = float(np.mean(l_radial_mm <= 4.0) * 100.0)

        # Pixels
        l_sdr2_0 = float(np.mean(l_radial <= 2.0) * 100.0)
        l_sdr2_5 = float(np.mean(l_radial <= 2.5) * 100.0)
        l_sdr3_0 = float(np.mean(l_radial <= 3.0) * 100.0)
        l_sdr4_0 = float(np.mean(l_radial <= 4.0) * 100.0)

        landmark_metrics.append({
            "id": i,
            "name": name,
            "count": int(np.sum(l_mask)),
            # Millimeters
            "mre_mm": l_mre_mm,
            "rmse_mm": l_rmse_mm,
            "medre_mm": l_medre_mm,
            "sdre_mm": l_sdre_mm,
            "sdr_2.0mm": l_sdr2_0mm,
            "sdr_2.5mm": l_sdr2_5mm,
            "sdr_3.0mm": l_sdr3_0mm,
            "sdr_4.0mm": l_sdr4_0mm,
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
        })

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
    lines.append(f"Pixel Spacing: {summary.get('pixel_spacing_mm_per_px', 0.1)} mm/px | Image Size: {summary.get('img_size', 640)}x{summary.get('img_size', 640)}")
    lines.append(f"Total Test Images: {summary['total_samples']} | Total Valid Landmarks: {summary['total_valid_landmarks']}")
    lines.append("-" * 96)

    # 1. Physical Millimeter Metrics Table (Primary Clinical)
    lines.append("\n" + "+" + "-" * 50 + "+" + "-" * 43 + "+")
    lines.append(f"| {'CLINICAL METRIC (MILLIMETERS)':<48} | {'VALUE':<41} |")
    lines.append("+" + "-" * 50 + "+" + "-" * 43 + "+")
    
    mm_rows = [
        ("MRE (Mean Radial Error)", f"{summary.get('mre_mm', 0.0):.3f} mm"),
        ("MedRE (Median Radial Error)", f"{summary.get('medre_mm', 0.0):.3f} mm"),
        ("RMSE (Root Mean Squared Error)", f"{summary.get('rmse_mm', 0.0):.3f} mm"),
        ("SDRE (Std Dev Radial Error)", f"{summary.get('sdre_mm', 0.0):.3f} mm"),
        ("Min Radial Error", f"{summary.get('min_error_mm', 0.0):.3f} mm"),
        ("Max Radial Error", f"{summary.get('max_error_mm', 0.0):.3f} mm"),
        ("SDR @ 2.0 mm (Clinical Standard)", f"{summary.get('sdr_2.0mm', 0.0):.2f} %"),
        ("SDR @ 2.5 mm (Challenge Benchmark)", f"{summary.get('sdr_2.5mm', 0.0):.2f} %"),
        ("SDR @ 3.0 mm", f"{summary.get('sdr_3.0mm', 0.0):.2f} %"),
        ("SDR @ 4.0 mm (Coarse Localization)", f"{summary.get('sdr_4.0mm', 0.0):.2f} %"),
    ]
    for label, val in mm_rows:
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
    lines.append(f"| {'PER-LANDMARK CLINICAL PERFORMANCE BREAKDOWN':^94} |")
    lines.append("+" + "-" * 4 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 11 + "+")
    lines.append(f"| {'ID':<2} | {'Landmark':<9} | {'MRE (mm)':<9} | {'RMSE (mm)':<9} | {'SDR@2.0mm':<10} | {'SDR@2.5mm':<10} | {'SDR@4.0mm':<10} | {'MAE(px)':<8} | {'SDR@2.5px':<9} |")
    lines.append("+" + "-" * 4 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 11 + "+")

    for lm in landmarks:
        lines.append(
            f"| {lm['id']:<2} | {lm['name']:<9} | {lm.get('mre_mm', 0.0):<9.2f} | {lm.get('rmse_mm', 0.0):<9.2f} | "
            f"{lm.get('sdr_2.0mm', 0.0):<9.1f}% | {lm.get('sdr_2.5mm', 0.0):<9.1f}% | {lm.get('sdr_4.0mm', 0.0):<9.1f}% | "
            f"{lm['mae_px']:<8.2f} | {lm.get('sdr_2.5px', 0.0):<8.1f}% |"
        )
    lines.append("+" + "-" * 4 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 11 + "+")

    return "\n".join(lines)


def generate_evaluation_charts(
    summary_metrics: dict,
    landmark_metrics: list[dict],
    valid_radial_errors: np.ndarray,
    pixel_spacing: float | np.ndarray = 0.1,
    save_dir: Path | str | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, str]:
    """
    Generates PNG diagram charts for all evaluation metrics (physical mm and canvas px)
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
    eff_spacing = float(np.mean(pixel_spacing)) if isinstance(pixel_spacing, np.ndarray) else float(pixel_spacing)

    try:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.family": "sans-serif"})

        lm_names = [lm["name"] for lm in landmark_metrics]
        lm_indices = np.arange(len(landmark_metrics))
        width = 0.35

        # 1. Landmark Errors Bar Chart (MRE mm vs RMSE mm)
        fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
        mre_mm_vals = [lm.get("mre_mm", lm["mre_px"] * eff_spacing) for lm in landmark_metrics]
        rmse_mm_vals = [lm.get("rmse_mm", lm["rmse_px"] * eff_spacing) for lm in landmark_metrics]

        ax1.bar(lm_indices - width / 2, mre_mm_vals, width, label="MRE (mm)", color="#e76f51", alpha=0.9)
        ax1.bar(lm_indices + width / 2, rmse_mm_vals, width, label="RMSE (mm)", color="#457b9d", alpha=0.9)

        ax1.set_xticks(lm_indices)
        ax1.set_xticklabels(lm_names, rotation=45, ha="right")
        ax1.set_ylabel("Error (Millimeters)")
        ax1.set_title("Per-Landmark Clinical Errors Breakdown (MRE mm vs RMSE mm)", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p1 = target_dir / "chart_landmark_errors.png"
        fig1.savefig(p1, bbox_inches="tight")
        plt.close(fig1)
        generated_charts["landmark_errors"] = str(p1)

        # 2. Per-Landmark SDR Bar Chart (SDR@2.0mm vs SDR@2.5mm vs SDR@4.0mm)
        fig2, ax2 = plt.subplots(figsize=(11, 5), dpi=300)
        w3 = 0.25
        sdr2_0 = [lm.get("sdr_2.0mm", 0.0) for lm in landmark_metrics]
        sdr2_5 = [lm.get("sdr_2.5mm", 0.0) for lm in landmark_metrics]
        sdr4_0 = [lm.get("sdr_4.0mm", 0.0) for lm in landmark_metrics]

        ax2.bar(lm_indices - w3, sdr2_0, w3, label="SDR @ 2.0mm (Clinical Standard)", color="#2a9d8f", alpha=0.9)
        ax2.bar(lm_indices, sdr2_5, w3, label="SDR @ 2.5mm (Challenge Benchmark)", color="#3a86ff", alpha=0.9)
        ax2.bar(lm_indices + w3, sdr4_0, w3, label="SDR @ 4.0mm (Coarse)", color="#8338ec", alpha=0.9)

        ax2.set_xticks(lm_indices)
        ax2.set_xticklabels(lm_names, rotation=45, ha="right")
        ax2.set_ylabel("Detection Rate (%)")
        ax2.set_ylim(0, 105)
        ax2.set_title("Per-Landmark SDR Comparison (2.0mm / 2.5mm / 4.0mm)", fontsize=12, fontweight="bold")
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p2 = target_dir / "chart_landmark_sdr.png"
        fig2.savefig(p2, bbox_inches="tight")
        plt.close(fig2)
        generated_charts["landmark_sdr"] = str(p2)

        # 3. SDR Threshold Curve (Cumulative Detection Rate in mm)
        fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=300)
        th_mm = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        valid_radial_errors_mm = valid_radial_errors * eff_spacing
        sdr_values = [
            summary_metrics.get(f"sdr_{th}mm", float(np.mean(valid_radial_errors_mm <= th) * 100.0))
            for th in th_mm
        ]

        ax3.plot(th_mm, sdr_values, marker="o", linewidth=2.5, color="#ff006e", markersize=8)
        for th, val in zip(th_mm, sdr_values):
            ax3.annotate(f"{val:.1f}%", (th, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

        ax3.set_xlabel("Radius Threshold (Millimeters)")
        ax3.set_ylabel("Successful Detection Rate (%)")
        ax3.set_ylim(0, 108)
        ax3.set_title("Cumulative SDR Curve Across Clinical Distance Thresholds (mm)", fontsize=12, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p3 = target_dir / "chart_sdr_thresholds.png"
        fig3.savefig(p3, bbox_inches="tight")
        plt.close(fig3)
        generated_charts["sdr_thresholds"] = str(p3)

        # 4. Radial Error Distribution Histogram (mm)
        fig4, ax4 = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax4.hist(valid_radial_errors_mm, bins=30, color="#fb8500", edgecolor="black", alpha=0.7, density=True, label="Radial Error (mm)")

        mean_err_mm = summary_metrics.get("mre_mm", np.mean(valid_radial_errors_mm))
        med_err_mm = summary_metrics.get("medre_mm", np.median(valid_radial_errors_mm))

        ax4.axvline(mean_err_mm, color="red", linestyle="--", linewidth=2, label=f"Mean (MRE): {mean_err_mm:.2f} mm")
        ax4.axvline(med_err_mm, color="green", linestyle="-.", linewidth=2, label=f"Median (MedRE): {med_err_mm:.2f} mm")

        ax4.set_xlabel("Radial Error (Millimeters)")
        ax4.set_ylabel("Density")
        ax4.set_title("Radial Error Distribution Across All Test Samples (mm)", fontsize=12, fontweight="bold")
        ax4.legend(loc="upper right")
        ax4.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p4 = target_dir / "chart_radial_error_distribution.png"
        fig4.savefig(p4, bbox_inches="tight")
        plt.close(fig4)
        generated_charts["radial_error_distribution"] = str(p4)

        # 5. Evaluation Dashboard (2x2 Grid)
        fig5, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

        # Subplot (0,0): Landmark Errors
        axes[0, 0].bar(lm_indices - width / 2, mre_mm_vals, width, label="MRE (mm)", color="#e76f51")
        axes[0, 0].bar(lm_indices + width / 2, rmse_mm_vals, width, label="RMSE (mm)", color="#457b9d")
        axes[0, 0].set_xticks(lm_indices)
        axes[0, 0].set_xticklabels(lm_names, rotation=45, ha="right", fontsize=8)
        axes[0, 0].set_ylabel("Error (mm)")
        axes[0, 0].set_title("Landmark Clinical Errors Breakdown (mm)", fontsize=10, fontweight="bold")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

        # Subplot (0,1): Per-Landmark SDR
        axes[0, 1].bar(lm_indices - w3, sdr2_0, w3, label="SDR@2.0mm", color="#2a9d8f")
        axes[0, 1].bar(lm_indices, sdr2_5, w3, label="SDR@2.5mm", color="#3a86ff")
        axes[0, 1].bar(lm_indices + w3, sdr4_0, w3, label="SDR@4.0mm", color="#8338ec")
        axes[0, 1].set_xticks(lm_indices)
        axes[0, 1].set_xticklabels(lm_names, rotation=45, ha="right", fontsize=8)
        axes[0, 1].set_ylabel("SDR (%)")
        axes[0, 1].set_title("Per-Landmark SDR (Clinical mm)", fontsize=10, fontweight="bold")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        # Subplot (1,0): SDR Curve
        axes[1, 0].plot(th_mm, sdr_values, marker="o", color="#ff006e", linewidth=2)
        axes[1, 0].set_title("Cumulative SDR Curve (mm)", fontsize=10, fontweight="bold")
        axes[1, 0].set_xlabel("Threshold (mm)", fontsize=9)
        axes[1, 0].set_ylabel("SDR (%)", fontsize=9)
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

        # Subplot (1,1): Error Distribution
        axes[1, 1].hist(valid_radial_errors_mm, bins=25, color="#fb8500", alpha=0.7, density=True)
        axes[1, 1].axvline(mean_err_mm, color="red", linestyle="--", label=f"MRE: {mean_err_mm:.2f}mm")
        axes[1, 1].set_title("Radial Error Distribution (mm)", fontsize=10, fontweight="bold")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("SWIN-GCN CLINICAL EVALUATION METRICS DASHBOARD", fontsize=14, fontweight="bold")
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
    pixel_spacing: float = 0.1,
    panel_size: int = 560,
) -> np.ndarray:
    """
    Creates an interpretable side-by-side ROI comparison image showing only the cervical vertebrae:
    - Left panel: Ground Truth
    - Right panel: Model Prediction
    - Top banner: Sample name and Mean Radial Error (pixels & mm)
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
        mre_mm = mre_px * pixel_spacing
        metric_str = f"Mean Radial Error: {mre_px:.2f} px ({mre_mm:.2f} mm)"
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
    pixel_spacing: float = 0.1,
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
            spacings = (
                batch["pixel_spacing"].cpu().numpy()
                if "pixel_spacing" in batch
                else np.full(len(filenames), pixel_spacing)
            )

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
                sample_spacing = float(spacings[i]) if hasattr(spacings, "__getitem__") else float(pixel_spacing)

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
                        pixel_spacing=sample_spacing,
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
    pixel_spacing: float = 0.1,
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
        pixel_spacing=pixel_spacing,
        num_workers=0,  # Safety for cross-platform evaluation
    )

    # 3. Aggregate all predictions and ground truth targets
    all_preds = []
    all_gts = []
    all_spacings = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            gt_coords = batch["coords"].cpu().numpy()
            _, pred_coords = predict_batch(
                model=model,
                images=images,
                use_tta=use_tta,
                tta_scales=tta_scales,
                tta_shifts=tta_shifts,
            )

            all_preds.append(pred_coords.cpu().numpy())
            all_gts.append(gt_coords)
            if "pixel_spacing" in batch:
                all_spacings.append(batch["pixel_spacing"].cpu().numpy())

    pred_array = np.concatenate(all_preds, axis=0)  # (N, 13, 2)
    gt_array = np.concatenate(all_gts, axis=0)      # (N, 13, 2)
    if len(all_spacings) > 0:
        spacings_array = np.concatenate(all_spacings, axis=0)
    else:
        spacings_array = pixel_spacing

    # 4. Compute metrics
    summary_metrics, landmark_metrics, valid_radial_errors = compute_metrics(
        pred_coords=pred_array,
        gt_coords=gt_array,
        img_size=img_size,
        pixel_spacing=spacings_array,
        threshold_px=threshold_px,
    )

    # Attach TTA metadata
    summary_metrics["use_tta"] = use_tta
    if use_tta:
        summary_metrics["tta_scales"] = list(tta_scales)
        if tta_shifts:
            summary_metrics["tta_shifts"] = [list(s) for s in tta_shifts]

    # 4b. Generate metric PNG charts
    generate_evaluation_charts(
        summary_metrics=summary_metrics,
        landmark_metrics=landmark_metrics,
        valid_radial_errors=valid_radial_errors,
        pixel_spacing=spacings_array,
        save_dir=run_folder / "charts",
        log_to_mlflow=False,  # We handle comprehensive MLflow logging below
    )

    # 5. Format & print table
    table_text = format_metrics_table(
        summary=summary_metrics,
        landmarks=landmark_metrics,
        run_name=run_name,
        weights_path=weights_path,
    )
    print("\n" + table_text + "\n")

    # 6. Save JSON metrics and summary report TXT
    metrics_export = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "weights_path": weights_path,
        "summary": summary_metrics,
        "landmarks": landmark_metrics,
    }

    json_path = run_folder / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_export, f, indent=4)
    print(f"--> Saved evaluation metrics JSON to '{json_path}'")

    txt_path = run_folder / "summary_report.txt"
    with open(txt_path, "w") as f:
        f.write(table_text + "\n")
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
        pixel_spacing=pixel_spacing,
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
                        "eval_pixel_spacing": pixel_spacing,
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
        "--pixel-spacing",
        type=float,
        default=0.1,
        help="Physical spacing in mm per pixel (default: 0.1 mm/px).",
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
        pixel_spacing=args.pixel_spacing,
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
    )


if __name__ == "__main__":
    main()
