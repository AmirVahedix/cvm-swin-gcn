#!/usr/bin/env python3
"""
Comprehensive Two-Stage Coarse-to-Fine Evaluation Suite for CVM Landmark Detection.

Evaluates and compares:
  Tier 1: Stage 1 Alone (Coarse model on full cephalograms)
  Tier 2: Stage 2 Oracle (Fine model on Ground Truth vertebrae ROIs - theoretical upper bound)
  Tier 3: Two-Stage Cascaded Pipeline (Stage 1 box -> Stage 2 -> Exact Inverse Re-projection)
  Tier 4 (Optional): 2-Pass Refined Cascade (Iterative re-centering)

Generates standardized publication-grade summary reports, metrics.json, prediction CSVs,
side-by-side ROI visualization panels, and Bland-Altman agreement charts.
All errors are normalized to ground-truth C3 inferior endplate width (L_ref).
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from dotenv import load_dotenv
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow

from src.data.constants import (
    LANDMARK_CLASSES,
    NUM_LANDMARKS,
)

C3_PI_IDX = LANDMARK_CLASSES.index("C3_PI")
C3_AI_IDX = LANDMARK_CLASSES.index("C3_AI")
from src.data.loaders.dataset import get_image_dimensions_map
from src.inference import resolve_device
from src.models.model import CephalometricSwinGCN
from src.models.two_stage import TwoStageCephalometricSwinGCN
from src.models.yolo_stage1 import CervicalYOLODetector
from src.utils.cvm_calculator import evaluate_landmarks_dict
from src.utils.metrics import (
    compute_icc_2_1,
    compute_mcnemar_test,
    compute_normalized_errors,
    compute_reference_length,
    compute_staging_metrics,
    compute_test_comprehensive_metrics,
    format_summary_report,
)
from src.utils.roi_utils import (
    ROIBoundingBox,
    apply_anatomical_height_prior,
    canvas_to_native_coords,
    compute_roi_bbox,
    crop_roi_with_padding,
    extract_subpixel_peaks,
    inverse_transform_coords,
    map_bbox_to_native,
    native_to_canvas_coords,
    transform_coords_to_roi,
    validate_stage1_box,
)

LANDMARK_COLORS_BGR = [
    (0, 0, 255),      # 0: C2_PI - Red
    (0, 140, 255),    # 1: C2_IC - Orange
    (0, 215, 255),    # 2: C2_AI - Yellow
    (0, 255, 128),    # 3: C3_PS - Spring Green
    (0, 255, 0),      # 4: C3_AS - Green
    (255, 255, 0),    # 5: C3_PI - Cyan
    (255, 128, 0),    # 6: C3_IC - Sky Blue
    (255, 0, 0),      # 7: C3_AI - Blue
    (128, 0, 255),    # 8: C4_PS - Purple
    (255, 0, 255),    # 9: C4_AS - Magenta
    (255, 0, 128),    # 10: C4_PI - Rose
    (128, 128, 255),  # 11: C4_IC - Light Coral
    (0, 255, 255),    # 12: C4_AI - Mint
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_normalized_tensor(img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Normalizes an RGB uint8 image [H, W, 3] using ImageNet statistics into a float tensor [1, 3, H, W]."""
    norm = ((img_rgb.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)


def load_model_checkpoint(
    weights_path: str,
    img_size: int,
    device: torch.device,
) -> CephalometricSwinGCN:
    """Loads a single CephalometricSwinGCN checkpoint."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    model = CephalometricSwinGCN(num_landmarks=NUM_LANDMARKS, pretrained=False, img_size=img_size)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("stage1.") or k.startswith("stage2."):
            k = k.split(".", 1)[1]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def render_roi_panel_with_contours(
    roi_rgb: np.ndarray,
    coords_norm_roi: np.ndarray,
    panel_size: int = 560,
    title: str = "PREDICTION",
    title_color: Tuple[int, int, int] = (33, 150, 243),
) -> np.ndarray:
    """Renders a single high-resolution ROI panel with vertebra landmark dots, contours, and badges."""
    panel = cv2.resize(roi_rgb, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)

    # 1. Collect valid landmark points
    pts = {}
    for i, pt in enumerate(coords_norm_roi):
        if pt[0] < 0 or pt[1] < 0 or np.isnan(pt[0]) or np.isnan(pt[1]):
            continue
        px = int(np.clip(round(pt[0] * panel_size), 0, panel_size - 1))
        py = int(np.clip(round(pt[1] * panel_size), 0, panel_size - 1))
        pts[i] = (px, py)

    # 2. Draw anatomical contour lines for vertebrae C2, C3, C4
    c2_conns = [(0, 1), (1, 2)]
    c3_conns = [(3, 4), (4, 7), (7, 6), (6, 5), (5, 3)]
    c4_conns = [(8, 9), (9, 12), (12, 11), (11, 10), (10, 8)]

    for u, v in c2_conns + c3_conns + c4_conns:
        if u in pts and v in pts:
            cv2.line(panel, pts[u], pts[v], (180, 180, 180), 1, lineType=cv2.LINE_AA)

    # 3. Draw landmark dots with contrast border
    for i in sorted(pts.keys()):
        px, py = pts[i]
        color_bgr = LANDMARK_COLORS_BGR[i % len(LANDMARK_COLORS_BGR)]
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        cv2.circle(panel, (px, py), 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (px, py), 5, color_rgb, -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (px, py), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # 4. Panel title banner at top
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
    return np.vstack([banner, panel])


def create_three_way_comparison_panel(
    roi_rgb: np.ndarray,
    gt_coords_roi: np.ndarray,
    s1_coords_roi: np.ndarray,
    cascaded_coords_roi: np.ndarray,
    sample_name: str,
    s1_norm_err: float,
    cascade_norm_err: float,
    panel_size: int = 500,
) -> np.ndarray:
    """Renders a 3-panel comparison: Ground Truth vs Stage 1 Coarse vs Two-Stage Cascaded Fine."""
    p_gt = render_roi_panel_with_contours(
        roi_rgb, gt_coords_roi, panel_size=panel_size, title="GROUND TRUTH", title_color=(76, 175, 80)
    )
    p_s1 = render_roi_panel_with_contours(
        roi_rgb, s1_coords_roi, panel_size=panel_size, title=f"STAGE 1 COARSE ({s1_norm_err:.1f}%)", title_color=(255, 152, 0)
    )
    p_casc = render_roi_panel_with_contours(
        roi_rgb, cascaded_coords_roi, panel_size=panel_size, title=f"TWO-STAGE FINE ({cascade_norm_err:.1f}%)", title_color=(33, 150, 243)
    )

    divider = np.full((p_gt.shape[0], 6, 3), 40, dtype=np.uint8)
    combined = np.hstack([p_gt, divider, p_s1, divider, p_casc])

    header_h = 44
    header = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8)
    header[:] = (18, 20, 24)

    delta_pct = cascade_norm_err - s1_norm_err
    delta_str = f"Delta: {delta_pct:+.2f}%"
    cv2.putText(
        header,
        f"Sample: {sample_name} | {delta_str}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        1,
        lineType=cv2.LINE_AA,
    )

    return np.vstack([header, combined])


def run_two_stage_evaluation(
    stage1_weights: str = "artifacts/stage1_best.pth",
    stage2_weights: str = "artifacts/stage2_best.pth",
    img_dir: str = "dataset/test/images",
    npz_dir: str = "dataset/test/labels",
    output_dir: str = "evaluation/two_stage",
    run_name: Optional[str] = None,
    stage1_img_size: int = 640,
    stage2_img_size: int = 448,
    margin_ratio: float = 0.22,
    bottom_margin_ratio: Optional[float] = 0.28,
    use_peaks: bool = True,
    stage1_use_peaks: bool = False,
    refine_2pass: bool = False,
    use_anatomical_prior: bool = False,
    num_visualizations: int = 10,
    device_str: Optional[str] = None,
    native_img_dir: Optional[str] = None,
    log_to_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    tracking_username: Optional[str] = None,
    tracking_password: Optional[str] = None,
    experiment_name: Optional[str] = "cvm-two-stage-cascade-eval",
    run_id: Optional[str] = None,
    stage1_run_id: Optional[str] = None,
    stage2_run_id: Optional[str] = None,
    stage1_artifact_path: str = "weights/best.pt",
    stage2_artifact_path: str = "checkpoints/best.pth",
    export_json: Optional[str] = "data/exports/export.json",
) -> Tuple[Dict[str, Any], str]:
    """
    Main programmatic routine evaluating Stage 1 alone, Stage 2 Oracle, and Two-Stage Cascade.
    """
    load_dotenv()
    device = resolve_device(device_str or "auto")

    # If MLflow run IDs are provided and local files are missing, automatically download
    if stage1_run_id:
        from scripts.download_checkpoint import download_checkpoint_from_mlflow
        ext = Path(stage1_artifact_path).suffix or ".pt"
        target_s1 = Path("artifacts") / f"stage1_{stage1_run_id[:8]}{ext}"
        if not target_s1.exists():
            print(f"--> Automatically downloading Stage 1 checkpoint ('{stage1_artifact_path}') from MLflow run '{stage1_run_id}'...")
            download_checkpoint_from_mlflow(
                run_id=stage1_run_id,
                artifact_path=stage1_artifact_path,
                dest_path=str(target_s1),
                tracking_uri=tracking_uri,
            )
        stage1_weights = str(target_s1)

    if stage2_run_id:
        from scripts.download_checkpoint import download_checkpoint_from_mlflow
        ext = Path(stage2_artifact_path).suffix or ".pth"
        target_s2 = Path("artifacts") / f"stage2_{stage2_run_id[:8]}{ext}"
        if not target_s2.exists():
            print(f"--> Automatically downloading Stage 2 checkpoint ('{stage2_artifact_path}') from MLflow run '{stage2_run_id}'...")
            download_checkpoint_from_mlflow(
                run_id=stage2_run_id,
                artifact_path=stage2_artifact_path,
                dest_path=str(target_s2),
                tracking_uri=tracking_uri,
            )
        stage2_weights = str(target_s2)

    print("=" * 85)
    print(f"STARTING TWO-STAGE CASCADED EVALUATION | Device: {device}")
    print(f"Stage 1 Weights: '{stage1_weights}' ({stage1_img_size}x{stage1_img_size})")
    print(f"Stage 2 Weights: '{stage2_weights}' ({stage2_img_size}x{stage2_img_size})")
    print(f"ROI Margin Ratio: {margin_ratio:.2f} | Bottom Margin: {bottom_margin_ratio if bottom_margin_ratio is not None else margin_ratio:.2f}")
    s1_extract_mode = "Sub-Pixel Peaks" if stage1_use_peaks else "SoftArgmax (Native Spatial Expectation)"
    s2_extract_mode = "Sub-Pixel Peaks (Argmax + Quadratic)" if use_peaks else "SoftArgmax"
    print(f"Stage 1 Extraction: {s1_extract_mode} | Stage 2 Extraction: {s2_extract_mode}")
    print(f"Image Directory: '{img_dir}' | Labels Directory: '{npz_dir}'")
    print("=" * 85)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    actual_run_name = run_name or f"cascade_eval_{timestamp}"
    run_dir = Path(output_dir) / actual_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Load 3-Professor Consensus Voted CVM stage mapping if available
    stage_map: Dict[str, int] = {}
    resolved_export = export_json if (export_json and os.path.exists(export_json)) else ("data/exports/export.json" if os.path.exists("data/exports/export.json") else None)
    if resolved_export:
        try:
            from src.eval import load_cvm_stage_mapping
            stage_map = load_cvm_stage_mapping(export_json_path=resolved_export)
            if stage_map:
                print(f"--> Loaded {len(stage_map)} stage mapping entries for voted finalStage comparison from '{resolved_export}'.")
        except Exception as e:
            print(f"ℹ️ Note: Could not load voted finalStage mapping ({e}).")

    # 1. Load models
    is_yolo_stage1 = stage1_weights.endswith(".pt") or not os.path.exists(stage1_weights)
    if is_yolo_stage1:
        stage1_detector = CervicalYOLODetector(weights=stage1_weights, device=device)
        stage1_model = None
    else:
        try:
            stage1_model = load_model_checkpoint(stage1_weights, stage1_img_size, device)
            stage1_detector = None
        except Exception:
            stage1_detector = CervicalYOLODetector(weights=stage1_weights, device=device)
            stage1_model = None

    stage2_model = load_model_checkpoint(stage2_weights, stage2_img_size, device)

    two_stage_net = TwoStageCephalometricSwinGCN(
        stage1_weights=stage1_weights,
        stage2_img_size=stage2_img_size,
        margin_ratio=margin_ratio,
        pretrained_stage2=False,
        device=device,
    )
    two_stage_net.stage2 = stage2_model
    two_stage_net.to(device)
    two_stage_net.eval()

    # 2. Collect test sample files
    img_files = sorted([
        f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ])
    if not img_files:
        raise FileNotFoundError(f"No valid image files found in '{img_dir}'")

    dims_map = get_image_dimensions_map()

    # Collectors for evaluation
    all_filenames = []
    all_orig_dims = []

    # Unletterboxed coordinates in native radiograph pixel space
    gt_native_list = []
    s1_native_list = []
    oracle_native_list = []
    cascaded_native_list = []

    # Stages and morphometrics
    gt_stages = []
    voted_stages = []
    s1_stages = []
    oracle_stages = []
    cascaded_stages = []

    gt_morph_list = []
    s1_morph_list = []
    oracle_morph_list = []
    cascaded_morph_list = []

    vis_saved = 0
    fallback_count = 0
    s1_latencies_ms = []
    casc_latencies_ms = []

    # Collectors for YOLO Stage 1 ROI Bounding Box Evaluation Metrics
    pred_bboxes_list = []
    gt_bboxes_list = []
    bbox_ious = []
    bbox_center_dists_px = []
    bbox_scale_errors = []

    print(f"\n--> Running inference on {len(img_files)} evaluation samples...")

    for fname in tqdm(img_files, desc="Evaluating Two-Stage Cascade"):
        stem = Path(fname).stem
        npz_path = Path(npz_dir) / f"{stem}.npz"
        if not npz_path.exists():
            continue

        label_data = np.load(npz_path)
        gt_canvas_norm = label_data["coords"].astype(np.float32)  # [13, 2] in [0, 1] canvas

        # Resolve native dimensions
        orig_w = dims_map.get(fname, {}).get("orig_w") or dims_map.get(f"{stem}.jpg", {}).get("orig_w") or dims_map.get(f"{stem}.png", {}).get("orig_w")
        orig_h = dims_map.get(fname, {}).get("orig_h") or dims_map.get(f"{stem}.jpg", {}).get("orig_h") or dims_map.get(f"{stem}.png", {}).get("orig_h")

        img_path = Path(img_dir) / fname
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue
        if orig_w is None or orig_h is None:
            orig_h, orig_w = raw_bgr.shape[:2]

        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        orig_w = int(orig_w)
        orig_h = int(orig_h)

        # Resolve high-resolution native image if available
        native_rgb = None
        if native_img_dir and os.path.exists(native_img_dir):
            for ext in [".png", ".jpg", ".jpeg", ".tif", ".bmp"]:
                cand = Path(native_img_dir) / f"{stem}{ext}"
                if cand.exists():
                    nbgr = cv2.imread(str(cand))
                    if nbgr is not None:
                        native_rgb = cv2.cvtColor(nbgr, cv2.COLOR_BGR2RGB)
                    break

        # Prepare 640x640 canvas tensor
        canvas_img = cv2.resize(raw_rgb, (stage1_img_size, stage1_img_size))
        tensor_in = to_normalized_tensor(canvas_img, device)

        # Helper to crop either from high-res native radiograph or canvas
        def _get_roi_crop(box: ROIBoundingBox) -> np.ndarray:
            if native_rgb is not None and max(orig_w, orig_h) > stage1_img_size:
                native_b = map_bbox_to_native(box, orig_w=orig_w, orig_h=orig_h, canvas_size=stage1_img_size)
                patch, _ = crop_roi_with_padding(
                    image=native_rgb,
                    bbox=native_b,
                    target_size=stage2_img_size,
                    interpolation=cv2.INTER_LINEAR,
                )
                return patch
            else:
                patch, _ = crop_roi_with_padding(
                    image=canvas_img,
                    bbox=box,
                    target_size=stage2_img_size,
                    interpolation=cv2.INTER_LINEAR,
                )
                return patch

        with torch.no_grad():
            # A. Stage 1 Forward
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            t0_s1 = time.perf_counter()

            if stage1_detector is not None:
                # YOLO Stage 1 ROI Detector
                s1_bbox = stage1_detector.predict_roi(
                    image=canvas_img,
                    target_size=stage1_img_size,
                    margin_ratio=0.0,
                )
                s1_coords_canvas = np.full((NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
            else:
                s1_hm, s1_coords_t = stage1_model(tensor_in)
                if stage1_use_peaks:
                    s1_coords_t = extract_subpixel_peaks(s1_hm)
                s1_coords_canvas = s1_coords_t.squeeze(0).cpu().numpy()  # [13, 2]
                s1_bbox = compute_roi_bbox(
                    coords=s1_coords_canvas,
                    img_w=stage1_img_size,
                    img_h=stage1_img_size,
                    margin_ratio=margin_ratio,
                    bottom_margin_ratio=bottom_margin_ratio,
                )

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            t1_s1 = time.perf_counter()
            s1_ms = (t1_s1 - t0_s1) * 1000.0
            s1_latencies_ms.append(s1_ms)

            # B. Stage 2 Oracle Forward (Crop from Ground Truth BBox)
            gt_bbox = compute_roi_bbox(
                coords=gt_canvas_norm,
                img_w=stage1_img_size,
                img_h=stage1_img_size,
                margin_ratio=margin_ratio,
                bottom_margin_ratio=bottom_margin_ratio,
            )

            # Evaluate Stage 1 predicted bounding box against ground-truth vertebra ROI
            pred_b_coords = [float(s1_bbox.u1), float(s1_bbox.v1), float(s1_bbox.u2), float(s1_bbox.v2)]
            gt_b_coords = [float(gt_bbox.u1), float(gt_bbox.v1), float(gt_bbox.u2), float(gt_bbox.v2)]
            pred_bboxes_list.append(pred_b_coords)
            gt_bboxes_list.append(gt_b_coords)

            # Box IoU & Spatial Alignment
            x1 = max(pred_b_coords[0], gt_b_coords[0])
            y1 = max(pred_b_coords[1], gt_b_coords[1])
            x2 = min(pred_b_coords[2], gt_b_coords[2])
            y2 = min(pred_b_coords[3], gt_b_coords[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_p = max(0.0, pred_b_coords[2] - pred_b_coords[0]) * max(0.0, pred_b_coords[3] - pred_b_coords[1])
            area_g = max(0.0, gt_b_coords[2] - gt_b_coords[0]) * max(0.0, gt_b_coords[3] - gt_b_coords[1])
            union = area_p + area_g - inter
            iou = inter / union if union > 0 else 0.0
            bbox_ious.append(iou)

            center_dist = math.hypot(s1_bbox.center_u - gt_bbox.center_u, s1_bbox.center_v - gt_bbox.center_v)
            bbox_center_dists_px.append(center_dist)

            diag_p = math.hypot(pred_b_coords[2] - pred_b_coords[0], pred_b_coords[3] - pred_b_coords[1])
            diag_g = math.hypot(gt_b_coords[2] - gt_b_coords[0], gt_b_coords[3] - gt_b_coords[1])
            scale_err = abs(diag_p - diag_g) / max(diag_g, 1e-6)
            bbox_scale_errors.append(scale_err)
            oracle_patch = _get_roi_crop(gt_bbox)
            oracle_in = to_normalized_tensor(oracle_patch, device)
            oracle_hm, oracle_coords_roi_t = stage2_model(oracle_in)
            if use_peaks:
                oracle_coords_roi_t = extract_subpixel_peaks(oracle_hm)
            oracle_coords_roi = oracle_coords_roi_t.squeeze(0).cpu().numpy()
            oracle_coords_canvas = inverse_transform_coords(
                roi_coords=oracle_coords_roi,
                bbox=gt_bbox,
                target_size=stage2_img_size,
                output_normalized=True,
            )

            # C. Two-Stage Cascaded Forward (Stage 1 BBox -> Stage 2)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            t0_casc = time.perf_counter()

            if s1_bbox.is_fallback:
                fallback_count += 1

            casc_patch = _get_roi_crop(s1_bbox)
            casc_in = to_normalized_tensor(casc_patch, device)
            casc_hm, casc_coords_roi_t = stage2_model(casc_in)
            if use_peaks:
                casc_coords_roi_t = extract_subpixel_peaks(casc_hm)
            casc_coords_roi = casc_coords_roi_t.squeeze(0).cpu().numpy()
            cascaded_coords_canvas = inverse_transform_coords(
                roi_coords=casc_coords_roi,
                bbox=s1_bbox,
                target_size=stage2_img_size,
                output_normalized=True,
            )

            # Optional Pass 2: Iterative re-centering refinement
            if refine_2pass and not s1_bbox.is_fallback:
                s2_bbox = compute_roi_bbox(
                    coords=cascaded_coords_canvas,
                    img_w=stage1_img_size,
                    img_h=stage1_img_size,
                    margin_ratio=margin_ratio,
                    bottom_margin_ratio=bottom_margin_ratio,
                )
                if not s2_bbox.is_fallback:
                    casc_patch_2 = _get_roi_crop(s2_bbox)
                    casc_in_2 = to_normalized_tensor(casc_patch_2, device)
                    casc_hm_2, casc_coords_roi_2_t = stage2_model(casc_in_2)
                    if use_peaks:
                        casc_coords_roi_2_t = extract_subpixel_peaks(casc_hm_2)
                    casc_coords_roi_2 = casc_coords_roi_2_t.squeeze(0).cpu().numpy()
                    cascaded_coords_canvas = inverse_transform_coords(
                        roi_coords=casc_coords_roi_2,
                        bbox=s2_bbox,
                        target_size=stage2_img_size,
                        output_normalized=True,
                    )

            # Safeguard against extreme localized crop drift: if Swin Stage 1 landmarks exist and drift > 50px, revert
            if not np.all(np.isnan(s1_coords_canvas)):
                drift_dist_px = np.linalg.norm(cascaded_coords_canvas - s1_coords_canvas, axis=-1) * float(stage1_img_size)
                cascaded_coords_canvas = np.where(
                    drift_dist_px[:, None] > 50.0,
                    s1_coords_canvas,
                    cascaded_coords_canvas,
                )

            # Apply Anatomical Vertebra Height Prior post-processing safeguard
            if use_anatomical_prior:
                cascaded_coords_canvas, _ = apply_anatomical_height_prior(
                    cascaded_coords_canvas, img_w=stage1_img_size, img_h=stage1_img_size
                )

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            t1_casc = time.perf_counter()
            casc_latencies_ms.append(s1_ms + (t1_casc - t0_casc) * 1000.0)

        # Unletterbox all coordinates from canvas [0, 1] to native radiograph pixel space
        gt_canvas_px = gt_canvas_norm * float(stage1_img_size)
        s1_canvas_px = s1_coords_canvas * float(stage1_img_size)
        oracle_canvas_px = oracle_coords_canvas * float(stage1_img_size)
        casc_canvas_px = cascaded_coords_canvas * float(stage1_img_size)

        gt_native = canvas_to_native_coords(gt_canvas_px, orig_w=orig_w, orig_h=orig_h, canvas_size=stage1_img_size)
        s1_native = canvas_to_native_coords(s1_canvas_px, orig_w=orig_w, orig_h=orig_h, canvas_size=stage1_img_size)
        oracle_native = canvas_to_native_coords(oracle_canvas_px, orig_w=orig_w, orig_h=orig_h, canvas_size=stage1_img_size)
        casc_native = canvas_to_native_coords(casc_canvas_px, orig_w=orig_w, orig_h=orig_h, canvas_size=stage1_img_size)

        all_filenames.append(fname)
        all_orig_dims.append((orig_w, orig_h))
        gt_native_list.append(gt_native)
        s1_native_list.append(s1_native)
        oracle_native_list.append(oracle_native)
        cascaded_native_list.append(casc_native)

        # Morphometrics & Staging evaluation via PercentCVMEngine
        gt_dict = {LANDMARK_CLASSES[k]: (gt_native[k, 0], gt_native[k, 1]) for k in range(NUM_LANDMARKS)}
        s1_dict = {LANDMARK_CLASSES[k]: (s1_native[k, 0], s1_native[k, 1]) for k in range(NUM_LANDMARKS)}
        oracle_dict = {LANDMARK_CLASSES[k]: (oracle_native[k, 0], oracle_native[k, 1]) for k in range(NUM_LANDMARKS)}
        casc_dict = {LANDMARK_CLASSES[k]: (casc_native[k, 0], casc_native[k, 1]) for k in range(NUM_LANDMARKS)}

        try:
            gt_cvm = evaluate_landmarks_dict(gt_dict, img_width=orig_w, img_height=orig_h)
        except Exception:
            gt_cvm = {}
        try:
            s1_cvm = evaluate_landmarks_dict(s1_dict, img_width=orig_w, img_height=orig_h)
        except Exception:
            s1_cvm = {}
        try:
            oracle_cvm = evaluate_landmarks_dict(oracle_dict, img_width=orig_w, img_height=orig_h)
        except Exception:
            oracle_cvm = {}
        try:
            casc_cvm = evaluate_landmarks_dict(casc_dict, img_width=orig_w, img_height=orig_h)
        except Exception:
            casc_cvm = {}

        def _parse_stage(cvm_res: Dict[str, Any]) -> int:
            st = cvm_res.get("stage_number") or cvm_res.get("stage")
            if st is None:
                return -1
            s = str(st).replace("CS", "").strip()
            try:
                iv = int(s)
                return iv if 1 <= iv <= 6 else -1
            except (ValueError, TypeError):
                return -1

        # Determine ground-truth clinical voted stage from stage_map if present
        voted_stg = -1
        for cand in (fname, stem, str(int(stem)) if stem.isdigit() else None):
            if cand and cand in stage_map:
                voted_stg = stage_map[cand]
                break
        voted_stages.append(voted_stg)

        gt_stages.append(_parse_stage(gt_cvm))
        s1_stages.append(_parse_stage(s1_cvm))
        oracle_stages.append(_parse_stage(oracle_cvm))
        cascaded_stages.append(_parse_stage(casc_cvm))

        gt_morph_list.append(gt_cvm)
        s1_morph_list.append(s1_cvm)
        oracle_morph_list.append(oracle_cvm)
        cascaded_morph_list.append(casc_cvm)

        # Save side-by-side ROI visualization panels
        target_visualizations = (
            len(img_files)
            if (num_visualizations is None or num_visualizations < 0)
            else num_visualizations
        )
        if vis_saved < target_visualizations:
            # Map predictions into the ground truth ROI panel space for direct visual comparison
            s1_in_gt_roi = transform_coords_to_roi(s1_coords_canvas, gt_bbox, target_size=stage2_img_size, is_normalized_input=True)
            casc_in_gt_roi = transform_coords_to_roi(cascaded_coords_canvas, gt_bbox, target_size=stage2_img_size, is_normalized_input=True)
            gt_in_gt_roi = transform_coords_to_roi(gt_canvas_norm, gt_bbox, target_size=stage2_img_size, is_normalized_input=True)

            # Compute sample-level error
            ref_len = math.hypot(
                gt_native[C3_AI_IDX, 0] - gt_native[C3_PI_IDX, 0],
                gt_native[C3_AI_IDX, 1] - gt_native[C3_PI_IDX, 1],
            )
            s1_err = float(np.mean(np.sqrt(np.sum((s1_native - gt_native) ** 2, axis=1))) / ref_len * 100.0)
            casc_err = float(np.mean(np.sqrt(np.sum((casc_native - gt_native) ** 2, axis=1))) / ref_len * 100.0)

            vis_panel = create_three_way_comparison_panel(
                roi_rgb=oracle_patch,
                gt_coords_roi=gt_in_gt_roi,
                s1_coords_roi=s1_in_gt_roi,
                cascaded_coords_roi=casc_in_gt_roi,
                sample_name=fname,
                s1_norm_err=s1_err,
                cascade_norm_err=casc_err,
                panel_size=480,
            )
            pad_len = max(2, len(str(len(img_files))))
            out_vis_path = vis_dir / f"vis_{vis_saved + 1:0{pad_len}d}_{stem}.png"
            cv2.imwrite(str(out_vis_path), cv2.cvtColor(vis_panel, cv2.COLOR_RGB2BGR))
            vis_saved += 1

    # Format morphometrics structure for compute_test_comprehensive_metrics
    def build_morph_pack(cvm_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "concavities": {
                "C2": [c.get("percentages", {}).get("C2", 0.0) for c in cvm_list],
                "C3": [c.get("percentages", {}).get("C3", 0.0) for c in cvm_list],
                "C4": [c.get("percentages", {}).get("C4", 0.0) for c in cvm_list],
            },
            "shape_indices": {
                "C3": [c.get("shapes", {}).get("C3", {}).get("shape_index", c.get("shapes", {}).get("C3", {}).get("ratio", 1.0)) for c in cvm_list],
                "C4": [c.get("shapes", {}).get("C4", {}).get("shape_index", c.get("shapes", {}).get("C4", {}).get("ratio", 1.0)) for c in cvm_list],
            },
            "notches": {
                "C2": [c.get("notches", {}).get("C2", False) for c in cvm_list],
                "C3": [c.get("notches", {}).get("C3", False) for c in cvm_list],
                "C4": [c.get("notches", {}).get("C4", False) for c in cvm_list],
            },
            "shapes": {
                "C3": [c.get("shapes", {}).get("C3", {}).get("shape", "Unknown") for c in cvm_list],
                "C4": [c.get("shapes", {}).get("C4", {}).get("shape", "Unknown") for c in cvm_list],
            },
        }

    gt_morph_pack = build_morph_pack(gt_morph_list)
    s1_morph_pack = build_morph_pack(s1_morph_list)
    oracle_morph_pack = build_morph_pack(oracle_morph_list)
    casc_morph_pack = build_morph_pack(cascaded_morph_list)

    gt_arr = np.array(gt_native_list)
    s1_arr = np.array(s1_native_list)
    oracle_arr = np.array(oracle_native_list)
    casc_arr = np.array(cascaded_native_list)

    print("\n--> Computing comprehensive statistical & clinical metrics...")
    metrics_s1 = compute_test_comprehensive_metrics(
        pred_coords=s1_arr,
        gt_coords=gt_arr,
        sample_ids=all_filenames,
        orig_dims=all_orig_dims,
        img_size=stage1_img_size,
        pred_stages=s1_stages,
        gt_stages=gt_stages,
        pred_morphometrics=s1_morph_pack,
        gt_morphometrics=gt_morph_pack,
        n_bootstrap=500,
    )

    metrics_oracle = compute_test_comprehensive_metrics(
        pred_coords=oracle_arr,
        gt_coords=gt_arr,
        sample_ids=all_filenames,
        orig_dims=all_orig_dims,
        img_size=stage1_img_size,
        pred_stages=oracle_stages,
        gt_stages=gt_stages,
        pred_morphometrics=oracle_morph_pack,
        gt_morphometrics=gt_morph_pack,
        n_bootstrap=500,
    )

    metrics_cascade = compute_test_comprehensive_metrics(
        pred_coords=casc_arr,
        gt_coords=gt_arr,
        sample_ids=all_filenames,
        orig_dims=all_orig_dims,
        img_size=stage1_img_size,
        pred_stages=cascaded_stages,
        gt_stages=gt_stages,
        pred_morphometrics=casc_morph_pack,
        gt_morphometrics=gt_morph_pack,
        n_bootstrap=500,
    )

    # 4. Compute Improvement Delta Table & Statistical Significance Tests
    gen_s1 = metrics_s1["general"]
    gen_oracle = metrics_oracle["general"]
    gen_casc = metrics_cascade["general"]

    delta_mean_norm = gen_casc["mean"] - gen_s1["mean"]
    pct_reduction = ((gen_s1["mean"] - gen_casc["mean"]) / gen_s1["mean"]) * 100.0 if gen_s1["mean"] > 0 else 0.0

    delta_med_norm = gen_casc["median"] - gen_s1["median"]
    delta_sdr_2_5 = gen_casc["share_within_2_5pct"] - gen_s1["share_within_2_5pct"]
    delta_sdr_5 = gen_casc["share_within_5pct"] - gen_s1["share_within_5pct"]
    delta_sdr_10 = gen_casc["share_within_10pct"] - gen_s1["share_within_10pct"]

    kappa_s1 = metrics_s1.get("staging", {}).get("quadratic_weighted_kappa", 0.0)
    kappa_casc = metrics_cascade.get("staging", {}).get("quadratic_weighted_kappa", 0.0)
    kappa_oracle = metrics_oracle.get("staging", {}).get("quadratic_weighted_kappa", 0.0)

    acc_s1 = metrics_s1.get("staging", {}).get("overall_accuracy", 0.0)
    acc_casc = metrics_cascade.get("staging", {}).get("overall_accuracy", 0.0)
    acc_oracle = metrics_oracle.get("staging", {}).get("overall_accuracy", 0.0)

    icc_s1 = metrics_s1.get("staging", {}).get("icc_2_1", 0.0)
    icc_casc = metrics_cascade.get("staging", {}).get("icc_2_1", 0.0)
    icc_oracle = metrics_oracle.get("staging", {}).get("icc_2_1", 0.0)
    delta_icc = icc_casc - icc_s1

    # Staging evaluation against 3-Professor Consensus Voted stage if available
    has_voted_stages = sum(1 for s in voted_stages if s > 0) > 0
    metrics_s1_voted: Dict[str, Any] = {}
    metrics_oracle_voted: Dict[str, Any] = {}
    metrics_cascade_voted: Dict[str, Any] = {}
    if has_voted_stages:
        metrics_s1_voted = compute_staging_metrics(
            pred_stages=s1_stages,
            gt_stages=voted_stages,
            n_bootstrap=500,
        )
        metrics_oracle_voted = compute_staging_metrics(
            pred_stages=oracle_stages,
            gt_stages=voted_stages,
            n_bootstrap=500,
        )
        metrics_cascade_voted = compute_staging_metrics(
            pred_stages=cascaded_stages,
            gt_stages=voted_stages,
            n_bootstrap=500,
        )

    # Statistical Significance: Paired Wilcoxon Signed-Rank Test across all landmark errors
    norm_dict_s1 = compute_normalized_errors(s1_arr, gt_arr, img_size=stage1_img_size)
    norm_dict_casc = compute_normalized_errors(casc_arr, gt_arr, img_size=stage1_img_size)
    valid_mask = (gt_arr[:, :, 0] >= 0) & (gt_arr[:, :, 1] >= 0)
    s1_errors = norm_dict_s1["normalized_error_pct"][valid_mask]
    casc_errors = norm_dict_casc["normalized_error_pct"][valid_mask]

    diffs = s1_errors - casc_errors
    if len(diffs) > 0 and not np.all(diffs == 0):
        try:
            wilcoxon_res = scipy.stats.wilcoxon(s1_errors, casc_errors, alternative="two-sided")
            wilcoxon_stat = float(wilcoxon_res.statistic)
            wilcoxon_p = float(wilcoxon_res.pvalue)
        except Exception:
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0
    else:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0

    # Statistical Significance: McNemar's Test for Paired Stage Predictions
    mcnemar_res = compute_mcnemar_test(
        y_true=gt_stages,
        y_pred1=s1_stages,
        y_pred2=cascaded_stages,
    )

    # Compute Stage 1 YOLO Cervical Spine Bounding Box Metrics (for paper reporting)
    bbox_ious_arr = np.array(bbox_ious) if bbox_ious else np.array([0.0])
    stage1_bbox_metrics = {
        "mean_iou": float(np.mean(bbox_ious_arr)),
        "median_iou": float(np.median(bbox_ious_arr)),
        "std_iou": float(np.std(bbox_ious_arr)),
        "iqr_iou": [float(np.percentile(bbox_ious_arr, 25)), float(np.percentile(bbox_ious_arr, 75))],
        "iou_at_50_pct": float(np.mean(bbox_ious_arr >= 0.50) * 100.0),
        "iou_at_75_pct": float(np.mean(bbox_ious_arr >= 0.75) * 100.0),
        "iou_at_85_pct": float(np.mean(bbox_ious_arr >= 0.85) * 100.0),
        "iou_at_90_pct": float(np.mean(bbox_ious_arr >= 0.90) * 100.0),
        "iou_at_95_pct": float(np.mean(bbox_ious_arr >= 0.95) * 100.0),
        "mean_center_distance_px": float(np.mean(bbox_center_dists_px)) if bbox_center_dists_px else 0.0,
        "median_center_distance_px": float(np.median(bbox_center_dists_px)) if bbox_center_dists_px else 0.0,
        "mean_scale_error_pct": float(np.mean(bbox_scale_errors) * 100.0) if bbox_scale_errors else 0.0,
        "median_scale_error_pct": float(np.median(bbox_scale_errors) * 100.0) if bbox_scale_errors else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate_pct": (fallback_count / max(len(all_filenames), 1)) * 100.0,
    }

    # Latency summary (ms per image)
    latency_summary = {
        "stage1_ms": float(np.mean(s1_latencies_ms)) if s1_latencies_ms else 0.0,
        "stage1_ms_median": float(np.median(s1_latencies_ms)) if s1_latencies_ms else 0.0,
        "stage1_ms_iqr": float(np.percentile(s1_latencies_ms, 75.0) - np.percentile(s1_latencies_ms, 25.0)) if s1_latencies_ms else 0.0,
        "stage1_ms_p90": float(np.percentile(s1_latencies_ms, 90.0)) if s1_latencies_ms else 0.0,
        "cascade_ms": float(np.mean(casc_latencies_ms)) if casc_latencies_ms else 0.0,
        "cascade_ms_median": float(np.median(casc_latencies_ms)) if casc_latencies_ms else 0.0,
        "cascade_ms_iqr": float(np.percentile(casc_latencies_ms, 75.0) - np.percentile(casc_latencies_ms, 25.0)) if casc_latencies_ms else 0.0,
        "cascade_ms_p90": float(np.percentile(casc_latencies_ms, 90.0)) if casc_latencies_ms else 0.0,
        "total_images": len(s1_latencies_ms),
    }

    delta_summary = {
        "s1_mean_norm_err": gen_s1["mean"],
        "cascade_mean_norm_err": gen_casc["mean"],
        "oracle_mean_norm_err": gen_oracle["mean"],
        "delta_mean_norm_err": delta_mean_norm,
        "relative_error_reduction_pct": pct_reduction,
        "s1_median_norm_err": gen_s1["median"],
        "cascade_median_norm_err": gen_casc["median"],
        "oracle_median_norm_err": gen_oracle["median"],
        "delta_median_norm_err": delta_med_norm,
        "delta_share_within_2_5pct": delta_sdr_2_5,
        "delta_share_within_5pct": delta_sdr_5,
        "delta_share_within_10pct": delta_sdr_10,
        "s1_quadratic_kappa": kappa_s1,
        "cascade_quadratic_kappa": kappa_casc,
        "oracle_quadratic_kappa": kappa_oracle,
        "delta_quadratic_kappa": kappa_casc - kappa_s1,
        "s1_icc_2_1": icc_s1,
        "cascade_icc_2_1": icc_casc,
        "oracle_icc_2_1": icc_oracle,
        "delta_icc_2_1": delta_icc,
        "s1_staging_acc": acc_s1,
        "cascade_staging_acc": acc_casc,
        "oracle_staging_acc": acc_oracle,
        "delta_staging_acc": acc_casc - acc_s1,
        "stage1_ms": latency_summary["stage1_ms"],
        "cascade_ms": latency_summary["cascade_ms"],
        "delta_ms": latency_summary["cascade_ms"] - latency_summary["stage1_ms"],
        "wilcoxon_signed_rank_stat": wilcoxon_stat,
        "wilcoxon_signed_rank_p": wilcoxon_p,
        "mcnemar_b": mcnemar_res["b"],
        "mcnemar_c": mcnemar_res["c"],
        "mcnemar_p_value": mcnemar_res["p_value"],
        "fallback_count": fallback_count,
        "fallback_rate_pct": (fallback_count / len(all_filenames)) * 100.0 if all_filenames else 0.0,
    }

    # 5. Format Summary Report
    report_cascade_text = format_summary_report(
        metrics_cascade, title=f"TWO-STAGE CASCADED PIPELINE EVALUATION REPORT — {actual_run_name}"
    )

    voted_report_block = ""
    voted_comparison_rows = ""
    if has_voted_stages and metrics_cascade_voted.get("valid_samples", 0) > 0:
        v_stg = metrics_cascade_voted
        w_ci = v_stg.get("accuracy_ci_95_wilson", {})
        b_ci = v_stg.get("accuracy_ci_95_bootstrap", {})
        adj_w = v_stg.get("adjacent_accuracy_ci_95_wilson", {})
        adj_b = v_stg.get("adjacent_accuracy_ci_95_bootstrap", {})
        qwk_ci = v_stg.get("quadratic_weighted_kappa_ci_95", {})
        uwk_ci = v_stg.get("unweighted_cohen_kappa_ci_95", {})
        icc_ci = v_stg.get("icc_2_1_ci_95", {})

        v_lines = [
            "\n" + "=" * 88,
            "     SECTION D.2: CLINICAL CONSENSUS STAGING EVALUATION (3-PROFESSOR FINAL VOTED STAGE) ",
            "=" * 88,
            f" Total Evaluated Staging Samples : {v_stg.get('valid_samples', 0)} / {v_stg.get('total_samples', 0)}",
            f" Overall Accuracy vs Voted Stage : {v_stg.get('overall_accuracy', 0.0):.2f}%",
            f"   • 95% Wilson CI               : [{w_ci.get('lower', 0.0):.2f}%, {w_ci.get('upper', 0.0):.2f}%]",
            f"   • 95% Bootstrap CI            : [{b_ci.get('lower', 0.0):.2f}%, {b_ci.get('upper', 0.0):.2f}%]",
            f" Adjacent-Stage Accuracy (±1)    : {v_stg.get('adjacent_accuracy', 0.0):.2f}%",
            f"   • 95% Wilson CI               : [{adj_w.get('lower', 0.0):.2f}%, {adj_w.get('upper', 0.0):.2f}%]",
            f"   • 95% Bootstrap CI            : [{adj_b.get('lower', 0.0):.2f}%, {adj_b.get('upper', 0.0):.2f}%]",
            f" Mean Absolute Stage Error (MASE): {v_stg.get('mean_absolute_stage_error', 0.0):.3f} stages",
            "",
            "Inter-Rater Reliability Agreement with Consensus Panel:",
            f" • Quadratic Weighted Kappa (κw) : {v_stg.get('quadratic_weighted_kappa', 0.0):.3f} (95% CI: [{qwk_ci.get('lower', 0.0):.3f}, {qwk_ci.get('upper', 0.0):.3f}])",
            f" • Unweighted Cohen's Kappa (κ)  : {v_stg.get('unweighted_cohen_kappa', 0.0):.3f} (95% CI: [{uwk_ci.get('lower', 0.0):.3f}, {uwk_ci.get('upper', 0.0):.3f}])",
            f" • ICC(2,1) Absolute Agreement   : {v_stg.get('icc_2_1', 0.0):.3f} (95% CI: [{icc_ci.get('lower', 0.0):.3f}, {icc_ci.get('upper', 0.0):.3f}])",
        ]

        cm = v_stg.get("confusion_matrix", [])
        if len(cm) == 6:
            v_lines.append("\n6×6 Confusion Matrix (Rows: 3-Prof Consensus, Columns: Predicted):")
            v_lines.append("  True \\ Pred |   CS1   CS2   CS3   CS4   CS5   CS6 | Total | Recall (%)")
            v_lines.append("  " + "-" * 66)
            col_totals = [0] * 6
            for r_idx, row in enumerate(cm):
                row_sum = sum(row)
                row_rec = (row[r_idx] / row_sum * 100.0) if row_sum > 0 else 0.0
                for c_idx in range(6):
                    col_totals[c_idx] += row[c_idx]
                row_str = " ".join(f"{val:>5d}" for val in row)
                v_lines.append(f"  CS{r_idx+1:<8} | {row_str} | {row_sum:>5d} | {row_rec:>9.2f}%")
            v_lines.append("  " + "-" * 66)
            col_str = " ".join(f"{val:>5d}" for val in col_totals)
            v_lines.append(f"  Total       | {col_str} | {sum(col_totals):>5d} |")
            prec_vals = []
            for c_idx in range(6):
                p_v = (cm[c_idx][c_idx] / col_totals[c_idx] * 100.0) if col_totals[c_idx] > 0 else 0.0
                prec_vals.append(f"{p_v:>5.1f}%")
            v_lines.append(f"  Precision   | {' '.join(prec_vals)} |")

        per_stage = v_stg.get("per_stage", [])
        macro = v_stg.get("macro_average", {})
        if per_stage:
            v_lines.append("\nPer-Stage Classification Performance vs Consensus:")
            v_lines.append("  Stage | Precision (%) | Recall (%) | F1-Score (%) | Support")
            v_lines.append("  " + "-" * 56)
            for row in per_stage:
                v_lines.append(
                    f"  {row.get('stage', ''):<5} | {row.get('precision', 0.0):>13.2f}% | {row.get('recall', 0.0):>10.2f}% | {row.get('f1_score', 0.0):>12.2f}% | {row.get('support', 0):>7d}"
                )
            v_lines.append("  " + "-" * 56)
            v_lines.append(
                f"  Macro | {macro.get('precision', 0.0):>13.2f}% | {macro.get('recall', 0.0):>10.2f}% | {macro.get('f1_score', 0.0):>12.2f}% | {v_stg.get('valid_samples', 0):>7d}"
            )
        v_lines.append("=" * 88 + "\n")
        voted_report_block = "\n".join(v_lines)

        acc_s1_v = metrics_s1_voted.get("overall_accuracy", 0.0)
        acc_orc_v = metrics_oracle_voted.get("overall_accuracy", 0.0)
        acc_casc_v = metrics_cascade_voted.get("overall_accuracy", 0.0)
        kappa_s1_v = metrics_s1_voted.get("quadratic_weighted_kappa", 0.0)
        kappa_orc_v = metrics_oracle_voted.get("quadratic_weighted_kappa", 0.0)
        kappa_casc_v = metrics_cascade_voted.get("quadratic_weighted_kappa", 0.0)
        voted_comparison_rows = (
            f" CVM Staging (Consensus GT)  | {acc_s1_v:6.2f}%        | {acc_orc_v:6.2f}%        | {acc_casc_v:6.2f}%            | {acc_casc_v - acc_s1_v:+6.2f}%\n"
            f" κw Kappa (Consensus GT)     | {kappa_s1_v:6.3f}         | {kappa_orc_v:6.3f}         | {kappa_casc_v:6.3f}             | {kappa_casc_v - kappa_s1_v:+6.3f}\n"
        )

    comparison_table = (
        "\n" + "=" * 88 + "\n"
        + "                  SECTION E: TWO-STAGE CASCADED IMPROVEMENT COMPARISON                  \n"
        + "=" * 88 + "\n"
        + f" Total Samples Evaluated    : {len(all_filenames)}\n"
        + f" BBox Sanity Fallback Count : {fallback_count} ({delta_summary['fallback_rate_pct']:.2f}%)\n"
        + f" Inference Latency          : Stage 1 = {latency_summary['stage1_ms']:.1f} ms | Cascade = {latency_summary['cascade_ms']:.1f} ms\n"
        + f" Paired Wilcoxon Test (N={len(s1_errors)}) : W = {wilcoxon_stat:.1f}, P = {wilcoxon_p:.4e}\n"
        + f" McNemar's Test (N={len(gt_stages)})    : discordant (b={mcnemar_res['b']}, c={mcnemar_res['c']}), P = {mcnemar_res['p_value']:.4f}\n"
        + "----------------------------------------------------------------------------------------\n"
        + " [STAGE 1 YOLO ROI LOCALIZATION PERFORMANCE]\n"
        + f" Mean Box IoU               : {stage1_bbox_metrics['mean_iou']:.4f} ± {stage1_bbox_metrics['std_iou']:.4f} (Median: {stage1_bbox_metrics['median_iou']:.4f})\n"
        + f" Success Rate at IoU ≥ 0.50 : {stage1_bbox_metrics['iou_at_50_pct']:.2f}%\n"
        + f" Success Rate at IoU ≥ 0.75 : {stage1_bbox_metrics['iou_at_75_pct']:.2f}%\n"
        + f" Success Rate at IoU ≥ 0.85 : {stage1_bbox_metrics['iou_at_85_pct']:.2f}%\n"
        + f" Success Rate at IoU ≥ 0.90 : {stage1_bbox_metrics['iou_at_90_pct']:.2f}%\n"
        + f" Mean Center Distance Error : {stage1_bbox_metrics['mean_center_distance_px']:.2f} px (Median: {stage1_bbox_metrics['median_center_distance_px']:.2f} px)\n"
        + f" Mean Scale / Diagonal Err  : {stage1_bbox_metrics['mean_scale_error_pct']:.2f}%\n"
        + "----------------------------------------------------------------------------------------\n"
        + " Metric                     | Stage 1 Coarse | Stage 2 Oracle | Two-Stage Cascade | Delta (Cascade - S1) \n"
        "----------------------------------------------------------------------------------------\n"
        f" Mean Normalized Error (%)  | {gen_s1['mean']:6.2f}%        | {gen_oracle['mean']:6.2f}%        | {gen_casc['mean']:6.2f}%            | {delta_mean_norm:+6.2f}% ({pct_reduction:+.1f}% rel)\n"
        f" Median Norm Error (%)      | {gen_s1['median']:6.2f}%        | {gen_oracle['median']:6.2f}%        | {gen_casc['median']:6.2f}%            | {delta_med_norm:+6.2f}%\n"
        f" Share ≤ 2.5% L_ref         | {gen_s1['share_within_2_5pct']:6.2f}%        | {gen_oracle['share_within_2_5pct']:6.2f}%        | {gen_casc['share_within_2_5pct']:6.2f}%            | {delta_sdr_2_5:+6.2f}%\n"
        f" Share ≤ 5.0% L_ref         | {gen_s1['share_within_5pct']:6.2f}%        | {gen_oracle['share_within_5pct']:6.2f}%        | {gen_casc['share_within_5pct']:6.2f}%            | {delta_sdr_5:+6.2f}%\n"
        f" Share ≤ 10.0% L_ref        | {gen_s1['share_within_10pct']:6.2f}%        | {gen_oracle['share_within_10pct']:6.2f}%        | {gen_casc['share_within_10pct']:6.2f}%            | {delta_sdr_10:+6.2f}%\n"
        f" Share ≤ 11.5% (2.0mm Clin) | {gen_s1.get('share_within_11_5pct', 0.0):6.2f}%        | {gen_oracle.get('share_within_11_5pct', 0.0):6.2f}%        | {gen_casc.get('share_within_11_5pct', 0.0):6.2f}%            | {gen_casc.get('share_within_11_5pct', 0.0) - gen_s1.get('share_within_11_5pct', 0.0):+6.2f}%\n"
        f" Share ≤ 15.0% L_ref        | {gen_s1.get('share_within_15pct', 0.0):6.2f}%        | {gen_oracle.get('share_within_15pct', 0.0):6.2f}%        | {gen_casc.get('share_within_15pct', 0.0):6.2f}%            | {gen_casc.get('share_within_15pct', 0.0) - gen_s1.get('share_within_15pct', 0.0):+6.2f}%\n"
        f" 90th Percentile (P90)      | {gen_s1['p90']:6.2f}%        | {gen_oracle['p90']:6.2f}%        | {gen_casc['p90']:6.2f}%            | {gen_casc['p90'] - gen_s1['p90']:+6.2f}%\n"
        f" CVM Staging (Geometric GT) | {acc_s1:6.2f}%        | {acc_oracle:6.2f}%        | {acc_casc:6.2f}%            | {acc_casc - acc_s1:+6.2f}%\n"
        + voted_comparison_rows +
        f" Quadratic Weighted Kappa   | {kappa_s1:6.3f}         | {kappa_oracle:6.3f}         | {kappa_casc:6.3f}             | {kappa_casc - kappa_s1:+6.3f}\n"
        f" ICC(2,1) Agreement         | {icc_s1:6.3f}         | {icc_oracle:6.3f}         | {icc_casc:6.3f}             | {delta_icc:+6.3f}\n"
        f" Inference Latency (ms)     | {latency_summary['stage1_ms']:6.1f} ms      |    N/A         | {latency_summary['cascade_ms']:6.1f} ms         | {latency_summary['cascade_ms'] - latency_summary['stage1_ms']:+6.1f} ms\n"
        "========================================================================================\n"
    )

    full_report_text = report_cascade_text + voted_report_block + comparison_table

    # Save summary report text
    report_file = run_dir / "summary_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(full_report_text)
    print(full_report_text)

    # 6. Save JSON Metrics
    combined_metrics_payload = {
        "timestamp": timestamp,
        "run_name": actual_run_name,
        "stage1_weights": stage1_weights,
        "stage2_weights": stage2_weights,
        "stage1_img_size": stage1_img_size,
        "stage2_img_size": stage2_img_size,
        "margin_ratio": margin_ratio,
        "delta_comparison": delta_summary,
        "stage1_bbox_metrics": stage1_bbox_metrics,
        "inference_latency": latency_summary,
        "statistical_tests": {
            "paired_wilcoxon": {
                "n_landmarks": len(s1_errors),
                "statistic": wilcoxon_stat,
                "p_value": wilcoxon_p,
            },
            "mcnemar": mcnemar_res,
        },
        "stage1_alone": metrics_s1,
        "stage2_oracle": metrics_oracle,
        "two_stage_cascade": metrics_cascade,
        "voted_staging": {
            "has_voted_data": has_voted_stages,
            "stage1_alone": metrics_s1_voted,
            "stage2_oracle": metrics_oracle_voted,
            "two_stage_cascade": metrics_cascade_voted,
        },
    }

    metrics_json_file = run_dir / "metrics.json"
    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump(combined_metrics_payload, f, indent=2)

    # 7. Save Predictions CSVs
    csv_file = run_dir / "predictions_landmarks.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["sample_id", "landmark_index", "landmark_name"]
        header += ["gt_x_px", "gt_y_px", "s1_x_px", "s1_y_px", "cascade_x_px", "cascade_y_px"]
        header += ["s1_radial_err_px", "cascade_radial_err_px", "s1_norm_err_pct", "cascade_norm_err_pct"]
        writer.writerow(header)

        for s_idx, fname in enumerate(all_filenames):
            ref_l = math.hypot(
                gt_arr[s_idx, C3_AI_IDX, 0] - gt_arr[s_idx, C3_PI_IDX, 0],
                gt_arr[s_idx, C3_AI_IDX, 1] - gt_arr[s_idx, C3_PI_IDX, 1],
            )
            for l_idx in range(NUM_LANDMARKS):
                g_x, g_y = gt_arr[s_idx, l_idx]
                if g_x < 0 or g_y < 0:
                    continue
                s1_x, s1_y = s1_arr[s_idx, l_idx]
                c_x, c_y = casc_arr[s_idx, l_idx]
                s1_r = math.hypot(s1_x - g_x, s1_y - g_y)
                c_r = math.hypot(c_x - g_x, c_y - g_y)
                writer.writerow([
                    fname,
                    l_idx,
                    LANDMARK_CLASSES[l_idx],
                    f"{g_x:.2f}", f"{g_y:.2f}",
                    f"{s1_x:.2f}", f"{s1_y:.2f}",
                    f"{c_x:.2f}", f"{c_y:.2f}",
                    f"{s1_r:.2f}", f"{c_r:.2f}",
                    f"{(s1_r / ref_l * 100.0):.2f}",
                    f"{(c_r / ref_l * 100.0):.2f}",
                ])

    staging_csv_file = run_dir / "predictions_staging.csv"
    with open(staging_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "gt_geom_stage",
            "voted_prof_stage",
            "s1_stage",
            "oracle_stage",
            "cascade_stage",
            "cascade_matches_geom",
            "cascade_matches_voted",
        ])
        for idx, fn in enumerate(all_filenames):
            g_st = gt_stages[idx]
            v_st = voted_stages[idx] if has_voted_stages else -1
            s1_st = s1_stages[idx]
            orc_st = oracle_stages[idx]
            cas_st = cascaded_stages[idx]
            writer.writerow([
                fn,
                f"CS{g_st}" if g_st > 0 else "N/A",
                f"CS{v_st}" if v_st > 0 else "N/A",
                f"CS{s1_st}" if s1_st > 0 else "N/A",
                f"CS{orc_st}" if orc_st > 0 else "N/A",
                f"CS{cas_st}" if cas_st > 0 else "N/A",
                (cas_st == g_st) if g_st > 0 else "N/A",
                (cas_st == v_st) if v_st > 0 else "N/A",
            ])

    # 8. Save structured Predictions JSON (predicted landmarks only) for Web/GUI applications
    predictions_json_file = run_dir / "predictions_landmarks.json"
    predictions_dict: Dict[str, Any] = {}
    for idx, fn in enumerate(all_filenames):
        orig_w, orig_h = all_orig_dims[idx]
        casc_coords_native = casc_arr[idx]  # [13, 2] in native px

        sample_entry: Dict[str, Any] = {
            "image_size": [int(orig_w), int(orig_h)],
            "landmarks_pixel": {},
            "landmarks_normalized": {},
        }

        for l_idx, lm_name in enumerate(LANDMARK_CLASSES):
            px_x, px_y = float(casc_coords_native[l_idx, 0]), float(casc_coords_native[l_idx, 1])
            sample_entry["landmarks_pixel"][lm_name] = [round(px_x, 2), round(px_y, 2)]
            sample_entry["landmarks_normalized"][lm_name] = [
                round(px_x / max(orig_w, 1), 5),
                round(px_y / max(orig_h, 1), 5),
            ]

        predictions_dict[fn] = sample_entry

    with open(predictions_json_file, "w", encoding="utf-8") as f:
        json.dump(predictions_dict, f, indent=2)

    print(f"\n--> Successfully saved evaluation artifacts to '{run_dir}':")
    print(f"    • Report: {report_file.name}")
    print(f"    • Metrics: {metrics_json_file.name}")
    print(f"    • Predictions JSON: {predictions_json_file.name}")
    print(f"    • Landmarks CSV: {csv_file.name}")
    print(f"    • Staging CSV: {staging_csv_file.name}")
    print(f"    • Visualizations: {vis_saved} panels in '{vis_dir.name}'")

    if log_to_mlflow:
        try:
            user = tracking_username or os.getenv("MLFLOW_TRACKING_USERNAME")
            pwd = tracking_password or os.getenv("MLFLOW_TRACKING_PASSWORD")
            if user:
                os.environ["MLFLOW_TRACKING_USERNAME"] = user
            if pwd:
                os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd

            uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
            if uri:
                mlflow.set_tracking_uri(uri)
            exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME") or "cvm-two-stage-cascade-eval"
            mlflow.set_experiment(exp_name)

            if run_id:
                run_context = mlflow.start_run(run_id=run_id)
            else:
                run_name_to_use = actual_run_name if actual_run_name.startswith("eval_") else f"eval_{actual_run_name}"
                run_context = mlflow.start_run(run_name=run_name_to_use)

            with run_context:
                mlflow.log_params({
                    "stage1_weights": stage1_weights,
                    "stage2_weights": stage2_weights,
                    "margin_ratio": margin_ratio,
                    "stage2_img_size": stage2_img_size,
                    "samples": len(all_filenames),
                })
                metrics_to_log = {
                    "s1_mean_norm_err": gen_s1["mean"],
                    "cascade_mean_norm_err": gen_casc["mean"],
                    "oracle_mean_norm_err": gen_oracle["mean"],
                    "delta_mean_norm_err": delta_mean_norm,
                    "relative_reduction_pct": pct_reduction,
                    "s1_share_within_5pct": gen_s1["share_within_5pct"],
                    "cascade_share_within_5pct": gen_casc["share_within_5pct"],
                    "oracle_share_within_5pct": gen_oracle["share_within_5pct"],
                    "s1_share_within_2_5pct": gen_s1["share_within_2_5pct"],
                    "cascade_share_within_2_5pct": gen_casc["share_within_2_5pct"],
                    "oracle_share_within_2_5pct": gen_oracle["share_within_2_5pct"],
                    "s1_share_within_10pct": gen_s1["share_within_10pct"],
                    "cascade_share_within_10pct": gen_casc["share_within_10pct"],
                    "oracle_share_within_10pct": gen_oracle["share_within_10pct"],
                    "s1_staging_acc": acc_s1,
                    "cascade_staging_acc": acc_casc,
                    "oracle_staging_acc": acc_oracle,
                    "s1_quadratic_kappa": kappa_s1,
                    "cascade_quadratic_kappa": kappa_casc,
                    "oracle_quadratic_kappa": kappa_oracle,
                    "s1_icc_2_1": icc_s1,
                    "cascade_icc_2_1": icc_casc,
                    "oracle_icc_2_1": icc_oracle,
                    "delta_icc_2_1": delta_icc,
                    "stage1_latency_ms": latency_summary["stage1_ms"],
                    "cascade_latency_ms": latency_summary["cascade_ms"],
                    "wilcoxon_signed_rank_p": wilcoxon_p,
                    "wilcoxon_signed_rank_stat": wilcoxon_stat,
                    "mcnemar_p_value": mcnemar_res["p_value"],
                    "mcnemar_b": mcnemar_res["b"],
                    "mcnemar_c": mcnemar_res["c"],
                    "fallback_rate_pct": delta_summary["fallback_rate_pct"],
                }
                if has_voted_stages and metrics_cascade_voted.get("valid_samples", 0) > 0:
                    metrics_to_log.update({
                        "s1_voted_staging_acc": metrics_s1_voted.get("overall_accuracy", 0.0),
                        "cascade_voted_staging_acc": metrics_cascade_voted.get("overall_accuracy", 0.0),
                        "oracle_voted_staging_acc": metrics_oracle_voted.get("overall_accuracy", 0.0),
                        "cascade_voted_adjacent_acc": metrics_cascade_voted.get("adjacent_accuracy", 0.0),
                        "cascade_voted_quadratic_kappa": metrics_cascade_voted.get("quadratic_weighted_kappa", 0.0),
                    })
                mlflow.log_metrics(metrics_to_log)
                mlflow.log_artifact(str(report_file))
                mlflow.log_artifact(str(metrics_json_file))
                mlflow.log_artifact(str(predictions_json_file))
                mlflow.log_artifact(str(csv_file))
                mlflow.log_artifact(str(staging_csv_file))
                mlflow.log_artifacts(str(vis_dir), artifact_path="visualizations")

                # Also log available dataset split evaluation files (val & train) into splits_eval artifact path
                eval_base = Path("evaluation")
                if eval_base.exists():
                    for split_art in [
                        "metrics_train.json", "metrics_val.json",
                        "summary_report_train.txt", "summary_report_val.txt",
                        "predictions_landmarks_train.csv", "predictions_landmarks_val.csv",
                    ]:
                        cand = eval_base / split_art
                        if cand.exists():
                            try:
                                mlflow.log_artifact(str(cand), artifact_path="splits_eval")
                            except Exception as e:
                                pass

                    # Log light predictions outputs if present
                    pred_full_json = eval_base / "predictions_light" / "predictions_full.json"
                    if pred_full_json.exists():
                        try:
                            mlflow.log_artifact(str(pred_full_json), artifact_path="predictions_light")
                        except Exception as e:
                            pass
                    pred_full_csv = eval_base / "predictions_light" / "predictions_full.csv"
                    if pred_full_csv.exists():
                        try:
                            mlflow.log_artifact(str(pred_full_csv), artifact_path="predictions_light")
                        except Exception as e:
                            pass
        except Exception as ml_err:
            print(f"⚠️ MLflow logging notice: {ml_err}")

    return combined_metrics_payload, str(run_dir)


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Coarse-to-Fine Evaluation CLI")
    parser.add_argument("--stage1-weights", type=str, default="artifacts/stage1_best.pth", help="Stage 1 weights")
    parser.add_argument("--stage2-weights", type=str, default="artifacts/stage2_best.pth", help="Stage 2 weights")
    parser.add_argument("--img-dir", type=str, default="dataset/test/images", help="Evaluation images directory")
    parser.add_argument("--npz-dir", type=str, default="dataset/test/labels", help="Evaluation labels directory")
    parser.add_argument("--output-dir", type=str, default="evaluation/two_stage", help="Output directory")
    parser.add_argument("--stage1-size", type=int, default=640, help="Stage 1 input resolution")
    parser.add_argument("--stage2-size", type=int, default=448, help="Stage 2 input resolution")
    parser.add_argument("--margin-ratio", type=float, default=0.22, help="ROI margin ratio (default: 0.22)")
    parser.add_argument("--bottom-margin-ratio", type=float, default=0.28, help="ROI bottom (inferior) margin ratio")
    parser.add_argument(
        "--use-peaks",
        dest="use_peaks",
        action="store_true",
        default=True,
        help="Extract Stage 2 coordinates using sub-pixel heatmap peak finding (default: True)",
    )
    parser.add_argument(
        "--no-peaks",
        dest="use_peaks",
        action="store_false",
        help="Use default SoftArgmax2D spatial expectation instead of heatmap peaks for Stage 2",
    )
    parser.add_argument(
        "--stage1-use-peaks",
        action="store_true",
        default=False,
        help="Extract Stage 1 coarse coordinates using heatmap peak finding instead of SoftArgmax (default: False)",
    )
    parser.add_argument("--refine-2pass", action="store_true", help="Enable 2-pass iterative refinement")
    parser.add_argument(
        "--use-anatomical-prior",
        dest="use_anatomical_prior",
        action="store_true",
        default=False,
        help="Enable Anatomical Vertebra Height Prior safeguard (default: False)",
    )
    parser.add_argument(
        "--no-anatomical-prior",
        dest="use_anatomical_prior",
        action="store_false",
        help="Disable Anatomical Vertebra Height Prior safeguard",
    )
    parser.add_argument("--visualize", type=int, default=10, help="Number of visualization panels to save (-1 for all)")
    parser.add_argument("--run-name", type=str, default="test_split_cascade_eval", help="MLflow and report run name")
    parser.add_argument("--run-id", type=str, default=None, help="Existing MLflow Run ID to attach evaluation metrics and artifacts to")
    parser.add_argument("--stage1-run-id", type=str, default=None, help="MLflow Run ID to automatically download Stage 1 weights from")
    parser.add_argument("--stage2-run-id", type=str, default=None, help="MLflow Run ID to automatically download Stage 2 weights from")
    parser.add_argument("--stage1-artifact", "--stage1-artifact-path", dest="stage1_artifact", type=str, default="weights/best.pt", help="Stage 1 artifact path inside MLflow run (e.g. 'weights/best.pt' or 'checkpoints/best.pth')")
    parser.add_argument("--stage2-artifact", "--stage2-artifact-path", dest="stage2_artifact", type=str, default="checkpoints/best.pth", help="Stage 2 artifact path inside MLflow run (e.g. 'checkpoints/best.pth')")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--native-img-dir", type=str, default=None, help="Directory of high-resolution native radiographs")
    parser.add_argument("--export-json", type=str, default="data/exports/export.json", help="Path to export.json for consensus finalStage mapping")
    parser.add_argument("--skip-mlflow", action="store_true", help="Skip MLflow logging")

    args = parser.parse_args()

    run_two_stage_evaluation(
        stage1_weights=args.stage1_weights,
        stage2_weights=args.stage2_weights,
        img_dir=args.img_dir,
        npz_dir=args.npz_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        stage1_img_size=args.stage1_size,
        stage2_img_size=args.stage2_size,
        margin_ratio=args.margin_ratio,
        bottom_margin_ratio=args.bottom_margin_ratio,
        use_peaks=args.use_peaks,
        stage1_use_peaks=args.stage1_use_peaks,
        refine_2pass=args.refine_2pass,
        use_anatomical_prior=args.use_anatomical_prior,
        num_visualizations=args.visualize,
        device_str=args.device,
        native_img_dir=args.native_img_dir,
        log_to_mlflow=not args.skip_mlflow,
        run_id=args.run_id,
        stage1_run_id=args.stage1_run_id,
        stage2_run_id=args.stage2_run_id,
        stage1_artifact_path=args.stage1_artifact,
        stage2_artifact_path=args.stage2_artifact,
        export_json=args.export_json,
    )


if __name__ == "__main__":
    main()
