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
from src.utils.cvm_calculator import evaluate_landmarks_dict
from src.utils.metrics import (
    compute_reference_length,
    compute_test_comprehensive_metrics,
    format_summary_report,
)
from src.utils.roi_utils import (
    ROIBoundingBox,
    canvas_to_native_coords,
    compute_roi_bbox,
    crop_roi_with_padding,
    inverse_transform_coords,
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
    """Renders a single high-resolution ROI panel with vertebra landmark dots and badges."""
    panel = cv2.resize(roi_rgb, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)

    for i, pt in enumerate(coords_norm_roi):
        if pt[0] < 0 or pt[1] < 0 or np.isnan(pt[0]) or np.isnan(pt[1]):
            continue
        px = int(round(pt[0] * panel_size))
        py = int(round(pt[1] * panel_size))

        color_bgr = LANDMARK_COLORS_BGR[i % len(LANDMARK_COLORS_BGR)]
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        cv2.circle(panel, (px, py), 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (px, py), 5, color_rgb, -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (px, py), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

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
    margin_ratio: float = 0.18,
    refine_2pass: bool = False,
    num_visualizations: int = 10,
    device_str: Optional[str] = None,
    log_to_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    tracking_username: Optional[str] = None,
    tracking_password: Optional[str] = None,
    experiment_name: Optional[str] = "cvm-two-stage-cascade-eval",
) -> Tuple[Dict[str, Any], str]:
    """
    Main programmatic routine evaluating Stage 1 alone, Stage 2 Oracle, and Two-Stage Cascade.
    """
    load_dotenv()
    device = resolve_device(device_str or "auto")

    print("=" * 85)
    print(f"STARTING TWO-STAGE CASCADED EVALUATION | Device: {device}")
    print(f"Stage 1 Weights: '{stage1_weights}' ({stage1_img_size}x{stage1_img_size})")
    print(f"Stage 2 Weights: '{stage2_weights}' ({stage2_img_size}x{stage2_img_size})")
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

    # 1. Load models
    stage1_model = load_model_checkpoint(stage1_weights, stage1_img_size, device)
    stage2_model = load_model_checkpoint(stage2_weights, stage2_img_size, device)

    two_stage_net = TwoStageCephalometricSwinGCN(
        stage1_img_size=stage1_img_size,
        stage2_img_size=stage2_img_size,
        margin_ratio=margin_ratio,
        pretrained_stage1=False,
        pretrained_stage2=False,
    )
    two_stage_net.stage1 = stage1_model
    two_stage_net.stage2 = stage2_model
    two_stage_net.freeze_stage1()
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
    s1_stages = []
    oracle_stages = []
    cascaded_stages = []

    gt_morph_list = []
    s1_morph_list = []
    oracle_morph_list = []
    cascaded_morph_list = []

    vis_saved = 0
    fallback_count = 0

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

        # Prepare 640x640 canvas tensor
        canvas_img = cv2.resize(raw_rgb, (stage1_img_size, stage1_img_size))
        tensor_in = torch.from_numpy(canvas_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            # A. Stage 1 Forward
            _, s1_coords_t = stage1_model(tensor_in)
            s1_coords_canvas = s1_coords_t.squeeze(0).cpu().numpy()  # [13, 2]

            # B. Stage 2 Oracle Forward (Crop from Ground Truth BBox)
            gt_bbox = compute_roi_bbox(
                coords=gt_canvas_norm,
                img_w=stage1_img_size,
                img_h=stage1_img_size,
                margin_ratio=margin_ratio,
            )
            oracle_patch, _ = crop_roi_with_padding(
                image=canvas_img,
                bbox=gt_bbox,
                target_size=stage2_img_size,
            )
            oracle_in = torch.from_numpy(oracle_patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            _, oracle_coords_roi_t = stage2_model(oracle_in)
            oracle_coords_roi = oracle_coords_roi_t.squeeze(0).cpu().numpy()
            oracle_coords_canvas = inverse_transform_coords(
                roi_coords=oracle_coords_roi,
                bbox=gt_bbox,
                target_size=stage2_img_size,
                output_normalized=True,
            )

            # C. Two-Stage Cascaded Forward (Stage 1 BBox -> Stage 2)
            s1_bbox = compute_roi_bbox(
                coords=s1_coords_canvas,
                img_w=stage1_img_size,
                img_h=stage1_img_size,
                margin_ratio=margin_ratio,
            )
            if s1_bbox.is_fallback:
                fallback_count += 1
                cascaded_coords_canvas = s1_coords_canvas.copy()
            else:
                casc_patch, _ = crop_roi_with_padding(
                    image=canvas_img,
                    bbox=s1_bbox,
                    target_size=stage2_img_size,
                )
                casc_in = torch.from_numpy(casc_patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                _, casc_coords_roi_t = stage2_model(casc_in)
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
                    )
                    if not s2_bbox.is_fallback:
                        casc_patch_2, _ = crop_roi_with_padding(
                            image=canvas_img,
                            bbox=s2_bbox,
                            target_size=stage2_img_size,
                        )
                        casc_in_2 = torch.from_numpy(casc_patch_2.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                        _, casc_coords_roi_2_t = stage2_model(casc_in_2)
                        casc_coords_roi_2 = casc_coords_roi_2_t.squeeze(0).cpu().numpy()
                        cascaded_coords_canvas = inverse_transform_coords(
                            roi_coords=casc_coords_roi_2,
                            bbox=s2_bbox,
                            target_size=stage2_img_size,
                            output_normalized=True,
                        )

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

        gt_stages.append(_parse_stage(gt_cvm))
        s1_stages.append(_parse_stage(s1_cvm))
        oracle_stages.append(_parse_stage(oracle_cvm))
        cascaded_stages.append(_parse_stage(casc_cvm))

        gt_morph_list.append(gt_cvm)
        s1_morph_list.append(s1_cvm)
        oracle_morph_list.append(oracle_cvm)
        cascaded_morph_list.append(casc_cvm)

        # Save side-by-side ROI visualization panels
        if vis_saved < num_visualizations:
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
            out_vis_path = vis_dir / f"vis_{vis_saved + 1:02d}_{stem}.png"
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
                "C3": [c.get("shapes", {}).get("C3", {}).get("ratio", 1.0) for c in cvm_list],
                "C4": [c.get("shapes", {}).get("C4", {}).get("ratio", 1.0) for c in cvm_list],
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

    # 4. Compute Improvement Delta Table
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
    acc_s1 = metrics_s1.get("staging", {}).get("overall_accuracy", 0.0)
    acc_casc = metrics_cascade.get("staging", {}).get("overall_accuracy", 0.0)

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
        "delta_quadratic_kappa": kappa_casc - kappa_s1,
        "s1_staging_acc": acc_s1,
        "cascade_staging_acc": acc_casc,
        "fallback_count": fallback_count,
        "fallback_rate_pct": (fallback_count / len(all_filenames)) * 100.0 if all_filenames else 0.0,
    }

    # 5. Format Summary Report
    report_cascade_text = format_summary_report(
        metrics_cascade, title=f"TWO-STAGE CASCADED PIPELINE EVALUATION REPORT — {actual_run_name}"
    )

    comparison_table = (
        "\n" + "=" * 88 + "\n"
        + "                  SECTION E: TWO-STAGE CASCADED IMPROVEMENT COMPARISON                  \n"
        + "=" * 88 + "\n"
        + f" Total Samples Evaluated    : {len(all_filenames)}\n"
        + f" BBox Sanity Fallback Count : {fallback_count} ({delta_summary['fallback_rate_pct']:.2f}%)\n"
        "----------------------------------------------------------------------------------------\n"
        " Metric                     | Stage 1 Coarse | Stage 2 Oracle | Two-Stage Cascade | Delta (Cascade - S1) \n"
        "----------------------------------------------------------------------------------------\n"
        f" Mean Normalized Error (%)  | {gen_s1['mean']:6.2f}%        | {gen_oracle['mean']:6.2f}%        | {gen_casc['mean']:6.2f}%            | {delta_mean_norm:+6.2f}% ({pct_reduction:+.1f}% rel)\n"
        f" Median Norm Error (%)      | {gen_s1['median']:6.2f}%        | {gen_oracle['median']:6.2f}%        | {gen_casc['median']:6.2f}%            | {delta_med_norm:+6.2f}%\n"
        f" Share ≤ 2.5% L_ref         | {gen_s1['share_within_2_5pct']:6.2f}%        | {gen_oracle['share_within_2_5pct']:6.2f}%        | {gen_casc['share_within_2_5pct']:6.2f}%            | {delta_sdr_2_5:+6.2f}%\n"
        f" Share ≤ 5.0% L_ref         | {gen_s1['share_within_5pct']:6.2f}%        | {gen_oracle['share_within_5pct']:6.2f}%        | {gen_casc['share_within_5pct']:6.2f}%            | {delta_sdr_5:+6.2f}%\n"
        f" Share ≤ 10.0% L_ref        | {gen_s1['share_within_10pct']:6.2f}%        | {gen_oracle['share_within_10pct']:6.2f}%        | {gen_casc['share_within_10pct']:6.2f}%            | {delta_sdr_10:+6.2f}%\n"
        f" 90th Percentile (P90)      | {gen_s1['p90']:6.2f}%        | {gen_oracle['p90']:6.2f}%        | {gen_casc['p90']:6.2f}%            | {gen_casc['p90'] - gen_s1['p90']:+6.2f}%\n"
        f" CVM Staging Accuracy       | {acc_s1:6.2f}%        | {metrics_oracle.get('staging', {}).get('overall_accuracy', 0.0):6.2f}%        | {acc_casc:6.2f}%            | {acc_casc - acc_s1:+6.2f}%\n"
        f" Quadratic Weighted Kappa   | {kappa_s1:6.3f}         | {metrics_oracle.get('staging', {}).get('quadratic_weighted_kappa', 0.0):6.3f}         | {kappa_casc:6.3f}             | {kappa_casc - kappa_s1:+6.3f}\n"
        "========================================================================================\n"
    )

    full_report_text = report_cascade_text + comparison_table

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
        "stage1_alone": metrics_s1,
        "stage2_oracle": metrics_oracle,
        "two_stage_cascade": metrics_cascade,
    }

    metrics_json_file = run_dir / "metrics.json"
    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump(combined_metrics_payload, f, indent=2)

    # 7. Save Predictions CSV
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

    print(f"\n--> Successfully saved evaluation artifacts to '{run_dir}':")
    print(f"    • Report: {report_file.name}")
    print(f"    • Metrics: {metrics_json_file.name}")
    print(f"    • Landmarks CSV: {csv_file.name}")
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

            with mlflow.start_run(run_name=f"eval_{actual_run_name}"):
                mlflow.log_params({
                    "stage1_weights": stage1_weights,
                    "stage2_weights": stage2_weights,
                    "margin_ratio": margin_ratio,
                    "stage2_img_size": stage2_img_size,
                    "samples": len(all_filenames),
                })
                mlflow.log_metrics({
                    "s1_mean_norm_err": gen_s1["mean"],
                    "cascade_mean_norm_err": gen_casc["mean"],
                    "oracle_mean_norm_err": gen_oracle["mean"],
                    "delta_mean_norm_err": delta_mean_norm,
                    "relative_reduction_pct": pct_reduction,
                    "cascade_share_within_5pct": gen_casc["share_within_5pct"],
                    "cascade_share_within_2_5pct": gen_casc["share_within_2_5pct"],
                    "cascade_staging_acc": acc_casc,
                    "cascade_quadratic_kappa": kappa_casc,
                })
                mlflow.log_artifact(str(report_file))
                mlflow.log_artifact(str(metrics_json_file))
                mlflow.log_artifact(str(csv_file))
                mlflow.log_artifacts(str(vis_dir), artifact_path="visualizations")
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
    parser.add_argument("--margin-ratio", type=float, default=0.18, help="ROI margin ratio")
    parser.add_argument("--refine-2pass", action="store_true", help="Enable 2-pass iterative refinement")
    parser.add_argument("--visualize", type=int, default=10, help="Number of visualization panels to save")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--skip-mlflow", action="store_true", help="Skip MLflow logging")

    args = parser.parse_args()

    run_two_stage_evaluation(
        stage1_weights=args.stage1_weights,
        stage2_weights=args.stage2_weights,
        img_dir=args.img_dir,
        npz_dir=args.npz_dir,
        output_dir=args.output_dir,
        stage1_img_size=args.stage1_size,
        stage2_img_size=args.stage2_size,
        margin_ratio=args.margin_ratio,
        refine_2pass=args.refine_2pass,
        num_visualizations=args.visualize,
        device_str=args.device,
        log_to_mlflow=not args.skip_mlflow,
    )


if __name__ == "__main__":
    main()
