"""
Comprehensive Landmark Error & Clinical Metric Evaluation Module for CVM Analysis.
Features:
- Pure normalized errors (% of C3 inferior endplate width reference length)
- Projections along and perpendicular to inferior borders (for IC, AI, PI)
- Image-level bootstrap confidence intervals (resampling images, not individual landmarks)
- Clinical Bland-Altman agreement for Concavity (C%) and Shape Index (SI)
- Notch classification agreement (Accuracy, Sensitivity, Specificity, Cohen's Kappa)
- Shape classification agreement & confusion matrix
- Borderline analysis (±1-2 percentage points from concavity cutoffs)
- Source breakdown: institutional (TUMS) vs public (Roboflow)
- Detailed per-sample, per-landmark CSV export rows
- Legacy mm conversion preservation
"""

import math
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.data.constants import LANDMARK_CLASSES, NUM_LANDMARKS
from src.utils.cvm_calculator import (
    PercentCVMEngine,
    PercentEngineConfig,
    evaluate_landmarks_dict,
)


# Indices for key landmarks in LANDMARK_CLASSES:
# 0: C2_PI, 1: C2_IC, 2: C2_AI
# 3: C3_PS, 4: C3_AS, 5: C3_PI, 6: C3_IC, 7: C3_AI
# 8: C4_PS, 9: C4_AS, 10: C4_PI, 11: C4_IC, 12: C4_AI
C3_PI_IDX = 5
C3_AI_IDX = 7

VERTEBRA_INFERIOR_LANDMARKS = {
    "C2": {"PI": 0, "IC": 1, "AI": 2},
    "C3": {"PI": 5, "IC": 6, "AI": 7},
    "C4": {"PI": 10, "IC": 11, "AI": 12},
}

VERTEBRA_GROUPS = {
    "C2": [0, 1, 2],
    "C3": [3, 4, 5, 6, 7],
    "C4": [8, 9, 10, 11, 12],
}


def determine_image_source(
    filename: str,
    orig_w: Optional[int] = None,
    orig_h: Optional[int] = None,
) -> str:
    """
    Determines whether a radiograph originates from the institutional clinical cohort
    (TUMS Department of Oral & Maxillofacial Radiology, native scanner resolutions e.g. 1935x2400)
    or public multi-center repositories (Roboflow Universe, pre-resized 640x640).
    """
    fname = str(filename).lower()
    if fname.startswith("tums") or "tums" in fname:
        return "institutional"
    if orig_w is not None and orig_h is not None:
        if orig_w == 640 and orig_h == 640:
            return "public"
        if max(orig_w, orig_h) > 640:
            return "institutional"
    return "institutional" if (orig_w and orig_w > 640) else "public"


def compute_reference_length(
    gt_coords: Union[np.ndarray, torch.Tensor],
    img_size: float = 640.0,
    pi_idx: int = C3_PI_IDX,
    ai_idx: int = C3_AI_IDX,
) -> Union[float, np.ndarray]:
    """
    Computes the anatomical reference length (C3 inferior endplate width) from ground truth.
    Supports single sample [13, 2] or batch [N, 13, 2] in either normalized [0, 1] or pixel scale.
    """
    is_tensor = isinstance(gt_coords, torch.Tensor)
    if is_tensor:
        coords = gt_coords.detach().cpu().numpy()
    else:
        coords = np.asarray(gt_coords)

    if coords.ndim == 2:
        # Single sample [13, 2]
        p_pi = coords[pi_idx]
        p_ai = coords[ai_idx]
        if p_pi[0] < 0 or p_ai[0] < 0:
            return float(img_size)  # fallback safeguard
        dx = (p_ai[0] - p_pi[0]) * (img_size if np.max(coords) <= 1.0 else 1.0)
        dy = (p_ai[1] - p_pi[1]) * (img_size if np.max(coords) <= 1.0 else 1.0)
        dist = math.hypot(dx, dy)
        return float(dist) if dist > 1e-4 else float(img_size)

    # Batch [N, 13, 2]
    p_pi = coords[:, pi_idx]
    p_ai = coords[:, ai_idx]
    scale = img_size if np.max(coords) <= 1.0 else 1.0
    dx = (p_ai[:, 0] - p_pi[:, 0]) * scale
    dy = (p_ai[:, 1] - p_pi[:, 1]) * scale
    dists = np.hypot(dx, dy)
    # Fallback to img_size if any distance is invalid
    invalid = (p_pi[:, 0] < 0) | (p_ai[:, 0] < 0) | (dists <= 1e-4)
    dists[invalid] = float(img_size)
    return dists


def compute_normalized_errors(
    pred_coords: Union[np.ndarray, torch.Tensor],
    gt_coords: Union[np.ndarray, torch.Tensor],
    ref_lengths: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
    img_size: float = 640.0,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Computes radial pixel errors, normalized percentage errors, and signed directional errors:
    - radial_error_px: ||p_hat - p*||_2 in pixels
    - normalized_error_pct: (radial_error_px / L_ref) * 100%
    - signed_dx_px: x_hat - x* in pixels
    - signed_dy_px: y_hat - y* in pixels
    - signed_dx_pct: (signed_dx_px / L_ref) * 100%
    - signed_dy_pct: (signed_dy_px / L_ref) * 100%
    - ref_length: L_ref (C3 inferior endplate width) in pixels
    """
    is_2d = False
    if isinstance(pred_coords, torch.Tensor):
        p_coords = pred_coords.detach().cpu().numpy()
    else:
        p_coords = np.asarray(pred_coords, dtype=float)

    if isinstance(gt_coords, torch.Tensor):
        g_coords = gt_coords.detach().cpu().numpy()
    else:
        g_coords = np.asarray(gt_coords, dtype=float)

    if p_coords.ndim == 2:
        is_2d = True
        p_coords = p_coords[None, ...]
        g_coords = g_coords[None, ...]

    scale = img_size if np.max(g_coords) <= 1.0 else 1.0
    pred_px = p_coords * scale
    gt_px = g_coords * scale

    if ref_lengths is None:
        refs = compute_reference_length(g_coords, img_size=img_size)
    else:
        if isinstance(ref_lengths, torch.Tensor):
            refs = ref_lengths.detach().cpu().numpy()
        else:
            refs = np.asarray(ref_lengths, dtype=float)
        if refs.ndim == 0:
            refs = np.full(p_coords.shape[0], float(refs))

    refs_safe = np.where(refs > 1e-4, refs, float(img_size))
    refs_2d = refs_safe[:, None]  # [N, 1]

    dx_px = pred_px[:, :, 0] - gt_px[:, :, 0]
    dy_px = pred_px[:, :, 1] - gt_px[:, :, 1]
    radial_px = np.sqrt(dx_px**2 + dy_px**2)

    norm_pct = (radial_px / refs_2d) * 100.0
    dx_pct = (dx_px / refs_2d) * 100.0
    dy_pct = (dy_px / refs_2d) * 100.0

    if is_2d:
        return {
            "radial_error_px": radial_px[0],
            "normalized_error_pct": norm_pct[0],
            "signed_dx_px": dx_px[0],
            "signed_dy_px": dy_px[0],
            "signed_dx_pct": dx_pct[0],
            "signed_dy_pct": dy_pct[0],
            "ref_length": float(refs[0]),
        }

    return {
        "radial_error_px": radial_px,
        "normalized_error_pct": norm_pct,
        "signed_dx_px": dx_px,
        "signed_dy_px": dy_px,
        "signed_dx_pct": dx_pct,
        "signed_dy_pct": dy_pct,
        "ref_length": refs,
    }


def compute_inferior_border_projections(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    vertebra: str,
    landmark_key: str,
    img_size: float = 640.0,
    ref_lengths: Optional[Union[float, np.ndarray]] = None,
    return_pct: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Computes the error of a landmark projected:
    1) Along the inferior border (anteriorly directed tangent)
    2) Perpendicular to the inferior border (superiorly directed normal into vertebral body)

    Returns:
        If return_pct is False: (error_along_px, error_perp_px)
        If return_pct is True:  (error_along_px, error_perp_px, error_along_pct, error_perp_pct)
    """
    is_2d = False
    p_coords = np.asarray(pred_coords, dtype=float)
    g_coords = np.asarray(gt_coords, dtype=float)
    if g_coords.ndim == 2:
        is_2d = True
        p_coords = p_coords[None, ...]
        g_coords = g_coords[None, ...]
        if ref_lengths is not None and np.ndim(ref_lengths) == 0:
            ref_lengths = np.array([float(ref_lengths)])

    v_dict = VERTEBRA_INFERIOR_LANDMARKS[vertebra]
    pi_idx = v_dict["PI"]
    ai_idx = v_dict["AI"]
    target_idx = v_dict[landmark_key]

    # Convert coordinates to pixel scale
    scale = img_size if np.max(g_coords) <= 1.0 else 1.0
    gt_px = g_coords * scale
    pred_px = p_coords * scale

    # Inferior border baseline vector (PI -> AI)
    base_dx = gt_px[:, ai_idx, 0] - gt_px[:, pi_idx, 0]
    base_dy = gt_px[:, ai_idx, 1] - gt_px[:, pi_idx, 1]
    base_len = np.hypot(base_dx, base_dy)
    base_len_safe = np.where(base_len > 1e-4, base_len, 1.0)

    # Unit tangent vector along inferior border
    u_tangent_x = base_dx / base_len_safe
    u_tangent_y = base_dy / base_len_safe

    # Normal vector pointing superiorly into the vertebral body (smaller Y)
    s = np.where(base_dx >= 0, 1.0, -1.0)
    norm_x = s * base_dy
    norm_y = s * (-base_dx)
    norm_len = np.hypot(norm_x, norm_y)
    norm_len_safe = np.where(norm_len > 1e-4, norm_len, 1.0)
    u_norm_x = norm_x / norm_len_safe
    u_norm_y = norm_y / norm_len_safe

    # Prediction error vector
    err_dx = pred_px[:, target_idx, 0] - gt_px[:, target_idx, 0]
    err_dy = pred_px[:, target_idx, 1] - gt_px[:, target_idx, 1]

    # Projections in pixels
    error_along = err_dx * u_tangent_x + err_dy * u_tangent_y
    error_perp = err_dx * u_norm_x + err_dy * u_norm_y

    if return_pct:
        if ref_lengths is not None:
            ref_arr = np.asarray(ref_lengths, dtype=float)
            ref_safe = np.where(ref_arr > 1e-4, ref_arr, 1.0)
            along_pct = (error_along / ref_safe) * 100.0
            perp_pct = (error_perp / ref_safe) * 100.0
        else:
            along_pct = np.zeros_like(error_along)
            perp_pct = np.zeros_like(error_perp)
        if is_2d:
            return error_along[0], error_perp[0], along_pct[0], perp_pct[0]
        return error_along, error_perp, along_pct, perp_pct

    if is_2d:
        return error_along[0], error_perp[0]
    return error_along, error_perp


def compute_distribution_statistics(
    errors: np.ndarray,
    dx_errors: Optional[np.ndarray] = None,
    dy_errors: Optional[np.ndarray] = None,
    sample_ids: Optional[Union[List[str], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Computes standard publication summary statistics for error distributions:
    - mean ± sd
    - median with IQR (iqr = q75 - q25, q25, q75)
    - threshold shares: share within 2.5%, 5.0%, 7.5%, 10.0%
    - 90th percentile, 95th percentile
    - maximum error and associated sample ID(s)
    - rmse
    - mean signed dx and dy
    """
    errs = np.asarray(errors, dtype=float)
    if len(errs) == 0:
        return {
            "mean": 0.0,
            "sd": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "share_within_2_5pct": 0.0,
            "share_within_5pct": 0.0,
            "share_within_7_5pct": 0.0,
            "share_within_10pct": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "max_sample_ids": [],
            "rmse": 0.0,
            "mean_signed_dx": 0.0,
            "mean_signed_dy": 0.0,
            "count": 0,
        }

    mean_val = float(np.mean(errs))
    sd_val = float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0
    median_val = float(np.median(errs))
    q25 = float(np.percentile(errs, 25.0))
    q75 = float(np.percentile(errs, 75.0))
    iqr_val = q75 - q25

    share_2_5 = float(np.mean(errs <= 2.5) * 100.0)
    share_5_0 = float(np.mean(errs <= 5.0) * 100.0)
    share_7_5 = float(np.mean(errs <= 7.5) * 100.0)
    share_10_0 = float(np.mean(errs <= 10.0) * 100.0)

    p90 = float(np.percentile(errs, 90.0))
    p95 = float(np.percentile(errs, 95.0))
    max_val = float(np.max(errs))

    max_samples = []
    if sample_ids is not None:
        sample_ids_arr = np.asarray(sample_ids)
        if len(sample_ids_arr) == len(errs):
            max_mask = np.isclose(errs, max_val, atol=1e-4)
            max_samples = sample_ids_arr[max_mask].tolist()

    rmse_val = float(np.sqrt(np.mean(errs**2)))

    mean_dx = float(np.mean(dx_errors)) if dx_errors is not None and len(dx_errors) > 0 else 0.0
    mean_dy = float(np.mean(dy_errors)) if dy_errors is not None and len(dy_errors) > 0 else 0.0

    return {
        "mean": mean_val,
        "sd": sd_val,
        "median": median_val,
        "iqr": iqr_val,
        "q25": q25,
        "q75": q75,
        "share_within_2_5pct": share_2_5,
        "share_within_5pct": share_5_0,
        "share_within_7_5pct": share_7_5,
        "share_within_10pct": share_10_0,
        "p90": p90,
        "p95": p95,
        "max": max_val,
        "max_sample_ids": max_samples,
        "rmse": rmse_val,
        "mean_signed_dx": mean_dx,
        "mean_signed_dy": mean_dy,
        "count": len(errs),
    }


def compute_vertebra_breakdown(
    norm_errors: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Computes per-vertebra (C2, C3, C4) metrics:
    - mean ± sd
    - median
    - share within 5%
    """
    results = {}
    for vert, idxs in VERTEBRA_GROUPS.items():
        v_mask = valid_mask[:, idxs]
        v_errors = norm_errors[:, idxs][v_mask]
        if len(v_errors) > 0:
            m = float(np.mean(v_errors))
            sd = float(np.std(v_errors, ddof=1)) if len(v_errors) > 1 else 0.0
            med = float(np.median(v_errors))
            s5 = float(np.mean(v_errors <= 5.0) * 100.0)
        else:
            m, sd, med, s5 = 0.0, 0.0, 0.0, 0.0
        results[vert] = {
            "mean": m,
            "sd": sd,
            "median": med,
            "share_within_5pct": s5,
            "count": len(v_errors),
        }
    return results


def compute_landmark_breakdown(
    norm_errors: np.ndarray,
    valid_mask: np.ndarray,
    along_errors_px: Optional[Dict[str, np.ndarray]] = None,
    perp_errors_px: Optional[Dict[str, np.ndarray]] = None,
    along_errors_pct: Optional[Dict[str, np.ndarray]] = None,
    perp_errors_pct: Optional[Dict[str, np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    """
    Computes per-landmark performance for all 13 landmarks:
    - mean ± sd, median, share within 5%
    - for IC, AI, PI: mean error along and perpendicular to the inferior border
    """
    results = []
    for idx, name in enumerate(LANDMARK_CLASSES):
        vert = name.split("_")[0]
        lm_type = name.split("_")[1]
        mask_idx = valid_mask[:, idx]
        errs = norm_errors[:, idx][mask_idx]

        entry = {
            "index": idx,
            "name": name,
            "vertebra": vert,
            "type": lm_type,
            "mean": float(np.mean(errs)) if len(errs) > 0 else 0.0,
            "sd": float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0,
            "median": float(np.median(errs)) if len(errs) > 0 else 0.0,
            "share_within_5pct": float(np.mean(errs <= 5.0) * 100.0) if len(errs) > 0 else 0.0,
            "count": int(np.sum(mask_idx)),
        }

        # Add along / perp if inferior landmark (IC, AI, PI)
        if lm_type in ("IC", "AI", "PI"):
            key = f"{vert}_{lm_type}"
            if along_errors_px and key in along_errors_px:
                entry["mean_along_px"] = float(np.mean(along_errors_px[key]))
            if perp_errors_px and key in perp_errors_px:
                entry["mean_perp_px"] = float(np.mean(perp_errors_px[key]))
            if along_errors_pct and key in along_errors_pct:
                entry["mean_along_pct"] = float(np.mean(along_errors_pct[key]))
            if perp_errors_pct and key in perp_errors_pct:
                entry["mean_perp_pct"] = float(np.mean(perp_errors_pct[key]))

        results.append(entry)
    return results


def compute_source_breakdown(
    norm_errors: np.ndarray,
    valid_mask: np.ndarray,
    sources: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Stratifies metrics by image source (institutional vs public).
    """
    sources_arr = np.asarray(sources)
    unique_sources = sorted(list(set(sources)))
    results = {}
    for src in unique_sources:
        src_mask = sources_arr == src
        if not np.any(src_mask):
            continue
        sub_errors = norm_errors[src_mask]
        sub_valid = valid_mask[src_mask]
        pooled_errors = sub_errors[sub_valid]

        stats = compute_distribution_statistics(pooled_errors)
        vert_stats = compute_vertebra_breakdown(sub_errors, sub_valid)
        stats["by_vertebra"] = vert_stats
        results[src] = stats

    return results


def compute_bland_altman(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Computes Bland-Altman statistics:
    - bias: mean difference (pred - gt)
    - sd_diff: standard deviation of differences
    - loa_lower: bias - 1.96 * sd_diff
    - loa_upper: bias + 1.96 * sd_diff
    - mae: mean absolute error
    """
    diffs = np.asarray(pred, dtype=float) - np.asarray(gt, dtype=float)
    bias = float(np.mean(diffs))
    mae = float(np.mean(np.abs(diffs)))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    loa_lower = bias - 1.96 * sd_diff
    loa_upper = bias + 1.96 * sd_diff
    return {
        "bias": bias,
        "sd_diff": sd_diff,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
        "mae": mae,
    }


def compute_notch_classification_metrics(pred_notches: np.ndarray, gt_notches: np.ndarray) -> Dict[str, float]:
    """
    Computes binary notch agreement metrics:
    - accuracy
    - sensitivity (recall)
    - specificity
    - cohen_kappa
    """
    pred_b = np.asarray(pred_notches, dtype=bool)
    gt_b = np.asarray(gt_notches, dtype=bool)

    tp = int(np.sum(pred_b & gt_b))
    fp = int(np.sum(pred_b & (~gt_b)))
    tn = int(np.sum((~pred_b) & (~gt_b)))
    fn = int(np.sum((~pred_b) & gt_b))
    n = tp + fp + tn + fn

    accuracy = (tp + tn) / n if n > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

    # Cohen's Kappa
    p_o = accuracy
    p_yes = ((tp + fp) * (tp + fn)) / (n * n) if n > 0 else 0.0
    p_no = ((tn + fn) * (tn + fp)) / (n * n) if n > 0 else 0.0
    p_e = p_yes + p_no
    if abs(1.0 - p_e) < 1e-6:
        kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return {
        "accuracy": float(accuracy * 100.0),
        "sensitivity": float(sensitivity * 100.0),
        "specificity": float(specificity * 100.0),
        "cohen_kappa": float(kappa),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": n,
    }


def compute_shape_agreement(
    pred_shapes: List[str],
    gt_shapes: List[str],
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes multiclass shape agreement accuracy and confusion matrix.
    """
    if categories is None:
        categories = ["Trapezoidal", "Rectangular Horizontal", "Square", "Rectangular Vertical"]

    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    num_cats = len(categories)
    cm = np.zeros((num_cats, num_cats), dtype=int)

    correct = 0
    total = len(pred_shapes)

    for p, g in zip(pred_shapes, gt_shapes):
        if p == g:
            correct += 1
        p_idx = cat_to_idx.get(p, -1)
        g_idx = cat_to_idx.get(g, -1)
        if p_idx >= 0 and g_idx >= 0:
            cm[g_idx, p_idx] += 1

    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "total": total,
        "categories": categories,
        "confusion_matrix": cm.tolist(),
    }


def compute_borderline_analysis(
    gt_percentages: Dict[str, np.ndarray],
    pred_stages: np.ndarray,
    gt_stages: np.ndarray,
    cutoffs: Optional[Dict[str, float]] = None,
    deltas: Tuple[float, ...] = (1.0, 2.0),
) -> Dict[str, Any]:
    """
    Borderline sensitivity analysis:
    Computes the share of test cases where ground-truth C% is within ±delta of a cutoff,
    and compares staging accuracy within that subset versus outside it.
    """
    if cutoffs is None:
        cutoffs = {"C2": 5.0, "C3": 5.0, "C4": 6.0}

    n_samples = len(pred_stages)
    results = {}

    for delta in deltas:
        borderline_mask = np.zeros(n_samples, dtype=bool)
        vert_masks = {}
        for vert, cutoff in cutoffs.items():
            c_pct = gt_percentages[vert]
            v_mask = np.abs(c_pct - cutoff) <= delta
            vert_masks[vert] = v_mask
            borderline_mask = borderline_mask | v_mask

        n_bl = int(np.sum(borderline_mask))
        n_clear = n_samples - n_bl
        share_bl = (n_bl / n_samples * 100.0) if n_samples > 0 else 0.0

        correct_bl = int(np.sum((pred_stages == gt_stages) & borderline_mask))
        acc_bl = (correct_bl / n_bl * 100.0) if n_bl > 0 else 0.0

        correct_clear = int(np.sum((pred_stages == gt_stages) & (~borderline_mask)))
        acc_clear = (correct_clear / n_clear * 100.0) if n_clear > 0 else 0.0

        overall_correct = int(np.sum(pred_stages == gt_stages))
        acc_all = (overall_correct / n_samples * 100.0) if n_samples > 0 else 0.0

        results[f"delta_{delta}pct"] = {
            "delta": delta,
            "borderline_count": n_bl,
            "borderline_share_pct": round(share_bl, 2),
            "borderline_accuracy_pct": round(acc_bl, 2),
            "outside_borderline_count": n_clear,
            "outside_borderline_accuracy_pct": round(acc_clear, 2),
            "overall_accuracy_pct": round(acc_all, 2),
        }

    return results


def bootstrap_image_confidence_interval(
    sample_values_by_image: List[np.ndarray],
    stat_fn=np.mean,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Computes non-parametric bootstrap confidence intervals by resampling IMAGES
    (with replacement) rather than individual landmarks to account for intra-image correlation.
    """
    n_images = len(sample_values_by_image)
    if n_images == 0:
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    bootstrap_stats = np.empty(n_resamples, dtype=float)

    for b in range(n_resamples):
        resampled_indices = rng.randint(0, n_images, size=n_images)
        resampled_pooled = np.concatenate([sample_values_by_image[idx] for idx in resampled_indices])
        bootstrap_stats[b] = stat_fn(resampled_pooled)

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(bootstrap_stats, alpha * 100.0))
    upper = float(np.percentile(bootstrap_stats, (1.0 - alpha) * 100.0))
    return lower, upper


def compute_validation_summary_metrics(
    pred_coords: Union[np.ndarray, torch.Tensor],
    gt_coords: Union[np.ndarray, torch.Tensor],
    img_size: float = 640.0,
    pixel_spacing: Optional[Union[float, np.ndarray, torch.Tensor]] = 0.1,
) -> Dict[str, float]:
    """
    Lightweight, fast metric computation executed each epoch on the validation set.
    Computes:
    - mean_norm_err: Mean normalized error (% of C3 inferior endplate reference length)
    - median_norm_err: Median normalized error (%)
    - share_within_5pct: Share of landmarks within 5% of reference length
    - Legacy mm and px metrics for continuous logging
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()

    # Valid mask for GT coordinates
    valid_mask = (gt_coords[:, :, 0] >= 0) & (gt_coords[:, :, 1] >= 0)
    if not np.any(valid_mask):
        return {
            "mean_norm_err": 0.0,
            "median_norm_err": 0.0,
            "share_within_5pct": 0.0,
            "share_within_2_5pct": 0.0,
            "share_within_7_5pct": 0.0,
            "share_within_10pct": 0.0,
            "rmse_norm_err": 0.0,
            "mre_px": 0.0,
            "mre_mm": 0.0,
            "sdr_2_0mm": 0.0,
        }

    # Absolute pixel coords on canvas
    pred_px = pred_coords * img_size
    gt_px = gt_coords * img_size

    # Reference length per sample (C3 inferior endplate width)
    ref_lengths = compute_reference_length(gt_coords, img_size=img_size)  # shape (N,)
    ref_lengths_2d = ref_lengths[:, None]  # shape (N, 1)

    # Radial Euclidean errors in pixels
    dx = pred_px[:, :, 0] - gt_px[:, :, 0]
    dy = pred_px[:, :, 1] - gt_px[:, :, 1]
    radial_px = np.sqrt(dx**2 + dy**2)  # shape (N, 13)

    # Normalized error in %
    norm_err = (radial_px / ref_lengths_2d) * 100.0  # shape (N, 13)

    valid_norm_err = norm_err[valid_mask]
    valid_radial_px = radial_px[valid_mask]

    mean_norm = float(np.mean(valid_norm_err))
    median_norm = float(np.median(valid_norm_err))
    share_5 = float(np.mean(valid_norm_err <= 5.0) * 100.0)
    share_2_5 = float(np.mean(valid_norm_err <= 2.5) * 100.0)
    share_7_5 = float(np.mean(valid_norm_err <= 7.5) * 100.0)
    share_10 = float(np.mean(valid_norm_err <= 10.0) * 100.0)
    rmse_norm = float(np.sqrt(np.mean(valid_norm_err**2)))

    # Legacy metrics
    mre_px = float(np.mean(valid_radial_px))
    if pixel_spacing is not None:
        if isinstance(pixel_spacing, torch.Tensor):
            ps = pixel_spacing.detach().cpu().numpy()
        else:
            ps = np.asarray(pixel_spacing)
        if ps.ndim > 0:
            radial_mm = radial_px * ps[:, None]
        else:
            radial_mm = radial_px * float(ps)
        mre_mm = float(np.mean(radial_mm[valid_mask]))
        sdr_2mm = float(np.mean(radial_mm[valid_mask] <= 2.0) * 100.0)
    else:
        mre_mm = 0.0
        sdr_2mm = 0.0

    return {
        "mean_norm_err": mean_norm,
        "median_norm_err": median_norm,
        "share_within_5pct": share_5,
        "share_within_2_5pct": share_2_5,
        "share_within_7_5pct": share_7_5,
        "share_within_10pct": share_10,
        "rmse_norm_err": rmse_norm,
        "mre_px": mre_px,
        "mre_mm": mre_mm,
        "sdr_2_0mm": sdr_2mm,
    }


def compute_test_comprehensive_metrics(
    pred_coords: Union[np.ndarray, torch.Tensor],
    gt_coords: Union[np.ndarray, torch.Tensor],
    sample_ids: Optional[List[str]] = None,
    orig_dims: Optional[List[Tuple[int, int]]] = None,
    pixel_spacing: Optional[Union[float, np.ndarray, torch.Tensor]] = 0.1,
    img_size: float = 640.0,
    pred_stages: Optional[Union[np.ndarray, List[int]]] = None,
    gt_stages: Optional[Union[np.ndarray, List[int]]] = None,
    pred_morphometrics: Optional[Dict[str, Any]] = None,
    gt_morphometrics: Optional[Dict[str, Any]] = None,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """
    Comprehensive test set aggregator producing all structured outputs required
    by the CVM clinical evaluation suite and publication reporting:
    - Section B: Normalized landmark errors (General, Per-Vertebra, Per-Landmark, Source, Bootstrap 95% CIs)
    - Section C: Clinical morphometric validation (Bland-Altman, Notch Agreement, Shape Agreement, Borderline Sensitivity)
    """
    if isinstance(pred_coords, torch.Tensor):
        p_coords = pred_coords.detach().cpu().numpy()
    else:
        p_coords = np.asarray(pred_coords, dtype=float)

    if isinstance(gt_coords, torch.Tensor):
        g_coords = gt_coords.detach().cpu().numpy()
    else:
        g_coords = np.asarray(gt_coords, dtype=float)

    n_samples = p_coords.shape[0]
    if sample_ids is None:
        sample_ids = [f"sample_{i:04d}" for i in range(n_samples)]

    # 1. Valid mask
    valid_mask = (g_coords[:, :, 0] >= 0) & (g_coords[:, :, 1] >= 0)

    # 2. Reference lengths and normalized errors
    norm_dict = compute_normalized_errors(p_coords, g_coords, img_size=img_size)
    radial_px = norm_dict["radial_error_px"]
    norm_pct = norm_dict["normalized_error_pct"]
    dx_px = norm_dict["signed_dx_px"]
    dy_px = norm_dict["signed_dy_px"]
    ref_lengths = norm_dict["ref_length"]

    # 3. Inferior border projections for all inferior landmarks
    along_px_dict: Dict[str, np.ndarray] = {}
    perp_px_dict: Dict[str, np.ndarray] = {}
    along_pct_dict: Dict[str, np.ndarray] = {}
    perp_pct_dict: Dict[str, np.ndarray] = {}

    for vert in ("C2", "C3", "C4"):
        for lm_k in ("PI", "IC", "AI"):
            a_px, p_px, a_pct, p_pct = compute_inferior_border_projections(
                p_coords,
                g_coords,
                vertebra=vert,
                landmark_key=lm_k,
                img_size=img_size,
                ref_lengths=ref_lengths,
                return_pct=True,
            )
            key = f"{vert}_{lm_k}"
            along_px_dict[key] = a_px
            perp_px_dict[key] = p_px
            along_pct_dict[key] = a_pct
            perp_pct_dict[key] = p_pct

    # 4. Section B.1: General Statistics
    # Flatten sample IDs to match pooled valid landmarks
    pooled_sample_ids = []
    for i in range(n_samples):
        for j in range(NUM_LANDMARKS):
            if valid_mask[i, j]:
                pooled_sample_ids.append(sample_ids[i])

    general_stats = compute_distribution_statistics(
        errors=norm_pct[valid_mask],
        dx_errors=dx_px[valid_mask],
        dy_errors=dy_px[valid_mask],
        sample_ids=pooled_sample_ids,
    )

    # 5. Section B.2: Vertebra Breakdown
    vertebra_breakdown = compute_vertebra_breakdown(norm_pct, valid_mask)

    # 6. Section B.3: Per-Landmark Breakdown
    landmark_breakdown = compute_landmark_breakdown(
        norm_errors=norm_pct,
        valid_mask=valid_mask,
        along_errors_px=along_px_dict,
        perp_errors_px=perp_px_dict,
        along_errors_pct=along_pct_dict,
        perp_errors_pct=perp_pct_dict,
    )

    # 7. Section B.4: Cross-Source Stratification
    sources = []
    for i in range(n_samples):
        fn = sample_ids[i]
        orig_w = orig_dims[i][0] if orig_dims and i < len(orig_dims) else None
        orig_h = orig_dims[i][1] if orig_dims and i < len(orig_dims) else None
        sources.append(determine_image_source(fn, orig_w, orig_h))

    source_breakdown = compute_source_breakdown(norm_pct, valid_mask, sources)

    # 8. Section B.5: Image-Level Bootstrapping (1,000 resamples)
    sample_values_by_image = [norm_pct[i][valid_mask[i]] for i in range(n_samples) if np.any(valid_mask[i])]
    mean_ci = bootstrap_image_confidence_interval(
        sample_values_by_image, stat_fn=np.mean, n_resamples=n_bootstrap, seed=bootstrap_seed
    )
    median_ci = bootstrap_image_confidence_interval(
        sample_values_by_image, stat_fn=np.median, n_resamples=n_bootstrap, seed=bootstrap_seed
    )

    # 9. Section C: Clinical Morphometrics
    clinical_metrics: Dict[str, Any] = {}
    if pred_morphometrics is not None and gt_morphometrics is not None:
        # Bland-Altman for Concavity % (C2, C3, C4)
        ba_concavities = {}
        for vert in ("C2", "C3", "C4"):
            if vert in pred_morphometrics.get("concavities", {}) and vert in gt_morphometrics.get("concavities", {}):
                p_c = np.asarray(pred_morphometrics["concavities"][vert], dtype=float)
                g_c = np.asarray(gt_morphometrics["concavities"][vert], dtype=float)
                ba_concavities[vert] = compute_bland_altman(p_c, g_c)
        clinical_metrics["bland_altman_concavity"] = ba_concavities

        # Bland-Altman for Shape Index (C3, C4)
        ba_shapes = {}
        for vert in ("C3", "C4"):
            if vert in pred_morphometrics.get("shape_indices", {}) and vert in gt_morphometrics.get("shape_indices", {}):
                p_si = np.asarray(pred_morphometrics["shape_indices"][vert], dtype=float)
                g_si = np.asarray(gt_morphometrics["shape_indices"][vert], dtype=float)
                ba_shapes[vert] = compute_bland_altman(p_si, g_si)
        clinical_metrics["bland_altman_shape_index"] = ba_shapes

        # Notch Classification Agreement (C2, C3, C4)
        notch_agreements = {}
        for vert in ("C2", "C3", "C4"):
            if vert in pred_morphometrics.get("notches", {}) and vert in gt_morphometrics.get("notches", {}):
                p_n = np.asarray(pred_morphometrics["notches"][vert], dtype=bool)
                g_n = np.asarray(gt_morphometrics["notches"][vert], dtype=bool)
                notch_agreements[vert] = compute_notch_classification_metrics(p_n, g_n)
        clinical_metrics["notch_agreement"] = notch_agreements

        # Shape Classification Agreement (C3, C4)
        shape_agreements = {}
        for vert in ("C3", "C4"):
            if vert in pred_morphometrics.get("shapes", {}) and vert in gt_morphometrics.get("shapes", {}):
                p_s = pred_morphometrics["shapes"][vert]
                g_s = gt_morphometrics["shapes"][vert]
                shape_agreements[vert] = compute_shape_agreement(p_s, g_s)
        clinical_metrics["shape_agreement"] = shape_agreements

        # Borderline Sensitivity Analysis
        if pred_stages is not None and gt_stages is not None and "concavities" in gt_morphometrics:
            gt_p_dict = {
                vert: np.asarray(gt_morphometrics["concavities"][vert], dtype=float)
                for vert in ("C2", "C3", "C4")
                if vert in gt_morphometrics["concavities"]
            }
            if len(gt_p_dict) == 3:
                p_st = np.asarray(pred_stages)
                g_st = np.asarray(gt_stages)
                clinical_metrics["borderline_analysis"] = compute_borderline_analysis(
                    gt_percentages=gt_p_dict,
                    pred_stages=p_st,
                    gt_stages=g_st,
                    deltas=(1.0, 2.0),
                )

    # 10. Legacy validation summary for reference
    legacy_summary = compute_validation_summary_metrics(
        pred_coords=p_coords,
        gt_coords=g_coords,
        img_size=img_size,
        pixel_spacing=pixel_spacing,
    )

    return {
        "general": general_stats,
        "bootstrap_ci_95": {
            "mean_norm_err": {"lower": round(mean_ci[0], 2), "upper": round(mean_ci[1], 2)},
            "median_norm_err": {"lower": round(median_ci[0], 2), "upper": round(median_ci[1], 2)},
            "n_resamples": n_bootstrap,
        },
        "by_vertebra": vertebra_breakdown,
        "by_landmark": landmark_breakdown,
        "by_source": source_breakdown,
        "clinical": clinical_metrics,
        "legacy": legacy_summary,
        "total_samples": n_samples,
    }


def format_summary_report(metrics_dict: Dict[str, Any], title: str = "CVM SWIN-GCN EVALUATION REPORT") -> str:
    """
    Renders publication-ready, formatted ASCII tables summarizing Section B (Normalized Landmark Suite)
    and Section C (Clinical Morphometric Validation Suite).
    """
    lines = []
    lines.append("=" * 88)
    lines.append(f" {title.center(86)} ")
    lines.append("=" * 88)

    # Section B.1: Overall Normalized Performance
    gen = metrics_dict.get("general", {})
    boot = metrics_dict.get("bootstrap_ci_95", {})
    mean_ci = boot.get("mean_norm_err", {})
    med_ci = boot.get("median_norm_err", {})

    lines.append("\n" + "SECTION B.1: OVERALL NORMALIZED LANDMARK PERFORMANCE (% L_ref)".center(88))
    lines.append("-" * 88)
    lines.append(f" Total Evaluated Samples  : {metrics_dict.get('total_samples', 0)}")
    lines.append(f" Total Valid Landmarks    : {gen.get('count', 0)}")
    lines.append(f" Mean Normalized Error    : {gen.get('mean', 0.0):.2f}% ± {gen.get('sd', 0.0):.2f}%")
    lines.append(f"   95% Bootstrap CI (Mean): [{mean_ci.get('lower', 0.0):.2f}%, {mean_ci.get('upper', 0.0):.2f}%]")
    lines.append(f" Median Normalized Error  : {gen.get('median', 0.0):.2f}% (IQR: {gen.get('iqr', 0.0):.2f}%, Q25: {gen.get('q25', 0.0):.2f}%, Q75: {gen.get('q75', 0.0):.2f}%)")
    lines.append(f"   95% Bootstrap CI (Med) : [{med_ci.get('lower', 0.0):.2f}%, {med_ci.get('upper', 0.0):.2f}%]")
    lines.append(f" Normalized SDR Thresholds:")
    lines.append(f"   • Share within 2.5%    : {gen.get('share_within_2_5pct', 0.0):.2f}%")
    lines.append(f"   • Share within 5.0%    : {gen.get('share_within_5pct', 0.0):.2f}% (Primary Checkpoint Criterion)")
    lines.append(f"   • Share within 7.5%    : {gen.get('share_within_7_5pct', 0.0):.2f}%")
    lines.append(f"   • Share within 10.0%   : {gen.get('share_within_10pct', 0.0):.2f}%")
    lines.append(f" Distribution Tails & Outliers:")
    lines.append(f"   • 90th Percentile (P90): {gen.get('p90', 0.0):.2f}%")
    lines.append(f"   • 95th Percentile (P95): {gen.get('p95', 0.0):.2f}%")
    lines.append(f"   • Maximum Radial Error : {gen.get('max', 0.0):.2f}% (Samples: {', '.join(gen.get('max_sample_ids', [])[:3])})")
    lines.append(f" Directional Bias & RMSE  :")
    lines.append(f"   • Root Mean Square Err : {gen.get('rmse', 0.0):.2f}%")
    lines.append(f"   • Mean Signed dx       : {gen.get('mean_signed_dx', 0.0):+.2f} px")
    lines.append(f"   • Mean Signed dy       : {gen.get('mean_signed_dy', 0.0):+.2f} px")

    # Section B.2: Vertebra Breakdown
    lines.append("\n" + "SECTION B.2: PER-VERTEBRA BREAKDOWN (MAIN TEXT)".center(88))
    lines.append("-" * 88)
    lines.append(f"{'Vertebra':<12} | {'Mean ± SD (%)':<20} | {'Median (%)':<14} | {'Share <= 5%':<16} | {'Count':<8}")
    lines.append("-" * 88)
    for vert, v_stats in metrics_dict.get("by_vertebra", {}).items():
        m_sd = f"{v_stats.get('mean', 0.0):.2f} ± {v_stats.get('sd', 0.0):.2f}"
        lines.append(
            f"{vert:<12} | {m_sd:<20} | {v_stats.get('median', 0.0):<14.2f} | {v_stats.get('share_within_5pct', 0.0):<15.2f}% | {v_stats.get('count', 0):<8}"
        )

    # Section B.3: Per-Landmark Breakdown (Supplementary)
    lines.append("\n" + "SECTION B.3: PER-LANDMARK PERFORMANCE & INFERIOR PROJECTIONS (SUPPLEMENTARY)".center(88))
    lines.append("-" * 88)
    lines.append(f"{'Idx':<4} {'Landmark':<8} | {'Mean ± SD (%)':<16} | {'Med (%)':<8} | {'<=5%':<7} | {'Along(px)':<10} | {'Along(%)':<9} | {'Perp(px)':<9} | {'Perp(%)':<9}")
    lines.append("-" * 88)
    for lm in metrics_dict.get("by_landmark", []):
        m_sd = f"{lm.get('mean', 0.0):.1f}±{lm.get('sd', 0.0):.1f}"
        along_px_str = f"{lm.get('mean_along_px', 0.0):+.2f}" if "mean_along_px" in lm else "-"
        along_pct_str = f"{lm.get('mean_along_pct', 0.0):+.2f}%" if "mean_along_pct" in lm else "-"
        perp_px_str = f"{lm.get('mean_perp_px', 0.0):+.2f}" if "mean_perp_px" in lm else "-"
        perp_pct_str = f"{lm.get('mean_perp_pct', 0.0):+.2f}%" if "mean_perp_pct" in lm else "-"
        lines.append(
            f"{lm.get('index', 0):02d}  {lm.get('name', ''):<8} | {m_sd:<16} | {lm.get('median', 0.0):<8.1f} | {lm.get('share_within_5pct', 0.0):<6.1f}% | {along_px_str:<10} | {along_pct_str:<9} | {perp_px_str:<9} | {perp_pct_str:<9}"
        )

    # Section B.4: Cross-Source Stratification
    lines.append("\n" + "SECTION B.4: CROSS-SOURCE STRATIFICATION (INSTITUTIONAL VS PUBLIC)".center(88))
    lines.append("-" * 88)
    lines.append(f"{'Source':<16} | {'Mean ± SD (%)':<18} | {'Median (%)':<12} | {'Share <= 5%':<14} | {'Landmarks':<10}")
    lines.append("-" * 88)
    for src, s_stats in metrics_dict.get("by_source", {}).items():
        m_sd = f"{s_stats.get('mean', 0.0):.2f} ± {s_stats.get('sd', 0.0):.2f}"
        lines.append(
            f"{src.capitalize():<16} | {m_sd:<18} | {s_stats.get('median', 0.0):<12.2f} | {s_stats.get('share_within_5pct', 0.0):<13.2f}% | {s_stats.get('count', 0):<10}"
        )

    # Section C: Clinical Morphometric Validation
    clin = metrics_dict.get("clinical", {})
    if clin:
        lines.append("\n" + "SECTION C.1: BLAND-ALTMAN AGREEMENT (CONCAVITY % & SHAPE INDEX)".center(88))
        lines.append("-" * 88)
        lines.append(f"{'Parameter':<18} | {'Bias (Mean Diff)':<18} | {'SD Diff':<12} | {'95% Limits of Agreement':<28} | {'MAE':<8}")
        lines.append("-" * 88)
        for vert, ba in clin.get("bland_altman_concavity", {}).items():
            loa_str = f"[{ba.get('loa_lower', 0.0):+.2f}, {ba.get('loa_upper', 0.0):+.2f}]"
            lines.append(
                f"{vert + ' Concavity %':<18} | {ba.get('bias', 0.0):<+18.3f} | {ba.get('sd_diff', 0.0):<12.3f} | {loa_str:<28} | {ba.get('mae', 0.0):<8.3f}"
            )
        for vert, ba in clin.get("bland_altman_shape_index", {}).items():
            loa_str = f"[{ba.get('loa_lower', 0.0):+.2f}, {ba.get('loa_upper', 0.0):+.2f}]"
            lines.append(
                f"{vert + ' Shape Index':<18} | {ba.get('bias', 0.0):<+18.3f} | {ba.get('sd_diff', 0.0):<12.3f} | {loa_str:<28} | {ba.get('mae', 0.0):<8.3f}"
            )

        lines.append("\n" + "SECTION C.2: NOTCH CLASSIFICATION AGREEMENT".center(88))
        lines.append("-" * 88)
        lines.append(f"{'Vertebra':<10} | {'Accuracy (%)':<14} | {'Sensitivity (%)':<16} | {'Specificity (%)':<16} | {'Cohen Kappa (κ)':<18}")
        lines.append("-" * 88)
        for vert, na in clin.get("notch_agreement", {}).items():
            lines.append(
                f"{vert:<10} | {na.get('accuracy', 0.0):<14.2f} | {na.get('sensitivity', 0.0):<16.2f} | {na.get('specificity', 0.0):<16.2f} | {na.get('cohen_kappa', 0.0):<18.3f}"
            )

        lines.append("\n" + "SECTION C.3: VERTEBRAL SHAPE CLASSIFICATION AGREEMENT".center(88))
        lines.append("-" * 88)
        for vert, sa in clin.get("shape_agreement", {}).items():
            lines.append(f" {vert} Shape Agreement Accuracy: {sa.get('accuracy', 0.0):.2f}% (N={sa.get('total', 0)})")
            lines.append(f" Confusion Matrix ({', '.join(sa.get('categories', []))}):")
            cm = sa.get("confusion_matrix", [])
            for row in cm:
                lines.append(f"   {row}")

        bl = clin.get("borderline_analysis", {})
        if bl:
            lines.append("\n" + "SECTION C.4: BORDERLINE SENSITIVITY ANALYSIS (DIAGNOSTIC CONCAVITY CUTOFFS)".center(88))
            lines.append("-" * 88)
            lines.append(f"{'Window':<10} | {'Borderline Cases':<18} | {'Borderline Acc':<16} | {'Clear Cases':<14} | {'Clear Acc':<12} | {'Overall Acc':<12}")
            lines.append("-" * 88)
            for d_key, d_res in bl.items():
                w_str = f"±{d_res.get('delta', 0.0)}%"
                bl_cases = f"{d_res.get('borderline_count', 0)} ({d_res.get('borderline_share_pct', 0.0):.1f}%)"
                lines.append(
                    f"{w_str:<10} | {bl_cases:<18} | {d_res.get('borderline_accuracy_pct', 0.0):<15.2f}% | {d_res.get('outside_borderline_count', 0):<14} | {d_res.get('outside_borderline_accuracy_pct', 0.0):<11.2f}% | {d_res.get('overall_accuracy_pct', 0.0):<11.2f}%"
                )

    lines.append("=" * 88)
    return "\n".join(lines)
