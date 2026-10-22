import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


@dataclass
class ROIBoundingBox:
    """
    Represents an isotropic, strictly square cervical vertebrae bounding box.
    Coordinates (u1, v1) represent the top-left corner in the unpadded image coordinates,
    and D represents the square side length (u2 - u1 = v2 - v1 = D).
    """
    u1: int
    v1: int
    u2: int
    v2: int
    D: int
    center_u: float
    center_v: float
    span: float
    img_w: int
    img_h: int
    is_fallback: bool = False
    fallback_reason: str = ""

    @property
    def pad_margins(self) -> Tuple[int, int, int, int]:
        """Returns (pad_left, pad_top, pad_right, pad_bottom) needed to crop without out-of-bound errors."""
        pad_left = max(0, -self.u1)
        pad_top = max(0, -self.v1)
        pad_right = max(0, self.u2 - self.img_w)
        pad_bottom = max(0, self.v2 - self.img_h)
        return pad_left, pad_top, pad_right, pad_bottom


def validate_stage1_box(
    coords: Union[np.ndarray, torch.Tensor],
    img_w: int = 640,
    img_h: int = 640,
    min_span: float = 60.0,
    max_span: float = 350.0,
    check_anatomical_bounds: bool = True,
) -> Tuple[bool, str]:
    """
    Performs anatomical sanity checks on Stage 1 predicted landmarks.

    Args:
        coords: [N, 2] landmarks in normalized [0, 1] or pixel coordinates.
        img_w: Image canvas width.
        img_h: Image canvas height.
        min_span: Minimum plausible vertebrae span in pixels.
        max_span: Maximum plausible vertebrae span in pixels.
        check_anatomical_bounds: Whether to verify expected quadrant localization.

    Returns:
        (is_valid, failure_reason)
    """
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords)

    if coords_np.ndim != 2 or coords_np.shape[1] != 2:
        return False, f"Expected shape [N, 2], got {coords_np.shape}"

    # Check for NaN / Inf
    if np.isnan(coords_np).any() or np.isinf(coords_np).any():
        return False, "Coordinates contain NaN or Inf"

    # Identify valid points
    valid_mask = (coords_np[:, 0] >= 0) & (coords_np[:, 1] >= 0)
    if not np.any(valid_mask):
        return False, "No valid landmarks (all negative)"

    valid_pts = coords_np[valid_mask]
    if len(valid_pts) < 3:
        return False, f"Too few valid landmarks ({len(valid_pts)} < 3)"

    # Determine if normalized [0, 1] or pixel scale
    is_normalized = (valid_pts[:, 0].max() <= 1.05 and valid_pts[:, 1].max() <= 1.05)
    xs = valid_pts[:, 0] * (img_w if is_normalized else 1.0)
    ys = valid_pts[:, 1] * (img_h if is_normalized else 1.0)

    u_min, u_max = float(xs.min()), float(xs.max())
    v_min, v_max = float(ys.min()), float(ys.max())

    span_u = u_max - u_min
    span_v = v_max - v_min
    span = max(span_u, span_v)

    if span < min_span:
        return False, f"Vertebrae span too small ({span:.1f} px < {min_span:.1f} px)"
    if span > max_span:
        return False, f"Vertebrae span too large ({span:.1f} px > {max_span:.1f} px)"

    if check_anatomical_bounds:
        c_v = (v_min + v_max) / 2.0
        if c_v < img_h * 0.15 or c_v > img_h * 0.90:
            return False, f"Vertebrae vertical center out of bounds ({c_v:.1f} vs [0.15, 0.90]*H)"

    return True, ""


def apply_anatomical_height_prior(
    coords: Union[np.ndarray, torch.Tensor],
    img_w: int = 640,
    img_h: int = 640,
    min_c4_to_c3_ratio: float = 0.75,
) -> Tuple[Union[np.ndarray, torch.Tensor], bool]:
    """
    Applies the Anatomical Vertebra Height Prior.
    In human cervical spine anatomy (C2-C4), vertebra C4 is physiologically of comparable
    height to C3 (H_C4 >= 0.85 * H_C3).

    If lower neck radiographic occlusion, shoulder attenuation, or collimator cut-off causes
    C4 inferior landmarks (C4_PI: 10, C4_IC: 11, C4_AI: 12) to collapse upward into the C4
    body or into C3 (H_C4 < 0.75 * H_C3), this prior computes the reliable height of C3
    and ensures C4 inferior reaches its true physiological lower boundary:
        h_C3 = max(y[5,6,7]) - min(y[3,4])
        min_c4_h = min_c4_to_c3_ratio * h_C3
        c4_floor_y = max(y[8,9]) + min_c4_h

    Args:
        coords: [N, 2] landmarks in normalized [0, 1] or pixel scale.
        img_w: Canvas width.
        img_h: Canvas height.
        min_c4_to_c3_ratio: Minimum physiological threshold ratio of C4 height to C3 height (default 0.75).

    Returns:
        (corrected_coords, prior_applied)
    """
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy().copy()
        is_torch = True
    else:
        coords_np = np.asarray(coords).copy()
        is_torch = False

    if coords_np.ndim != 2 or coords_np.shape[0] != 13 or coords_np.shape[1] != 2:
        return coords, False

    valid_mask = (coords_np[:, 0] >= 0) & (coords_np[:, 1] >= 0)
    c3_sup = [3, 4]
    c3_inf = [5, 6, 7]
    c4_sup = [8, 9]
    c4_inf = [10, 11, 12]

    # Require C3 and C4 superior to be valid to establish anatomical reference
    if not all(valid_mask[k] for k in c3_sup + c3_inf + c4_sup):
        return coords, False

    is_normalized = bool(coords_np[valid_mask, 0].max() <= 1.05 and coords_np[valid_mask, 1].max() <= 1.05)
    scale_y = float(img_h) if is_normalized else 1.0

    all_ys = coords_np[:, 1] * scale_y
    c3_sup_y = min(float(all_ys[k]) for k in c3_sup)
    c3_inf_y = max(float(all_ys[k]) for k in c3_inf)
    h_c3 = c3_inf_y - c3_sup_y

    # Plausible vertebra height check (at least 15 pixels)
    if h_c3 < 15.0:
        return coords, False

    c4_sup_y = max(float(all_ys[k]) for k in c4_sup)
    expected_c4_h = min_c4_to_c3_ratio * h_c3
    c4_floor_y = c4_sup_y + expected_c4_h

    # Current predicted C4 inferior extent
    valid_c4_inf = [k for k in c4_inf if valid_mask[k]]
    current_c4_inf_max_y = max(float(all_ys[k]) for k in valid_c4_inf) if valid_c4_inf else c4_sup_y

    prior_applied = False
    # If C4 inferior is collapsed above the expected anatomical floor
    if current_c4_inf_max_y < c4_floor_y:
        deficit_y = c4_floor_y - current_c4_inf_max_y
        for k in c4_inf:
            if valid_mask[k]:
                coords_np[k, 1] += (deficit_y / scale_y)
            else:
                c4_sup_match = 8 if k == 10 else (9 if k == 12 else 8)
                coords_np[k, 0] = coords_np[c4_sup_match, 0]
                coords_np[k, 1] = c4_floor_y / scale_y
        prior_applied = True

    if is_torch:
        res = torch.from_numpy(coords_np).to(device=coords.device, dtype=coords.dtype)
        return res, prior_applied
    return coords_np, prior_applied


def compute_roi_bbox(
    coords: Optional[Union[np.ndarray, torch.Tensor]],
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.18,
    min_size: int = 180,
    jitter: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.RandomState] = None,
    fallback_anchor: Optional[Tuple[float, float, float]] = None,
    check_anatomical_bounds: bool = True,
    use_anatomical_prior: bool = True,
) -> ROIBoundingBox:
    """
    Computes an isotropic, strictly square Region of Interest (ROI) bounding box
    centered on the valid cervical landmarks (C2-C4).

    Args:
        coords: Landmark coordinates [N, 2] in normalized scale [0, 1] or pixel scale.
        img_w: Width of the image canvas.
        img_h: Height of the image canvas.
        margin_ratio: Padding ratio added to all sides (D = S * (1 + 2 * margin_ratio)).
        min_size: Minimum square dimension D in pixels.
        jitter: Optional dict with {"trans": float, "scale": float} for training jitter.
                e.g. {"trans": 0.06, "scale": 0.08}
        rng: Optional np.random.RandomState for deterministic jittering.
        fallback_anchor: Optional tuple (cx_norm, cy_norm, span_norm) for anchor fallback.
        check_anatomical_bounds: Whether to verify anatomical quadrant placement.

    Returns:
        ROIBoundingBox containing (u1, v1, u2, v2, D, center_u, center_v, ...).
    """
    is_valid = False
    fallback_reason = ""

    coords_for_box = coords
    if coords is not None and use_anatomical_prior:
        coords_for_box, _ = apply_anatomical_height_prior(coords, img_w=img_w, img_h=img_h)

    if coords_for_box is not None:
        is_valid, fallback_reason = validate_stage1_box(
            coords_for_box, img_w, img_h, check_anatomical_bounds=check_anatomical_bounds
        )

    if not is_valid:
        # Fallback to anatomical default anchor box
        if fallback_anchor is not None:
            c_u = fallback_anchor[0] * img_w
            c_v = fallback_anchor[1] * img_h
            span = fallback_anchor[2] * min(img_w, img_h)
        else:
            # Cervical spine standard default prior: middle-left region, mid-lower quadrant
            c_u = img_w * 0.42
            c_v = img_h * 0.52
            span = 170.0

        D = int(round(max(span * (1.0 + 2.0 * margin_ratio), float(min_size))))
        r = D / 2.0
        u1 = int(round(c_u - r))
        v1 = int(round(c_v - r))
        return ROIBoundingBox(
            u1=u1,
            v1=v1,
            u2=u1 + D,
            v2=v1 + D,
            D=D,
            center_u=c_u,
            center_v=c_v,
            span=span,
            img_w=img_w,
            img_h=img_h,
            is_fallback=True,
            fallback_reason=fallback_reason or "Missing or invalid coordinates",
        )

    # Process valid coordinates (with anatomical prior applied)
    if isinstance(coords_for_box, torch.Tensor):
        coords_np = coords_for_box.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords_for_box)

    valid_mask = (coords_np[:, 0] >= 0) & (coords_np[:, 1] >= 0)
    valid_pts = coords_np[valid_mask]

    is_normalized = bool(valid_pts[:, 0].max() <= 1.05 and valid_pts[:, 1].max() <= 1.05)
    xs = valid_pts[:, 0] * (img_w if is_normalized else 1.0)
    ys = valid_pts[:, 1] * (img_h if is_normalized else 1.0)

    u_min, u_max = float(xs.min()), float(xs.max())
    v_min, v_max = float(ys.min()), float(ys.max())

    span_u = u_max - u_min
    span_v = v_max - v_min
    span = max(span_u, span_v)

    c_u = (u_min + u_max) / 2.0
    c_v = (v_min + v_max) / 2.0

    # Base square dimension D
    base_D = max(span * (1.0 + 2.0 * margin_ratio), float(min_size))

    # Apply optional training BBox jitter
    if jitter is not None:
        trans_max = jitter.get("trans", 0.0)
        scale_max = jitter.get("scale", 0.0)
        r_gen = rng if rng is not None else np.random

        # Random center translation within [-trans_max, +trans_max] * base_D
        if trans_max > 0.0:
            shift_u = r_gen.uniform(-trans_max, trans_max) * base_D
            shift_v = r_gen.uniform(-trans_max, trans_max) * base_D
            c_u += shift_u
            c_v += shift_v

        # Random scale factor within [1 - scale_max, 1 + scale_max]
        if scale_max > 0.0:
            scale_factor = r_gen.uniform(1.0 - scale_max, 1.0 + scale_max)
            base_D *= scale_factor

    D = int(round(base_D))
    # Guarantee even/integer dimension for exact center-subtraction symmetry
    r = D / 2.0
    u1 = int(round(c_u - r))
    v1 = int(round(c_v - r))
    u2 = u1 + D
    v2 = v1 + D

    return ROIBoundingBox(
        u1=u1,
        v1=v1,
        u2=u2,
        v2=v2,
        D=D,
        center_u=c_u,
        center_v=c_v,
        span=span,
        img_w=img_w,
        img_h=img_h,
        is_fallback=False,
        fallback_reason="",
    )


def crop_roi_with_padding(
    image: np.ndarray,
    bbox: ROIBoundingBox,
    target_size: int = 448,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101,
    border_val: Union[int, Tuple[int, int, int]] = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Crops an isotropic, strictly square region of interest (ROI) from image with boundary padding,
    and resizes it to target_size x target_size.

    Args:
        image: Source image [H, W, C] or [H, W] (uint8 or float32).
        bbox: ROIBoundingBox instance defining the square crop.
        target_size: Target square pixel size (e.g. 448 or 672).
        interpolation: OpenCV interpolation mode (default cv2.INTER_LINEAR).
        border_mode: OpenCV border mode when bbox extends outside image (default cv2.BORDER_REFLECT_101).
        border_val: Border constant value if border_mode is cv2.BORDER_CONSTANT.

    Returns:
        cropped_roi: Resized square image [target_size, target_size, C].
        meta: Dictionary containing geometric mapping metadata.
    """
    img_h, img_w = image.shape[:2]
    u1, v1, u2, v2 = bbox.u1, bbox.v1, bbox.u2, bbox.v2
    D = bbox.D

    pad_left, pad_top, pad_right, pad_bottom = bbox.pad_margins

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        if border_mode == cv2.BORDER_CONSTANT:
            padded_img = cv2.copyMakeBorder(
                image,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                border_mode,
                value=border_val,
            )
        else:
            padded_img = cv2.copyMakeBorder(
                image,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                border_mode,
            )
        # Shift crop window by the applied padding
        crop_u1 = u1 + pad_left
        crop_v1 = v1 + pad_top
        crop_u2 = crop_u1 + D
        crop_v2 = crop_v1 + D
        roi_patch = padded_img[crop_v1:crop_v2, crop_u1:crop_u2]
    else:
        roi_patch = image[v1:v2, u1:u2]

    # Verify that roi_patch is exactly D x D
    if roi_patch.shape[0] != D or roi_patch.shape[1] != D:
        roi_patch = cv2.resize(roi_patch, (D, D), interpolation=cv2.INTER_NEAREST)

    if (D, D) != (target_size, target_size):
        resized_roi = cv2.resize(
            roi_patch, (target_size, target_size), interpolation=interpolation
        )
    else:
        resized_roi = roi_patch

    meta = {
        "u1": u1,
        "v1": v1,
        "D": D,
        "target_size": target_size,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
        "img_w": img_w,
        "img_h": img_h,
        "scale_roi_to_target": float(target_size) / float(D),
    }

    return resized_roi, meta


def transform_coords_to_roi(
    coords: Union[np.ndarray, torch.Tensor],
    bbox: ROIBoundingBox,
    target_size: Optional[int] = None,
    is_normalized_input: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transforms coordinates from full canvas space into isotropic normalized ROI space [0, 1].

    Includes the exact OpenCV half-pixel center correction when target_size is specified:
      (x_crop + 0.5) = (x_patch + 0.5) * (target_size / D)
      x_crop = (x_canvas - u1 + 0.5) * (target_size / D) - 0.5
      x_roi_norm = x_crop / target_size

    Args:
        coords: [..., 2] array/tensor in [x, y] format. Missing points must be negative (e.g. [-1, -1]).
        bbox: ROIBoundingBox instance.
        target_size: Target square crop resolution (e.g. 448). If None, continuous D-scale is used without resampling shift.
        is_normalized_input: True if coords are normalized in [0, 1], False if pixel units.

    Returns:
        roi_coords: [..., 2] in normalized ROI space [0, 1], with invalid points preserved as [-1.0, -1.0].
    """
    if isinstance(coords, torch.Tensor):
        is_torch = True
        device = coords.device
        dtype = coords.dtype
        coords_arr = coords
    else:
        is_torch = False
        coords_arr = torch.from_numpy(np.asarray(coords))
        device = None
        dtype = None

    orig_shape = coords_arr.shape
    flat_coords = coords_arr.reshape(-1, 2).clone()

    valid_mask = (flat_coords[:, 0] >= 0) & (flat_coords[:, 1] >= 0)

    if is_normalized_input:
        scale_w = float(bbox.img_w)
        scale_h = float(bbox.img_h)
        px_u = flat_coords[:, 0] * scale_w
        px_v = flat_coords[:, 1] * scale_h
    else:
        px_u = flat_coords[:, 0]
        px_v = flat_coords[:, 1]

    u1 = float(bbox.u1)
    v1 = float(bbox.v1)
    D = float(bbox.D)

    if target_size is not None and target_size > 0:
        K = float(target_size)
        # Exact OpenCV half-pixel alignment:
        # x_crop = (px_u - u1 + 0.5) * (K / D) - 0.5
        # x_roi_norm = x_crop / K = (px_u - u1 + 0.5) / D - 0.5 / K
        roi_x = (px_u - u1 + 0.5) / D - 0.5 / K
        roi_y = (px_v - v1 + 0.5) / D - 0.5 / K
    else:
        roi_x = (px_u - u1) / D
        roi_y = (px_v - v1) / D

    out_coords = torch.full_like(flat_coords, -1.0)
    out_coords[valid_mask, 0] = roi_x[valid_mask]
    out_coords[valid_mask, 1] = roi_y[valid_mask]

    out_coords = out_coords.reshape(orig_shape)

    if is_torch:
        return out_coords.to(device=device, dtype=dtype)
    else:
        return out_coords.cpu().numpy()


def inverse_transform_coords(
    roi_coords: Union[np.ndarray, torch.Tensor],
    bbox: ROIBoundingBox,
    target_size: Optional[int] = None,
    output_normalized: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Inversely transforms coordinates from normalized ROI space [0, 1] back to full canvas space.

    Includes the exact OpenCV half-pixel center correction when target_size is specified:
      x_canvas = u1 + (x_roi_norm * target_size + 0.5) * (D / target_size) - 0.5
               = u1 + x_roi_norm * D + 0.5 * (D / target_size - 1)

    Args:
        roi_coords: [..., 2] in normalized ROI space [0, 1].
        bbox: ROIBoundingBox instance.
        target_size: Target square crop resolution (e.g. 448). If None, continuous D-scale is used without resampling shift.
        output_normalized: True to output normalized [0, 1] on full canvas, False for pixels.

    Returns:
        canvas_coords: [..., 2] in canvas space, with invalid points preserved as [-1.0, -1.0].
    """
    if isinstance(roi_coords, torch.Tensor):
        is_torch = True
        device = roi_coords.device
        dtype = roi_coords.dtype
        coords_arr = roi_coords
    else:
        is_torch = False
        coords_arr = torch.from_numpy(np.asarray(roi_coords))
        device = None
        dtype = None

    orig_shape = coords_arr.shape
    flat_coords = coords_arr.reshape(-1, 2).clone()

    valid_mask = (flat_coords[:, 0] >= 0) & (flat_coords[:, 1] >= 0)

    u1 = float(bbox.u1)
    v1 = float(bbox.v1)
    D = float(bbox.D)

    if target_size is not None and target_size > 0:
        K = float(target_size)
        # Exact OpenCV half-pixel center inverse:
        # px_u = u1 + flat_coords[:, 0] * D + 0.5 * (D / K - 1.0)
        px_u = u1 + flat_coords[:, 0] * D + 0.5 * (D / K - 1.0)
        px_v = v1 + flat_coords[:, 1] * D + 0.5 * (D / K - 1.0)
    else:
        px_u = u1 + flat_coords[:, 0] * D
        px_v = v1 + flat_coords[:, 1] * D

    if output_normalized:
        out_u = px_u / float(bbox.img_w)
        out_v = px_v / float(bbox.img_h)
    else:
        out_u = px_u
        out_v = px_v

    out_coords = torch.full_like(flat_coords, -1.0)
    out_coords[valid_mask, 0] = out_u[valid_mask]
    out_coords[valid_mask, 1] = out_v[valid_mask]

    out_coords = out_coords.reshape(orig_shape)

    if is_torch:
        return out_coords.to(device=device, dtype=dtype)
    else:
        return out_coords.cpu().numpy()


def native_to_canvas_coords(
    native_coords: np.ndarray,
    orig_w: int,
    orig_h: int,
    canvas_size: int = 640,
) -> np.ndarray:
    """
    Maps coordinates from native radiograph pixel space to letterboxed canvas coordinates.
    """
    scale = float(canvas_size) / max(orig_w, orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (canvas_size - new_w) // 2
    pad_y = (canvas_size - new_h) // 2
    scale_x = float(new_w) / float(orig_w) if orig_w > 0 else 1.0
    scale_y = float(new_h) / float(orig_h) if orig_h > 0 else 1.0

    canvas_coords = np.full_like(native_coords, -1.0)
    valid_mask = (native_coords[:, 0] >= 0) & (native_coords[:, 1] >= 0)

    canvas_coords[valid_mask, 0] = native_coords[valid_mask, 0] * scale_x + pad_x
    canvas_coords[valid_mask, 1] = native_coords[valid_mask, 1] * scale_y + pad_y

    return canvas_coords


def canvas_to_native_coords(
    canvas_coords: np.ndarray,
    orig_w: int,
    orig_h: int,
    canvas_size: int = 640,
) -> np.ndarray:
    """
    Inversely maps coordinates from letterboxed canvas pixel space back to native radiograph pixel space.
    """
    scale = float(canvas_size) / max(orig_w, orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (canvas_size - new_w) // 2
    pad_y = (canvas_size - new_h) // 2
    scale_x = float(new_w) / float(orig_w) if orig_w > 0 else 1.0
    scale_y = float(new_h) / float(orig_h) if orig_h > 0 else 1.0

    native_coords = np.full_like(canvas_coords, -1.0)
    valid_mask = (canvas_coords[:, 0] >= 0) & (canvas_coords[:, 1] >= 0)

    native_coords[valid_mask, 0] = np.clip((canvas_coords[valid_mask, 0] - pad_x) / scale_x, 0.0, float(orig_w))
    native_coords[valid_mask, 1] = np.clip((canvas_coords[valid_mask, 1] - pad_y) / scale_y, 0.0, float(orig_h))

    return native_coords
