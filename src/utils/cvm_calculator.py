"""
Pure Geometric Percent-Based CVM Engine.
Computes scale-invariant CVM stages and morphometric measurements
using pure relative ratios and perpendicular drops.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class Point:
    x: float
    y: float


@dataclass
class VertebraC2:
    inferior_posterior: Point
    inferior_concavity: Point
    inferior_anterior: Point


@dataclass
class VertebraC3C4:
    inferior_posterior: Point
    inferior_concavity: Point
    inferior_anterior: Point
    superior_posterior: Point
    superior_anterior: Point


@dataclass
class CVMInput:
    c2: VertebraC2
    c3: VertebraC3C4
    c4: VertebraC3C4


def euclidean_distance(p1: Point, p2: Point) -> float:
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def perpendicular_distance(point: Point, line_start: Point, line_end: Point) -> float:
    dx = line_end.x - line_start.x
    dy = line_end.y - line_start.y
    length = math.hypot(dx, dy)
    if length == 0:
        return euclidean_distance(point, line_start)
    area_2x = abs(dy * point.x - dx * point.y + line_end.x * line_start.y - line_end.y * line_start.x)
    return area_2x / length


def signed_concavity_depth(point: Point, line_start: Point, line_end: Point) -> float:
    """
    Computes true anatomical concavity depth into the vertebral body.
    In image coordinates (Y increases downward):
    - Points superior to the baseline (into the body, smaller Y) have positive concavity depth.
    - Points inferior to the baseline (bulging downward as a lump/convexity) have 0.0 concavity depth.
    """
    dx = line_end.x - line_start.x
    dy = line_end.y - line_start.y
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0

    # Normal vector pointing superiorly (upward, towards smaller Y)
    s = 1.0 if dx >= 0 else -1.0
    nx = s * dy
    ny = s * (-dx)

    vx = point.x - line_start.x
    vy = point.y - line_start.y

    up_proj = vx * nx + vy * ny
    if up_proj <= 0:
        # Landmark is on or below the baseline (lump / protrusion) -> not a concavity
        return 0.0

    area_2x = abs(dy * point.x - dx * point.y + line_end.x * line_start.y - line_end.y * line_start.x)
    return area_2x / length


def calculate_shear_angle(pi: Point, ai: Point, as_pt: Point) -> float:
    """
    Computes rotation-invariant deviation of anterior border from orthogonal (90 degrees).
    Interior corner angle alpha at AI between vectors (PI - AI) and (AS - AI).
    Shear angle = |90.0 - alpha|.
    """
    v1_x = pi.x - ai.x
    v1_y = pi.y - ai.y
    v2_x = as_pt.x - ai.x
    v2_y = as_pt.y - ai.y

    m1 = math.hypot(v1_x, v1_y)
    m2 = math.hypot(v2_x, v2_y)
    if m1 == 0 or m2 == 0:
        return 0.0

    dot = v1_x * v2_x + v1_y * v2_y
    cos_val = max(-1.0, min(1.0, dot / (m1 * m2)))
    corner_angle = math.degrees(math.acos(cos_val))
    return abs(90.0 - corner_angle)


@dataclass
class PercentEngineConfig:
    engine_mode: str = "axis"  # "axis" (Anatomical Midline Axis Calibrated) or "legacy"
    c2_concavity_percent: float = 5.0
    c3_concavity_percent: float = 5.0
    c4_concavity_percent: float = 6.0
    concavity_order_guard: bool = True
    enable_smart_order_guard: bool = True
    enable_biological_guard: bool = True
    c3_biological_guard_percent: float = 5.5
    enable_contextual_tolerance: bool = False
    enable_biological_dominance: bool = True
    trapezoid_si_threshold: float = 0.75
    rect_horizontal_si_threshold: float = 0.85
    square_si_lower: float = 0.94
    square_si_upper: float = 1.08
    rect_vertical_si_threshold: float = 1.08
    trapezoid_taper_threshold: float = 1.15
    trapezoid_width_taper_threshold: float = 1.20
    width_taper_threshold: float = 1.15
    max_square_shear_angle: float = 18.0
    joint_square_si_threshold: float = 0.94
    # Calibrated thresholds for Anatomical Midline Axis projection
    axis_square_si_lower: float = 0.94
    axis_square_si_upper: float = 1.08


class PercentCVMEngine:
    def __init__(self, config: Optional[PercentEngineConfig] = None):
        self.config = config or PercentEngineConfig()

    def evaluate_sample(self, input_data: CVMInput) -> Dict[str, Any]:
        # C2 concavity
        c2_ip = input_data.c2.inferior_posterior
        c2_ic = input_data.c2.inferior_concavity
        c2_ia = input_data.c2.inferior_anterior
        c2_width = euclidean_distance(c2_ip, c2_ia)
        c2_depth = signed_concavity_depth(c2_ic, c2_ip, c2_ia)
        c2_pct = (c2_depth / c2_width * 100.0) if c2_width > 0 else 0.0

        # C3 concavity & shape
        c3_ip = input_data.c3.inferior_posterior
        c3_ic = input_data.c3.inferior_concavity
        c3_ia = input_data.c3.inferior_anterior
        c3_width = euclidean_distance(c3_ip, c3_ia)
        c3_depth = signed_concavity_depth(c3_ic, c3_ip, c3_ia)
        c3_pct = (c3_depth / c3_width * 100.0) if c3_width > 0 else 0.0
        c3_shape = self._classify_shape(input_data.c3, concavity_pct=c3_pct)

        # C4 concavity & shape
        c4_ip = input_data.c4.inferior_posterior
        c4_ic = input_data.c4.inferior_concavity
        c4_ia = input_data.c4.inferior_anterior
        c4_width = euclidean_distance(c4_ip, c4_ia)
        c4_depth = signed_concavity_depth(c4_ic, c4_ip, c4_ia)
        c4_pct = (c4_depth / c4_width * 100.0) if c4_width > 0 else 0.0
        c4_shape = self._classify_shape(input_data.c4, concavity_pct=c4_pct)

        # Notches
        n2 = 1 if c2_pct >= self.config.c2_concavity_percent else 0

        n3 = 1 if c3_pct >= self.config.c3_concavity_percent else 0
        n4 = 1 if c4_pct >= self.config.c4_concavity_percent else 0
        if self.config.enable_contextual_tolerance and n4 == 0:
            if c4_pct >= 5.5 and c2_pct >= 8.0 and c3_pct >= 8.0:
                n4 = 1

        # Monotonicity Order Guard
        c4_suppressed = False
        c3_suppressed = False
        c2_guarded = False
        anomaly_detected: Optional[str] = None

        if self.config.concavity_order_guard:
            if self.config.enable_smart_order_guard and n3 == 1 and n4 == 1:
                if n2 == 0:
                    anomaly_detected = "C2 under-annotation resolved: definitive concavities on C3 & C4 indicate C2 maturity."
                    n2 = 1
                    c2_guarded = True
            elif n4 == 1 and n3 == 0:
                n4 = 0
                c4_suppressed = True
            elif n3 == 1 and n2 == 0:
                bio_guard_thresh = getattr(self.config, "c3_biological_guard_percent", 5.5)
                bio_guard_enabled = getattr(self.config, "enable_biological_guard", True)
                if self.config.enable_smart_order_guard and bio_guard_enabled and c3_pct >= bio_guard_thresh:
                    anomaly_detected = f"Biological guard: Obvious C3 concavity ({c3_pct:.2f}% ≥ {bio_guard_thresh:.1f}%) indicates C2 maturity (C2 flat ignored)."
                    n2 = 1
                    c2_guarded = True
                elif self.config.enable_smart_order_guard and c3_pct >= 7.5:
                    anomaly_detected = "C2 shadow artifact resolved: deep C3 notch (>=7.5%) indicates C2 maturity."
                    n2 = 1
                    c2_guarded = True
                else:
                    n3 = 0
                    c3_suppressed = True

        # Shape analysis
        s3 = c3_shape["shape"]
        s4 = c4_shape["shape"]
        si3 = c3_shape["shape_index"]
        si4 = c4_shape["shape_index"]
        mean_si = (si3 + si4) / 2.0

        is_vertical = (s3 == "Rectangular Vertical" or s4 == "Rectangular Vertical")
        is_square = (s3 == "Square" or s4 == "Square")

        if n2 == 1 and n3 == 1 and n4 == 1:
            if is_vertical:
                stage_num = 6
            elif is_square:
                stage_num = 5
            else:
                stage_num = 4
        elif n2 == 1 and n3 == 1:
            stage_num = 3
        elif n2 == 1:
            stage_num = 2
        else:
            stage_num = 1

        if self.config.enable_biological_dominance:
            if (s3 == "Rectangular Vertical" and s4 == "Rectangular Vertical") and (n2 == 1 or n3 == 1):
                if stage_num < 6:
                    anomaly_detected = anomaly_detected or "Biological dominance: Both bodies rectangular vertical indicate CS6 maturity."
                stage_num = max(stage_num, 6)
            elif is_vertical and (n2 == 1 and n3 == 1):
                if stage_num < 6:
                    anomaly_detected = anomaly_detected or "Biological dominance: Rectangular vertical body indicates CS6 maturity."
                stage_num = max(stage_num, 6)
            elif is_square and (n2 == 1 and n3 == 1):
                if stage_num < 5:
                    anomaly_detected = anomaly_detected or "Biological dominance: Square body indicates CS5 maturity."
                stage_num = max(stage_num, 5)

        return {
            "stage": f"CS{stage_num}",
            "stage_number": stage_num,
            "notches": {"C2": bool(n2), "C3": bool(n3), "C4": bool(n4)},
            "percentages": {
                "C2": round(c2_pct, 2),
                "C3": round(c3_pct, 2),
                "C4": round(c4_pct, 2),
            },
            "shapes": {
                "C3": c3_shape,
                "C4": c4_shape,
            },
            "order_guard": {
                "c2_guarded": c2_guarded,
                "c3_suppressed": c3_suppressed,
                "c4_suppressed": c4_suppressed,
            },
            "anomaly_detected": anomaly_detected,
            "mean_shape_index": round(mean_si, 3),
            "engine_mode": getattr(self.config, "engine_mode", "legacy"),
        }

    def _classify_shape(self, vertebra: VertebraC3C4, concavity_pct: float = 0.0) -> Dict[str, Any]:
        pi = vertebra.inferior_posterior
        ai = vertebra.inferior_anterior
        ps = vertebra.superior_posterior
        as_pt = vertebra.superior_anterior

        mode = getattr(self.config, "engine_mode", "legacy")

        if mode == "axis":
            # Anatomical Midline Axis Mode
            # Midpoints of inferior (base) and superior borders
            m_inf_x = (pi.x + ai.x) / 2.0
            m_inf_y = (pi.y + ai.y) / 2.0
            m_sup_x = (ps.x + as_pt.x) / 2.0
            m_sup_y = (ps.y + as_pt.y) / 2.0

            # Anatomical longitudinal axis vector (height axis)
            vx = m_sup_x - m_inf_x
            vy = m_sup_y - m_inf_y
            h_mid = math.hypot(vx, vy)
            if h_mid == 0:
                ux, uy = 0.0, 1.0
            else:
                ux = vx / h_mid
                uy = vy / h_mid

            # Orthogonal transverse axis unit vector (width axis, pointing anteriorly)
            wx = -uy
            wy = ux
            base_dx = ai.x - pi.x
            base_dy = ai.y - pi.y
            if (wx * base_dx + wy * base_dy) < 0:
                wx, wy = -wx, -wy

            # Projected widths along transverse axis
            wi = abs((ai.x - pi.x) * wx + (ai.y - pi.y) * wy)
            ws = abs((as_pt.x - ps.x) * wx + (as_pt.y - ps.y) * wy)
            mean_width = (wi + ws) / 2.0
            width_sum = ws + wi

            # Projected heights along longitudinal axis
            hp = abs((ps.x - pi.x) * ux + (ps.y - pi.y) * uy)
            ha = abs((as_pt.x - ai.x) * ux + (as_pt.y - ai.y) * uy)
            mean_height = (hp + ha) / 2.0

            si = mean_height / mean_width if mean_width > 0 else 1.0

            h_max = max(hp, ha)
            h_min = min(hp, ha)
            taper_factor = (h_max / h_min) if h_min > 0 else 1.0
            tr_h = ha / hp if hp > 0 else 1.0

            w_max = max(ws, wi)
            w_min = min(ws, wi)
            width_taper_factor = (w_max / w_min) if w_min > 0 else 1.0
            tr_w = (ws / wi) if wi > 0 else 1.0
            width_disparity = abs(wi - ws)
            shear_angle = calculate_shear_angle(pi, ai, as_pt)

            sq_lower = getattr(self.config, "axis_square_si_lower", 0.94)
            sq_upper = getattr(self.config, "axis_square_si_upper", 1.08)
            trap_taper = getattr(self.config, "trapezoid_taper_threshold", 1.14)
            trap_width_taper = getattr(self.config, "trapezoid_width_taper_threshold", 1.18)
            trap_si = getattr(self.config, "trapezoid_si_threshold", 0.75)

            if si > sq_upper:
                shape = "Rectangular Vertical"
            elif (taper_factor >= trap_taper or width_taper_factor >= trap_width_taper) and si <= 0.96:
                shape = "Trapezoidal"
            elif sq_lower <= si <= sq_upper:
                if taper_factor < trap_taper and width_taper_factor < trap_width_taper and shear_angle <= 20.0:
                    shape = "Square"
                elif taper_factor >= trap_taper or width_taper_factor >= trap_width_taper:
                    shape = "Trapezoidal"
                else:
                    shape = "Rectangular Horizontal"
            else:
                shape = "Rectangular Horizontal"

            axis_angle = math.degrees(math.atan2(vy, vx))
            return {
                "shape": shape,
                "shape_index": round(si, 3),
                "mean_height": round(mean_height, 1),
                "mean_width": round(mean_width, 1),
                "taper_ratio": round(tr_h, 3),
                "height_taper_ratio": round(tr_h, 3),
                "width_taper_ratio": round(tr_w, 3),
                "width_ratio": round(tr_w, 3),
                "width_disparity": round(width_disparity, 1),
                "taper_factor": round(taper_factor, 3),
                "width_taper_factor": round(width_taper_factor, 3),
                "shear_angle": round(shear_angle, 1),
                "anterior_height": round(ha, 1),
                "posterior_height": round(hp, 1),
                "inferior_width": round(wi, 1),
                "superior_width": round(ws, 1),
                "engine_mode": "axis",
                "axis_angle": round(axis_angle, 1),
                "axis_vector": {"x": round(vx, 3), "y": round(vy, 3)},
                "midline_start": {"x": round(m_inf_x, 1), "y": round(m_inf_y, 1)},
                "midline_end": {"x": round(m_sup_x, 1), "y": round(m_sup_y, 1)},
            }

        # Legacy Mode: Pure Euclidean edge distances
        hp = euclidean_distance(ps, pi)
        ha = euclidean_distance(as_pt, ai)
        ws = euclidean_distance(ps, as_pt)
        wi = euclidean_distance(pi, ai)

        width_sum = ws + wi
        si = (ha + hp) / width_sum if width_sum > 0 else 1.0

        h_max = max(hp, ha)
        h_min = min(hp, ha)
        taper_factor = (h_max / h_min) if h_min > 0 else 1.0
        tr_h = ha / hp if hp > 0 else 1.0

        w_max = max(ws, wi)
        w_min = min(ws, wi)
        width_taper_factor = (w_max / w_min) if w_min > 0 else 1.0
        tr_w = (ws / wi) if wi > 0 else 1.0
        width_disparity = abs(wi - ws)
        shear_angle = calculate_shear_angle(pi, ai, as_pt)

        if si > self.config.square_si_upper:
            shape = "Rectangular Vertical"
        elif si >= self.config.square_si_lower:
            # Guard against borderline square (si < 0.96) with excessive shear or basal flaring
            if si < 0.96 and (shear_angle > self.config.max_square_shear_angle or width_taper_factor > self.config.width_taper_threshold):
                shape = "Rectangular Horizontal"
            else:
                shape = "Square"
        elif concavity_pct < 5.0 and (taper_factor >= self.config.trapezoid_taper_threshold or width_taper_factor >= self.config.trapezoid_width_taper_threshold or si <= self.config.trapezoid_si_threshold):
            shape = "Trapezoidal"
        else:
            shape = "Rectangular Horizontal"

        return {
            "shape": shape,
            "shape_index": round(si, 3),
            "taper_ratio": round(tr_h, 3),
            "height_taper_ratio": round(tr_h, 3),
            "width_taper_ratio": round(tr_w, 3),
            "width_ratio": round(tr_w, 3),
            "width_disparity": round(width_disparity, 1),
            "taper_factor": round(taper_factor, 3),
            "width_taper_factor": round(width_taper_factor, 3),
            "shear_angle": round(shear_angle, 1),
            "anterior_height": round(ha, 1),
            "posterior_height": round(hp, 1),
            "inferior_width": round(wi, 1),
            "superior_width": round(ws, 1),
            "engine_mode": "legacy",
        }


def evaluate_landmarks_dict(
    landmarks_map: Dict[str, Tuple[float, float]],
    img_width: float = 1000.0,
    img_height: float = 1000.0,
    config: Optional[PercentEngineConfig] = None,
) -> Dict[str, Any]:
    """Evaluates landmarks provided as dict id -> (x, y) coordinates."""
    # Convert points into Point objects, scaled by image dimensions if normalized (<= 1.0)
    pts = {}
    for lid, (raw_x, raw_y) in landmarks_map.items():
        x = float(raw_x)
        y = float(raw_y)
        if x <= 1.0 and y <= 1.0:
            x *= img_width
            y *= img_height
        pts[lid] = Point(x, y)

    required = [
        "C2_PI", "C2_IC", "C2_AI",
        "C3_PS", "C3_AS", "C3_PI", "C3_IC", "C3_AI",
        "C4_PS", "C4_AS", "C4_PI", "C4_IC", "C4_AI",
    ]
    for r in required:
        if r not in pts:
            raise ValueError(f"Missing required landmark: {r}")

    inp = CVMInput(
        c2=VertebraC2(pts["C2_PI"], pts["C2_IC"], pts["C2_AI"]),
        c3=VertebraC3C4(pts["C3_PI"], pts["C3_IC"], pts["C3_AI"], pts["C3_PS"], pts["C3_AS"]),
        c4=VertebraC3C4(pts["C4_PI"], pts["C4_IC"], pts["C4_AI"], pts["C4_PS"], pts["C4_AS"]),
    )
    engine = PercentCVMEngine(config)
    return engine.evaluate_sample(inp)
