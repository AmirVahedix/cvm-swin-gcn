import unittest
import numpy as np
import torch
import cv2

from src.utils.roi_utils import (
    ROIBoundingBox,
    apply_anatomical_height_prior,
    compute_roi_bbox,
    crop_roi_with_padding,
    transform_coords_to_roi,
    inverse_transform_coords,
    validate_stage1_box,
    native_to_canvas_coords,
    canvas_to_native_coords,
)


class TestROITransforms(unittest.TestCase):
    def setUp(self):
        # Realistic cervical vertebrae coordinates at 640x640 canvas (normalized [0, 1])
        # Centered roughly at cx=0.45, cy=0.55 with span ~ 160 px
        self.raw_coords_px = np.array([
            [260.0, 280.0],  # C2_PI
            [280.0, 290.0],  # C2_IC
            [300.0, 285.0],  # C2_AI
            [250.0, 320.0],  # C3_PS
            [310.0, 315.0],  # C3_AS
            [255.0, 360.0],  # C3_PI
            [285.0, 370.0],  # C3_IC
            [315.0, 365.0],  # C3_AI
            [250.0, 395.0],  # C4_PS
            [315.0, 390.0],  # C4_AS
            [255.0, 440.0],  # C4_PI
            [285.0, 450.0],  # C4_IC
            [320.0, 445.0],  # C4_AI
        ], dtype=np.float32)

        self.img_size = 640
        self.norm_coords = self.raw_coords_px / float(self.img_size)

    def test_algebraic_invertibility_numpy(self):
        """Tests that forward and inverse coordinate mappings are strictly invertible."""
        bbox = compute_roi_bbox(self.norm_coords, img_w=self.img_size, img_h=self.img_size, margin_ratio=0.18)
        self.assertFalse(bbox.is_fallback)
        self.assertEqual(bbox.u2 - bbox.u1, bbox.D)
        self.assertEqual(bbox.v2 - bbox.v1, bbox.D)

        for target_size in [None, 448, 672]:
            # Forward transform
            roi_coords = transform_coords_to_roi(self.norm_coords, bbox, target_size=target_size, is_normalized_input=True)
            # All valid points should be within [0, 1]
            self.assertTrue((roi_coords >= 0.0).all())
            self.assertTrue((roi_coords <= 1.0).all())

            # Inverse transform
            reconstructed = inverse_transform_coords(roi_coords, bbox, target_size=target_size, output_normalized=True)
            max_diff = np.max(np.abs(self.norm_coords - reconstructed))
            self.assertLess(max_diff, 1e-6, f"Inversion error {max_diff} exceeds tolerance 1e-6 for target_size={target_size}")

    def test_algebraic_invertibility_torch(self):
        """Tests invertibility using PyTorch tensors on GPU / CPU."""
        device = torch.device("cpu")
        coords_t = torch.tensor(self.norm_coords, device=device)
        bbox = compute_roi_bbox(coords_t, img_w=self.img_size, img_h=self.img_size, margin_ratio=0.20)

        for target_size in [None, 448, 672]:
            roi_t = transform_coords_to_roi(coords_t, bbox, target_size=target_size, is_normalized_input=True)
            rec_t = inverse_transform_coords(roi_t, bbox, target_size=target_size, output_normalized=True)

            diff = torch.max(torch.abs(coords_t - rec_t)).item()
            self.assertLess(diff, 1e-6, f"Torch diff {diff} exceeds tolerance for target_size={target_size}")

    def test_invalid_missing_landmarks_preserved(self):
        """Tests that missing landmarks (marked as [-1, -1]) remain [-1, -1] in ROI and reconstructed space."""
        coords_with_missing = self.norm_coords.copy()
        coords_with_missing[1] = [-1.0, -1.0]  # Missing landmark 1
        coords_with_missing[5] = [-1.0, -1.0]  # Missing landmark 5

        bbox = compute_roi_bbox(coords_with_missing, img_w=self.img_size, img_h=self.img_size, margin_ratio=0.18)
        roi_coords = transform_coords_to_roi(coords_with_missing, bbox, target_size=448, is_normalized_input=True)

        self.assertEqual(roi_coords[1, 0], -1.0)
        self.assertEqual(roi_coords[1, 1], -1.0)
        self.assertEqual(roi_coords[5, 0], -1.0)
        self.assertEqual(roi_coords[5, 1], -1.0)

        rec = inverse_transform_coords(roi_coords, bbox, target_size=448, output_normalized=True)
        self.assertEqual(rec[1, 0], -1.0)
        self.assertEqual(rec[1, 1], -1.0)
        self.assertEqual(rec[5, 0], -1.0)
        self.assertEqual(rec[5, 1], -1.0)

    def test_boundary_padding_produces_exact_square(self):
        """Tests that near-boundary landmarks trigger padding and produce exactly a square target."""
        # Shift landmarks close to left boundary, keeping vertical coordinate inside anatomical bounds
        edge_coords = self.norm_coords.copy()
        edge_coords[:, 0] = np.clip(edge_coords[:, 0] - 0.35, 0.02, 0.25)  # Left edge

        bbox = compute_roi_bbox(edge_coords, img_w=self.img_size, img_h=self.img_size, margin_ratio=0.35, min_size=250)
        self.assertTrue(bbox.u1 < 0 or bbox.v1 < 0, f"Expected boundary overlap, got u1={bbox.u1}, v1={bbox.v1}")

        # Crop from a synthetic image
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 128
        target_size = 448
        cropped_patch, meta = crop_roi_with_padding(img, bbox, target_size=target_size)

        self.assertEqual(cropped_patch.shape, (target_size, target_size, 3),
                         f"Cropped patch shape {cropped_patch.shape} must be exactly ({target_size}, {target_size}, 3)")

    def test_stage1_validation_and_fallback(self):
        """Tests sanity checks and anchor fallback on corrupted or outlier Stage 1 inputs."""
        # 1. Normal coords -> Valid
        is_val, reason = validate_stage1_box(self.norm_coords, self.img_size, self.img_size)
        self.assertTrue(is_val)
        self.assertEqual(reason, "")

        # 2. NaN coords -> Invalid
        nan_coords = self.norm_coords.copy()
        nan_coords[0, 0] = np.nan
        is_val, reason = validate_stage1_box(nan_coords, self.img_size, self.img_size)
        self.assertFalse(is_val)
        self.assertIn("NaN", reason)

        # 3. Span too small (collapsed box)
        collapsed = np.full_like(self.norm_coords, 0.5)
        is_val, reason = validate_stage1_box(collapsed, self.img_size, self.img_size)
        self.assertFalse(is_val)
        self.assertIn("span too small", reason)

        # 4. Fallback execution
        fallback_bbox = compute_roi_bbox(nan_coords, self.img_size, self.img_size)
        self.assertTrue(fallback_bbox.is_fallback)
        self.assertEqual(fallback_bbox.D, fallback_bbox.u2 - fallback_bbox.u1)

    def test_real_image_path_impulse_centroid(self):
        """
        Crucial end-to-end verification:
        Places an impulse (sharp Gaussian spot) at a known continuous pixel coordinate (u*, v*),
        passes it through the actual crop + resize pipeline to 448x448, detects the centroid in the crop,
        and verifies it matches the theoretical forward coordinate mapping with exact sub-pixel alignment.
        """
        canvas_size = 640
        target_size = 448
        img = np.zeros((canvas_size, canvas_size), dtype=np.float32)

        # Spot location near center of vertebrae cluster
        u_star = 285.4
        v_star = 368.7
        sigma_spot = 2.0

        # Draw smooth Gaussian spot
        radius = int(5 * sigma_spot)
        u0 = max(0, int(np.floor(u_star - radius)))
        u1 = min(canvas_size, int(np.ceil(u_star + radius + 1)))
        v0 = max(0, int(np.floor(v_star - radius)))
        v1 = min(canvas_size, int(np.ceil(v_star + radius + 1)))

        gy, gx = np.meshgrid(np.arange(v0, v1), np.arange(u0, u1), indexing="ij")
        spot = np.exp(-((gx - u_star) ** 2 + (gy - v_star) ** 2) / (2.0 * sigma_spot ** 2))
        img[v0:v1, u0:u1] = spot

        # Compute ROI bounding box enclosing the landmark cluster
        bbox = compute_roi_bbox(self.raw_coords_px, img_w=canvas_size, img_h=canvas_size, margin_ratio=0.18)

        # Crop and resize via actual image pipeline
        cropped_img, meta = crop_roi_with_padding(
            image=img,
            bbox=bbox,
            target_size=target_size,
            interpolation=cv2.INTER_LINEAR,
        )

        # Find sub-pixel centroid of the spot inside cropped image
        # Using spatial expectation identical to SoftArgmax2D
        mass = cropped_img.sum()
        self.assertGreater(mass, 0.1, "Impulse spot disappeared during crop/resize")

        grid_y, grid_x = np.meshgrid(
            np.arange(target_size, dtype=np.float32) / float(target_size),
            np.arange(target_size, dtype=np.float32) / float(target_size),
            indexing="ij",
        )
        detected_norm_x = float((cropped_img * grid_x).sum() / mass)
        detected_norm_y = float((cropped_img * grid_y).sum() / mass)

        # Theoretical prediction via forward transform WITH target_size half-pixel alignment
        spot_coord = np.array([[u_star, v_star]])
        expected_roi_norm = transform_coords_to_roi(
            spot_coord, bbox, target_size=target_size, is_normalized_input=False
        )[0]

        # Convert normalized discrepancy to crop pixel units
        diff_px_x = abs(detected_norm_x - expected_roi_norm[0]) * target_size
        diff_px_y = abs(detected_norm_y - expected_roi_norm[1]) * target_size

        # Sub-pixel discrepancy should now be virtually zero (< 0.05 pixels)
        self.assertLess(diff_px_x, 0.05,
                        f"X centroid error {diff_px_x:.4f} px exceeds 0.05 px tolerance")
        self.assertLess(diff_px_y, 0.05,
                        f"Y centroid error {diff_px_y:.4f} px exceeds 0.05 px tolerance")

    def test_native_letterbox_roundtrip(self):
        """Tests roundtrip between native image resolution and letterbox canvas."""
        orig_w = 1935
        orig_h = 2400
        canvas_size = 640

        native_pts = np.array([
            [600.0, 1100.0],
            [750.0, 1250.0],
            [900.0, 1400.0],
            [-1.0, -1.0],  # missing
        ], dtype=np.float32)

        canvas_pts = native_to_canvas_coords(native_pts, orig_w=orig_w, orig_h=orig_h, canvas_size=canvas_size)
        reconstructed_native = canvas_to_native_coords(canvas_pts, orig_w=orig_w, orig_h=orig_h, canvas_size=canvas_size)

        valid_mask = native_pts[:, 0] >= 0
        max_err = np.max(np.abs(native_pts[valid_mask] - reconstructed_native[valid_mask]))
        self.assertLess(max_err, 0.5, f"Native roundtrip error {max_err:.3f} px exceeds 0.5 px tolerance")
        self.assertEqual(reconstructed_native[3, 0], -1.0)

    def test_anatomical_height_prior_healthy(self):
        """Verify that physiologically healthy coordinates are NOT altered by the prior."""
        clean_coords, applied = apply_anatomical_height_prior(
            self.norm_coords, img_w=self.img_size, img_h=self.img_size
        )
        self.assertFalse(applied)
        np.testing.assert_allclose(clean_coords, self.norm_coords, atol=1e-6)

    def test_anatomical_height_prior_collapsed_c4(self):
        """Verify that upward-collapsed C4 inferior landmarks are corrected downward."""
        collapsed = self.norm_coords.copy()
        # Collapse C4 inferior (10, 11, 12) upward right next to C4 superior (y ~ 395)
        collapsed[10:13, 1] = collapsed[8:10, 1].mean() + 0.005

        corrected, applied = apply_anatomical_height_prior(
            collapsed, img_w=self.img_size, img_h=self.img_size
        )
        self.assertTrue(applied)

        # C4 inferior must now sit below C4 superior by at least ~ 40 px
        c4_sup_y = corrected[8:10, 1].max() * self.img_size
        c4_inf_y = corrected[10:13, 1].min() * self.img_size
        self.assertGreater(c4_inf_y - c4_sup_y, 35.0)

        # compute_roi_bbox with prior must cover the lower floor
        bbox = compute_roi_bbox(collapsed, img_w=self.img_size, img_h=self.img_size, use_anatomical_prior=True)
        self.assertFalse(bbox.is_fallback)
        self.assertGreater(bbox.v2, c4_inf_y)


if __name__ == "__main__":
    unittest.main()
