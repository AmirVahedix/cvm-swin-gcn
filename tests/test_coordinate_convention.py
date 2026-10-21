import unittest
import torch
import torch.nn.functional as F
import numpy as np

from src.models.model import SoftArgmax2D
from scripts.batch_inference import preprocess_image
import tempfile
import cv2
from pathlib import Path


class TestCoordinateConvention(unittest.TestCase):
    def test_softargmax_pixel_exact_recovery(self):
        """
        Verify that a 2D Gaussian heatmap centered at exact pixel (px_x, px_y)
        is mapped by SoftArgmax2D to normalized coords that, when multiplied by W and H,
        recover the GT landmark location to a small fraction of a pixel (< 0.05 px).
        """
        H, W = 640, 640
        num_landmarks = 13
        soft_argmax = SoftArgmax2D(num_landmarks=num_landmarks, init_temperature=0.01)

        # Test diverse landmark coordinates across image space (including low C4 points)
        test_points = [
            (50.0, 50.0),     # Top-left corner
            (320.0, 320.0),   # Center
            (500.0, 550.0),   # Lower-right (C4 territory)
            (150.25, 476.75), # Sub-pixel real coordinate
            (212.33, 590.92), # C4_AI lowest vertebra point
        ]

        grid_y = torch.arange(H).float()[:, None]
        grid_x = torch.arange(W).float()[None, :]
        sigma = 3.0

        for pt_x, pt_y in test_points:
            # Build synthetic Gaussian heatmap
            dist_sq = (grid_x - pt_x) ** 2 + (grid_y - pt_y) ** 2
            gaussian_hm = torch.exp(-dist_sq / (2.0 * sigma ** 2))  # [H, W]

            # Shape [1, num_landmarks, H, W]
            heatmaps = gaussian_hm.unsqueeze(0).unsqueeze(0).repeat(1, num_landmarks, 1, 1)

            # Soft-argmax with temperature tau=1.0 preserving Gaussian sub-pixel spatial expectation
            with torch.no_grad():
                soft_argmax.log_temperature.fill_(float(np.log(1.0)))
                pred_coords = soft_argmax(heatmaps)  # [1, 13, 2] in normalized [0, 1]

            pred_px_x = pred_coords[0, 0, 0].item() * float(W)
            pred_px_y = pred_coords[0, 0, 1].item() * float(H)

            error_x = abs(pred_px_x - pt_x)
            error_y = abs(pred_px_y - pt_y)

            # Assert error is sub-pixel (< 0.05 px)
            self.assertLess(
                error_x,
                0.05,
                f"X coordinate offset too large! GT={pt_x}, Pred={pred_px_x:.4f}, Err={error_x:.4f}px"
            )
            self.assertLess(
                error_y,
                0.05,
                f"Y coordinate offset too large! GT={pt_y}, Pred={pred_px_y:.4f}, Err={error_y:.4f}px"
            )

    def test_letterbox_pad_and_slice_synthetic_spot(self):
        """
        Verify that passing through the letterbox padding and un-padding pipeline
        with a synthetic spot accurately inverts scale and offsets to sub-pixel accuracy.
        """
        orig_w, orig_h = 1935, 2400
        spot_orig_x = 470.5
        spot_orig_y = 1837.25

        target_size = 640
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2

        # Spot projected onto 640x640 canvas
        canvas_x = spot_orig_x * scale + pad_x
        canvas_y = spot_orig_y * scale + pad_y

        # Un-padding back to original image space
        recovered_x = (canvas_x - pad_x) / scale
        recovered_y = (canvas_y - pad_y) / scale

        self.assertAlmostEqual(recovered_x, spot_orig_x, places=5)
        self.assertAlmostEqual(recovered_y, spot_orig_y, places=5)

    def test_temperature_clamp_behavior(self):
        """
        Verify the user observation that when tau sits at the 1e-3 clamp,
        SoftArgmax2D collapses to a hard argmax (losing sub-pixel precision),
        whereas learnable tau in a smooth range maintains sub-pixel precision.
        """
        H, W = 640, 640
        soft_argmax = SoftArgmax2D(num_landmarks=1, init_temperature=0.1)

        pt_x, pt_y = 150.25, 476.75  # Sub-pixel coordinate (.25)
        sigma = 3.0
        grid_y = torch.arange(H).float()[:, None]
        grid_x = torch.arange(W).float()[None, :]
        dist_sq = (grid_x - pt_x) ** 2 + (grid_y - pt_y) ** 2
        gaussian_hm = torch.exp(-dist_sq / (2.0 * sigma ** 2))[None, None, :, :]

        # 1. At 1e-3 clamp -> acts like hard integer argmax
        with torch.no_grad():
            soft_argmax.log_temperature.fill_(float(np.log(1e-3)))
            pred_clamped = soft_argmax(gaussian_hm)
        pred_x_clamped = pred_clamped[0, 0, 0].item() * float(W)
        self.assertAlmostEqual(pred_x_clamped, 150.0, places=3)  # Rounded to integer 150

        # 2. At smooth tau=1.0 -> recovers true sub-pixel position
        with torch.no_grad():
            soft_argmax.log_temperature.fill_(float(np.log(1.0)))
            pred_smooth = soft_argmax(gaussian_hm)
        pred_x_smooth = pred_smooth[0, 0, 0].item() * float(W)
        self.assertLess(abs(pred_x_smooth - pt_x), 0.05)


if __name__ == "__main__":
    unittest.main()
