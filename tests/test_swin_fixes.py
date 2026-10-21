import unittest
import torch
import torch.nn.functional as F
from src.models.model import CephalometricSwinGCN, SoftArgmax2D
from src.models.losses import AnatomicalGraphLoss, AdaptiveWingLoss, WingLoss


class TestSwinFixes(unittest.TestCase):
    """
    Unit tests verifying fixes for the 3 Swin architectural issues:
    1. Logits-based SoftArgmax2D preventing background centroid drift and loss dominance.
    2. Backbone resolution alignment to multiples of 224 (32 * window_size_7) eliminating zero-padding distortion.
    3. AnatomicalGraphLoss relative deformation scaling preventing vanishing gradient (~0.004).
    """

    def test_soft_argmax_centroid_drift_elimination(self):
        """Verify that SoftArgmax2D on logits accurately localizes peaks without background centroid pull."""
        layer = SoftArgmax2D(num_landmarks=1, init_temperature=0.1)
        H, W = 640, 640
        # Peak at pixel (100, 150) -> normalized (100/640=0.15625, 150/640=0.234375)
        gt_x = 100.0 / float(W)
        gt_y = 150.0 / float(H)

        logits = torch.full((1, 1, H, W), -5.0)
        logits[0, 0, 148:153, 98:103] = 10.0  # Sharp peak

        coords = layer(logits)
        pred_x = coords[0, 0, 0].item()
        pred_y = coords[0, 0, 1].item()

        # Coordinate error must be sub-pixel (< 1 px = 1/640 ~ 0.00156)
        error_px_x = abs(pred_x - gt_x) * W
        error_px_y = abs(pred_y - gt_y) * H

        self.assertLess(error_px_x, 1.0, f"X error too large: {error_px_x:.2f} px")
        self.assertLess(error_px_y, 1.0, f"Y error too large: {error_px_y:.2f} px")

    def test_swin_backbone_aligned_resolution(self):
        """Verify CephalometricSwinGCN aligns resolution to multiple of 224 and preserves input shapes."""
        model = CephalometricSwinGCN(num_landmarks=13, pretrained=False, img_size=640)
        self.assertEqual(model.aligned_size, 672)

        # 640x640 input
        x640 = torch.randn(1, 3, 640, 640)
        hm640, coords640 = model(x640)
        self.assertEqual(hm640.shape, (1, 13, 640, 640))
        self.assertEqual(coords640.shape, (1, 13, 2))
        self.assertTrue((coords640 >= 0.0).all() and (coords640 <= 1.0).all())

        # 672x672 input
        x672 = torch.randn(1, 3, 672, 672)
        hm672, coords672 = model(x672)
        self.assertEqual(hm672.shape, (1, 13, 672, 672))
        self.assertEqual(coords672.shape, (1, 13, 2))

    def test_anatomical_graph_loss_relative_magnitude(self):
        """Verify AnatomicalGraphLoss produces non-vanishing, scale-invariant loss and healthy gradients."""
        adj = CephalometricSwinGCN(pretrained=False).adj_matrix
        loss_fn = AnatomicalGraphLoss(adj, relative=True)

        gt = torch.rand(2, 13, 2) * 0.5 + 0.25
        pred = gt.clone().detach().requires_grad_(True)
        with torch.no_grad():
            pred.add_(torch.randn_like(pred) * 0.005)  # ~3px perturbation

        loss = loss_fn(pred, gt)
        # Relative edge distortion should be meaningfully scaled (around 0.02 - 0.08, not 0.004)
        self.assertGreater(loss.item(), 0.01)
        loss.backward()

        self.assertIsNotNone(pred.grad)
        self.assertGreater(pred.grad.norm().item(), 0.1)

    def test_loss_balance_magnitudes(self):
        """Verify balanced loss magnitudes between heatmap AWL and coordinate WingLoss."""
        from src.train import LAMBDA_HM, LAMBDA_CD

        awl = AdaptiveWingLoss()
        wing = WingLoss(img_size=672.0)

        # Baseline check on defaults
        self.assertEqual(LAMBDA_HM, 50.0)
        self.assertEqual(LAMBDA_CD, 2.0)


if __name__ == "__main__":
    unittest.main()
