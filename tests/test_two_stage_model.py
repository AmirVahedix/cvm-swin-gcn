import unittest
import torch
import numpy as np

from src.models.two_stage import TwoStageCephalometricSwinGCN


class TestTwoStageModel(unittest.TestCase):
    def setUp(self):
        # Lightweight dummy models without downloading full pretrained weights for fast testing
        self.device = torch.device("cpu")
        self.model = TwoStageCephalometricSwinGCN(
            num_landmarks=13,
            stage1_img_size=640,
            stage2_img_size=448,
            margin_ratio=0.18,
            pretrained_stage1=False,
            pretrained_stage2=False,
        ).to(self.device)

    def test_freeze_stage1(self):
        """Verifies that freeze_stage1 successfully sets requires_grad=False on all Stage 1 parameters."""
        self.model.freeze_stage1()
        for name, param in self.model.stage1.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"Parameter {name} in Stage 1 still requires grad after freeze_stage1()"
            )

        # Stage 2 parameters should remain trainable
        trainable_s2 = [p.requires_grad for p in self.model.stage2.parameters()]
        self.assertTrue(any(trainable_s2))

    def test_forward_coarse_and_fine_shapes(self):
        """Verifies shapes of independent forward passes."""
        B = 2
        dummy_full = torch.randn(B, 3, 640, 640, device=self.device)
        dummy_roi = torch.randn(B, 3, 448, 448, device=self.device)

        with torch.no_grad():
            s1_hm, s1_coords = self.model.forward_coarse(dummy_full)
            self.assertEqual(s1_hm.shape, (B, 13, 640, 640))
            self.assertEqual(s1_coords.shape, (B, 13, 2))

            s2_hm, s2_coords = self.model.forward_fine(dummy_roi)
            self.assertEqual(s2_hm.shape, (B, 13, 448, 448))
            self.assertEqual(s2_coords.shape, (B, 13, 2))

    def test_full_cascaded_forward(self):
        """Verifies full end-to-end cascaded forward pass with ROI cropping and coordinate re-projection."""
        B = 2
        # Initialize synthetic images with pixel values in [0, 1]
        dummy_full = torch.rand(B, 3, 640, 640, device=self.device)

        with torch.no_grad():
            out = self.model(dummy_full, return_intermediates=True)

            self.assertIn("final_coords", out)
            self.assertIn("stage1_coords", out)
            self.assertIn("stage2_coords_roi", out)
            self.assertIn("bboxes", out)
            self.assertIn("roi_batch", out)

            final_coords = out["final_coords"]
            self.assertEqual(final_coords.shape, (B, 13, 2))
            self.assertEqual(out["roi_batch"].shape, (B, 3, 448, 448))
            self.assertEqual(len(out["bboxes"]), B)

            # Coordinates should be strictly bounded in normalized canvas space [0, 1]
            self.assertTrue((final_coords >= 0.0).all())
            self.assertTrue((final_coords <= 1.0).all())

    def test_refine_2pass(self):
        """Verifies optional 2-pass iterative refinement."""
        B = 1
        dummy_full = torch.rand(B, 3, 640, 640, device=self.device)

        with torch.no_grad():
            pass1_coords = self.model(dummy_full, refine_2pass=False)
            pass2_coords = self.model(dummy_full, refine_2pass=True)

            self.assertEqual(pass1_coords.shape, (B, 13, 2))
            self.assertEqual(pass2_coords.shape, (B, 13, 2))


if __name__ == "__main__":
    unittest.main()
