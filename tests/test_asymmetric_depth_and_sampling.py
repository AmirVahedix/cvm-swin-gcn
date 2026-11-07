import os
import tempfile
import unittest
import numpy as np
import torch
from src.models.losses import AsymmetricEndplateConcavityLoss
from src.data.loaders.dataloader import build_off_midline_weighted_sampler


class TestAsymmetricDepthAndSampling(unittest.TestCase):
    def test_truly_asymmetric_depth_loss(self):
        """
        Verify that AsymmetricEndplateConcavityLoss applies a true asymmetric penalty:
        under-estimating a concavity (flattening it out) is penalized with penalty_underestimate multiplier.
        """
        criterion = AsymmetricEndplateConcavityLoss(
            weight_tangential=1.0,
            weight_depth=1.0,
            beta=0.01,
            penalty_underestimate=3.0,
            depth_threshold=0.025,
        )

        # Baseline: PI=(0.2, 0.8), AI=(0.8, 0.8), length = 0.6
        gt = torch.full((1, 13, 2), -1.0)
        gt[0, 10] = torch.tensor([0.2, 0.8])  # C4_PI
        gt[0, 12] = torch.tensor([0.8, 0.8])  # C4_AI
        # Genuine concavity: IC at (0.4, 0.74), depth = 0.06
        gt[0, 11] = torch.tensor([0.4, 0.74])

        # Pred 1: under-estimation (depth = 0.02 -> error = -0.04)
        pred_under = gt.clone()
        pred_under[0, 11] = torch.tensor([0.4, 0.78])
        pred_under.requires_grad_(True)

        # Pred 2: over-estimation (depth = 0.10 -> error = +0.04)
        pred_over = gt.clone()
        pred_over[0, 11] = torch.tensor([0.4, 0.70])
        pred_over.requires_grad_(True)

        loss_under = criterion(pred_under, gt)
        loss_over = criterion(pred_over, gt)

        # Ratio must be approximately 3.0x
        ratio = loss_under.item() / loss_over.item()
        self.assertAlmostEqual(ratio, 3.0, delta=0.1)

        # Test gradients backpropagate smoothly
        loss_under.backward()
        self.assertIsNotNone(pred_under.grad)
        self.assertTrue((pred_under.grad[0, 11] != 0).any())

    def test_off_midline_weighted_sampler_logic(self):
        """
        Verify that build_off_midline_weighted_sampler checks all 3 vertebrae (C2, C3, C4)
        and boosts an image if ANY single vertebra has an off-midline concavity (|r - 0.50| >= threshold).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            filenames = ["img_c2_off.png", "img_c3_off.png", "img_c4_off.png", "img_all_centered.png"]

            def make_coords(c2_r=0.5, c3_r=0.5, c4_r=0.5):
                coords = np.zeros((13, 2), dtype=np.float32)
                # C2 (0, 1, 2)
                coords[0] = [0.2, 0.2]
                coords[2] = [0.8, 0.2]
                coords[1] = [0.2 + c2_r * 0.6, 0.18]
                # C3 (5, 6, 7)
                coords[5] = [0.2, 0.5]
                coords[7] = [0.8, 0.5]
                coords[6] = [0.2 + c3_r * 0.6, 0.48]
                # C4 (10, 11, 12)
                coords[10] = [0.2, 0.8]
                coords[12] = [0.8, 0.8]
                coords[11] = [0.2 + c4_r * 0.6, 0.78]
                return coords

            # Sample 0: C2 off-midline (r = 0.35, |r-0.5|=0.15 >= 0.10)
            np.savez(os.path.join(tmp_dir, "img_c2_off.npz"), coords=make_coords(c2_r=0.35, c3_r=0.5, c4_r=0.5))
            # Sample 1: C3 off-midline (r = 0.65, |r-0.5|=0.15 >= 0.10)
            np.savez(os.path.join(tmp_dir, "img_c3_off.npz"), coords=make_coords(c2_r=0.5, c3_r=0.65, c4_r=0.5))
            # Sample 2: C4 off-midline (r = 0.30, |r-0.5|=0.20 >= 0.10)
            np.savez(os.path.join(tmp_dir, "img_c4_off.npz"), coords=make_coords(c2_r=0.5, c3_r=0.5, c4_r=0.30))
            # Sample 3: All 3 vertebrae centered at exactly 50%
            np.savez(os.path.join(tmp_dir, "img_all_centered.npz"), coords=make_coords(c2_r=0.5, c3_r=0.5, c4_r=0.5))

            sampler = build_off_midline_weighted_sampler(
                train_files=filenames,
                labels_dir=tmp_dir,
                off_midline_threshold=0.10,
                boost_weight=3.0,
                base_weight=1.0,
                num_samples=4,
            )

            self.assertIsNotNone(sampler)
            weights = sampler.weights.tolist()
            # Samples 0, 1, 2 must have boost weight (3.0), Sample 3 must have base weight (1.0)
            self.assertEqual(weights[0], 3.0)
            self.assertEqual(weights[1], 3.0)
            self.assertEqual(weights[2], 3.0)
            self.assertEqual(weights[3], 1.0)


if __name__ == "__main__":
    unittest.main()
