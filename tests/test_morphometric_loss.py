import unittest
import torch
import torch.nn as nn
import numpy as np

from src.models.losses import MorphometricShapeLoss
from src.utils.cvm_calculator import PercentCVMEngine, CVMInput, VertebraC2, VertebraC3C4, Point


class TestMorphometricShapeLoss(unittest.TestCase):
    """
    Unit tests for MorphometricShapeLoss, verifying mathematical correctness against
    the canonical implementation in cvm-backend/app/static/calculator.js, numerical stability,
    gradient flow, and masking of missing landmarks.
    """

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.loss_fn = MorphometricShapeLoss(weight_si=1.0, weight_taper=1.0, beta=0.05)

    def test_exact_match_with_calculator_js(self):
        """
        Verify that compute_vertebra_metrics exactly matches the anatomical midline axis
        algorithm from cvm-backend/app/static/calculator.js (classifyC3C4Shape).
        """
        # Load sample coordinates from test dataset
        coords_data = np.load("dataset/test/labels/0111.npz")["coords"]
        coords_tensor = torch.tensor(coords_data, dtype=torch.float32).unsqueeze(0)

        # C3 indices: ps=3, as=4, pi=5, ai=7
        c3_ps = coords_tensor[:, 3, :]
        c3_as = coords_tensor[:, 4, :]
        c3_pi = coords_tensor[:, 5, :]
        c3_ai = coords_tensor[:, 7, :]

        # C4 indices: ps=8, as=9, pi=10, ai=12
        c4_ps = coords_tensor[:, 8, :]
        c4_as = coords_tensor[:, 9, :]
        c4_pi = coords_tensor[:, 10, :]
        c4_ai = coords_tensor[:, 12, :]

        pred_si_c3, pred_tr_c3 = MorphometricShapeLoss.compute_vertebra_metrics(c3_ps, c3_as, c3_pi, c3_ai)
        pred_si_c4, pred_tr_c4 = MorphometricShapeLoss.compute_vertebra_metrics(c4_ps, c4_as, c4_pi, c4_ai)

        # Ground truth values from Python PercentCVMEngine (identical to calculator.js)
        calc = PercentCVMEngine()
        inp = CVMInput(
            c2=VertebraC2(
                inferior_posterior=Point(*coords_data[0]),
                inferior_concavity=Point(*coords_data[1]),
                inferior_anterior=Point(*coords_data[2]),
            ),
            c3=VertebraC3C4(
                inferior_posterior=Point(*coords_data[5]),
                inferior_concavity=Point(*coords_data[6]),
                inferior_anterior=Point(*coords_data[7]),
                superior_posterior=Point(*coords_data[3]),
                superior_anterior=Point(*coords_data[4]),
            ),
            c4=VertebraC3C4(
                inferior_posterior=Point(*coords_data[10]),
                inferior_concavity=Point(*coords_data[11]),
                inferior_anterior=Point(*coords_data[12]),
                superior_posterior=Point(*coords_data[8]),
                superior_anterior=Point(*coords_data[9]),
            ),
        )
        res = calc.evaluate_sample(inp)

        expected_si_c3 = res["shapes"]["C3"]["shape_index"]
        expected_tr_c3 = res["shapes"]["C3"]["taper_ratio"]
        expected_si_c4 = res["shapes"]["C4"]["shape_index"]
        expected_tr_c4 = res["shapes"]["C4"]["taper_ratio"]

        self.assertAlmostEqual(pred_si_c3.item(), expected_si_c3, places=3)
        self.assertAlmostEqual(pred_tr_c3.item(), expected_tr_c3, places=3)
        self.assertAlmostEqual(pred_si_c4.item(), expected_si_c4, places=3)
        self.assertAlmostEqual(pred_tr_c4.item(), expected_tr_c4, places=3)

    def test_zero_loss_for_identical_inputs(self):
        """Verify loss is virtually zero when predictions equal ground truth."""
        coords_data = np.load("dataset/test/labels/0111.npz")["coords"]
        coords_tensor = torch.tensor(coords_data, dtype=torch.float32).unsqueeze(0)

        loss = self.loss_fn(coords_tensor, coords_tensor)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_gradient_flow_and_no_nans(self):
        """Verify gradients flow backwards smoothly without any NaNs or Infs."""
        coords_data = np.load("dataset/test/labels/0111.npz")["coords"]
        gt_tensor = torch.tensor(coords_data, dtype=torch.float32).unsqueeze(0)

        # Perturb prediction coordinates slightly
        pred_tensor = gt_tensor.clone().requires_grad_(True)
        with torch.no_grad():
            pred_tensor[:, 10:, 1] += 0.02  # downward perturbation on C4

        loss = self.loss_fn(pred_tensor, gt_tensor)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(pred_tensor.grad)
        self.assertFalse(torch.isnan(pred_tensor.grad).any())
        self.assertFalse(torch.isinf(pred_tensor.grad).any())
        self.assertGreater(pred_tensor.grad.norm().item(), 0.0)

    def test_missing_landmark_masking(self):
        """Verify samples with missing/unannotated landmarks (-1.0) are safely masked out."""
        coords_data = np.load("dataset/test/labels/0111.npz")["coords"]
        gt_tensor = torch.tensor(coords_data, dtype=torch.float32).repeat(2, 1, 1)

        # Mark C4 landmarks as missing in the second sample
        gt_tensor[1, 8:, :] = -1.0

        pred_tensor = gt_tensor.clone().clamp(min=0.0).requires_grad_(True)

        loss = self.loss_fn(pred_tensor, gt_tensor)
        self.assertFalse(torch.isnan(loss))
        loss.backward()
        self.assertFalse(torch.isnan(pred_tensor.grad).any())

    def test_all_missing_returns_zero(self):
        """Verify all missing landmarks cleanly return zero without zero-division error."""
        gt_tensor = torch.full((2, 13, 2), -1.0)
        pred_tensor = torch.rand((2, 13, 2), requires_grad=True)

        loss = self.loss_fn(pred_tensor, gt_tensor)
        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
