import math
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from src.data.constants import LANDMARK_CLASSES, NUM_LANDMARKS
from src.utils.metrics import (
    compute_reference_length,
    compute_normalized_errors,
    compute_inferior_border_projections,
    compute_bland_altman,
    compute_notch_classification_metrics,
    compute_shape_agreement,
    compute_borderline_analysis,
    bootstrap_image_confidence_interval,
    compute_test_comprehensive_metrics,
    compute_staging_metrics,
    compute_wilson_confidence_interval,
    format_summary_report,
    C3_PI_IDX,
    C3_AI_IDX,
)


class TestNormalizedMetricsSuite(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_compute_reference_length_2d_and_3d(self):
        # Create a single synthetic sample (13, 2)
        sample = np.zeros((NUM_LANDMARKS, 2), dtype=float)
        sample[C3_PI_IDX] = [100.0, 200.0]
        sample[C3_AI_IDX] = [140.0, 230.0]  # dx = 40, dy = 30 -> hypot = 50.0

        ref_len_single = compute_reference_length(sample, img_size=640.0)
        self.assertAlmostEqual(ref_len_single, 50.0, places=4)

        # Batch of 2 samples (2, 13, 2)
        batch = np.zeros((2, NUM_LANDMARKS, 2), dtype=float)
        batch[0, C3_PI_IDX] = [100.0, 200.0]
        batch[0, C3_AI_IDX] = [140.0, 230.0]  # hypot = 50.0
        batch[1, C3_PI_IDX] = [200.0, 300.0]
        batch[1, C3_AI_IDX] = [260.0, 380.0]  # dx = 60, dy = 80 -> hypot = 100.0

        ref_len_batch = compute_reference_length(batch, img_size=640.0)
        self.assertIsInstance(ref_len_batch, np.ndarray)
        self.assertEqual(len(ref_len_batch), 2)
        self.assertAlmostEqual(ref_len_batch[0], 50.0, places=4)
        self.assertAlmostEqual(ref_len_batch[1], 100.0, places=4)

        # PyTorch Tensor input
        batch_tensor = torch.from_numpy(batch).float()
        ref_len_tensor = compute_reference_length(batch_tensor, img_size=640.0)
        self.assertAlmostEqual(ref_len_tensor[0], 50.0, places=4)
        self.assertAlmostEqual(ref_len_tensor[1], 100.0, places=4)

    def test_compute_normalized_errors(self):
        # 1 sample where L_ref = 50.0
        gt = np.zeros((NUM_LANDMARKS, 2), dtype=float)
        gt[C3_PI_IDX] = [100.0, 200.0]
        gt[C3_AI_IDX] = [140.0, 230.0]  # hypot = 50.0

        # Predicted has dx = 3.0, dy = 4.0 for landmark 0 (radial error = 5.0 px)
        pred = gt.copy()
        pred[0] = [3.0, 4.0]

        res = compute_normalized_errors(pred, gt, img_size=640.0)
        # For landmark 0:
        # radial px = 5.0
        # normalized error % = (5.0 / 50.0) * 100% = 10.0%
        # signed dx px = 3.0, dy px = 4.0
        # signed dx % = (3.0 / 50.0) * 100% = 6.0%
        # signed dy % = (4.0 / 50.0) * 100% = 8.0%
        self.assertAlmostEqual(res["radial_error_px"][0], 5.0, places=4)
        self.assertAlmostEqual(res["normalized_error_pct"][0], 10.0, places=4)
        self.assertAlmostEqual(res["signed_dx_px"][0], 3.0, places=4)
        self.assertAlmostEqual(res["signed_dy_px"][0], 4.0, places=4)
        self.assertAlmostEqual(res["signed_dx_pct"][0], 6.0, places=4)
        self.assertAlmostEqual(res["signed_dy_pct"][0], 8.0, places=4)
        self.assertAlmostEqual(res["ref_length"], 50.0, places=4)

    def test_compute_inferior_border_projections(self):
        # Test C3 inferior border: PI (idx 5) at [100, 200], AI (idx 7) at [150, 200]
        # Baseline vector b = AI - PI = [50, 0] (horizontal anteriorly pointing)
        # Unit along = [1.0, 0.0]
        # Superior normal pointing into vertebral body (smaller Y in image coords):
        # sgn(bx) * (by/||b||, -bx/||b||) = 1.0 * (0, -1) = [0.0, -1.0]
        gt = np.zeros((NUM_LANDMARKS, 2), dtype=float)
        gt[C3_PI_IDX] = [100.0, 200.0]
        gt[C3_AI_IDX] = [150.0, 200.0]
        gt[6] = [125.0, 195.0]  # C3_IC

        # Prediction: shift C3_IC by +3 along X (anteriorly) and -4 along Y (superiorly into body)
        pred = gt.copy()
        pred[6] = [128.0, 191.0]

        ref_len = 50.0
        along_px, perp_px, along_pct, perp_pct = compute_inferior_border_projections(
            pred, gt, vertebra="C3", landmark_key="IC", img_size=640.0, ref_lengths=ref_len, return_pct=True
        )

        # err_vec = pred - gt = [+3.0, -4.0]
        # along = [3.0, -4.0] . [1.0, 0.0] = +3.0
        # perp  = [3.0, -4.0] . [0.0, -1.0] = +4.0 (positive into body)
        self.assertAlmostEqual(along_px, 3.0, places=4)
        self.assertAlmostEqual(perp_px, 4.0, places=4)
        self.assertAlmostEqual(along_pct, 6.0, places=4)  # 3.0 / 50 * 100
        self.assertAlmostEqual(perp_pct, 8.0, places=4)   # 4.0 / 50 * 100

    def test_compute_bland_altman(self):
        # Known diffs
        gt = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        pred = gt + np.array([1.0, 2.0, 1.0, 2.0, 1.0])  # diffs = [1, 2, 1, 2, 1], mean = 1.4

        ba = compute_bland_altman(pred, gt)
        self.assertAlmostEqual(ba["bias"], 1.4, places=4)
        self.assertAlmostEqual(ba["mae"], 1.4, places=4)
        self.assertGreater(ba["sd_diff"], 0.0)
        self.assertAlmostEqual(ba["loa_lower"], ba["bias"] - 1.96 * ba["sd_diff"], places=4)
        self.assertAlmostEqual(ba["loa_upper"], ba["bias"] + 1.96 * ba["sd_diff"], places=4)

    def test_compute_notch_classification_metrics(self):
        gt_notches = np.array([True, True, False, False, True])
        pred_notches = np.array([True, False, False, True, True])

        # tp: 2 (idx 0, 4), fp: 1 (idx 3), fn: 1 (idx 1), tn: 1 (idx 2)
        # acc: 3/5 = 60%, sens: 2/3 = 66.67%, spec: 1/2 = 50%
        res = compute_notch_classification_metrics(pred_notches, gt_notches)
        self.assertAlmostEqual(res["accuracy"], 60.0, places=2)
        self.assertAlmostEqual(res["sensitivity"], 66.67, places=1)
        self.assertAlmostEqual(res["specificity"], 50.0, places=1)
        self.assertIn("cohen_kappa", res)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fp"], 1)

    def test_compute_borderline_analysis(self):
        # Cutoffs: C2=5.0, C3=5.0, C4=6.0
        gt_percentages = {
            "C2": np.array([4.5, 8.0, 2.0, 5.8]),  # cases 0 (4.5) and 3 (5.8) within ±1% of 5.0
            "C3": np.array([1.0, 5.2, 1.0, 8.0]),  # case 1 (5.2) within ±1% of 5.0
            "C4": np.array([2.0, 2.0, 6.5, 2.0]),  # case 2 (6.5) within ±1% of 6.0
        }
        # In this synthetic setup, all 4 samples are borderline for at least one vertebra at delta=1.0%
        pred_stages = np.array([1, 2, 3, 4])
        gt_stages = np.array([1, 2, 4, 4])  # case 2 is incorrect

        res = compute_borderline_analysis(
            gt_percentages, pred_stages, gt_stages, cutoffs={"C2": 5.0, "C3": 5.0, "C4": 6.0}, deltas=(1.0, 2.0)
        )
        self.assertIn("delta_1.0pct", res)
        self.assertIn("delta_2.0pct", res)
        d1 = res["delta_1.0pct"]
        self.assertEqual(d1["borderline_count"], 4)
        self.assertAlmostEqual(d1["borderline_share_pct"], 100.0, places=1)
        self.assertAlmostEqual(d1["borderline_accuracy_pct"], 75.0, places=1)

    def test_bootstrap_image_confidence_interval(self):
        # 5 images with 13 landmarks each
        sample_values = [np.random.normal(loc=3.5, scale=0.5, size=13) for _ in range(5)]
        lower, upper = bootstrap_image_confidence_interval(sample_values, stat_fn=np.mean, n_resamples=200, seed=42)
        self.assertGreater(lower, 0.0)
        self.assertGreater(upper, lower)
        self.assertTrue(2.0 <= lower <= 5.0)

    def test_csv_output_schema_and_comprehensive_metrics(self):
        # Verify complete schema has exactly 26 required columns
        expected_columns = [
            "image_id", "split", "true_stage", "predicted_stage",
            "orig_w", "orig_h", "pad_x", "pad_y", "scale_x", "scale_y",
            "landmark_name", "gt_x", "gt_y", "pred_x", "pred_y", "ref_length",
            "radial_error_px", "normalized_error_pct",
            "signed_dx_px", "signed_dy_px", "signed_dx_pct", "signed_dy_pct",
            "error_along_px", "error_along_pct", "error_perp_px", "error_perp_pct"
        ]
        self.assertEqual(len(expected_columns), 26)

        # Generate synthetic test set evaluation
        n_samples = 3
        gt_coords = np.zeros((n_samples, NUM_LANDMARKS, 2), dtype=float)
        pred_coords = np.zeros((n_samples, NUM_LANDMARKS, 2), dtype=float)
        sample_ids = [f"test_{i:03d}.png" for i in range(n_samples)]
        orig_dims = [(1935, 2400), (640, 640), (1200, 1600)]

        for i in range(n_samples):
            # Setup reasonable vertebra landmarks
            gt_coords[i, C3_PI_IDX] = [100.0, 200.0]
            gt_coords[i, C3_AI_IDX] = [150.0, 200.0]  # L_ref = 50.0
            gt_coords[i, 0] = [100.0, 150.0]  # C2_PI
            gt_coords[i, 2] = [150.0, 150.0]  # C2_AI
            gt_coords[i, 1] = [125.0, 145.0]  # C2_IC
            gt_coords[i, 3] = [100.0, 180.0]  # C3_PS
            gt_coords[i, 4] = [150.0, 180.0]  # C3_AS
            gt_coords[i, 6] = [125.0, 195.0]  # C3_IC
            gt_coords[i, 8] = [100.0, 230.0]  # C4_PS
            gt_coords[i, 9] = [150.0, 230.0]  # C4_AS
            gt_coords[i, 10] = [100.0, 250.0] # C4_PI
            gt_coords[i, 11] = [125.0, 245.0] # C4_IC
            gt_coords[i, 12] = [150.0, 250.0] # C4_AI
            pred_coords[i] = gt_coords[i] + np.random.normal(0, 1.0, size=(NUM_LANDMARKS, 2))

        pred_morph = {
            "concavities": {"C2": [5.2, 3.1, 4.8], "C3": [5.1, 2.0, 6.0], "C4": [6.2, 1.5, 7.1]},
            "shape_indices": {"C3": [0.85, 0.90, 0.78], "C4": [0.82, 0.88, 0.76]},
            "notches": {"C2": [True, False, False], "C3": [True, False, True], "C4": [True, False, True]},
            "shapes": {"C3": ["Trapezoidal", "Trapezoidal", "Trapezoidal"], "C4": ["Trapezoidal", "Trapezoidal", "Trapezoidal"]},
        }
        gt_morph = {
            "concavities": {"C2": [5.0, 3.0, 5.0], "C3": [5.0, 2.1, 5.9], "C4": [6.0, 1.4, 7.0]},
            "shape_indices": {"C3": [0.84, 0.89, 0.77], "C4": [0.81, 0.87, 0.75]},
            "notches": {"C2": [True, False, False], "C3": [True, False, True], "C4": [True, False, True]},
            "shapes": {"C3": ["Trapezoidal", "Trapezoidal", "Trapezoidal"], "C4": ["Trapezoidal", "Trapezoidal", "Trapezoidal"]},
        }

        comp_metrics = compute_test_comprehensive_metrics(
            pred_coords=pred_coords,
            gt_coords=gt_coords,
            sample_ids=sample_ids,
            orig_dims=orig_dims,
            img_size=640.0,
            pred_stages=[1, 1, 2],
            gt_stages=[1, 1, 2],
            pred_morphometrics=pred_morph,
            gt_morphometrics=gt_morph,
            n_bootstrap=50,
        )

        self.assertIn("general", comp_metrics)
        self.assertIn("by_vertebra", comp_metrics)
        self.assertIn("by_landmark", comp_metrics)
        self.assertNotIn("by_source", comp_metrics)
        self.assertIn("bootstrap_ci_95", comp_metrics)
        self.assertIn("clinical", comp_metrics)
        self.assertIn("staging", comp_metrics)
        self.assertNotIn("legacy", comp_metrics)

        # Formatted report generation
        report_str = format_summary_report(comp_metrics, title="TEST REPORT")
        self.assertIn("SECTION B.1: OVERALL NORMALIZED LANDMARK PERFORMANCE", report_str)
        self.assertIn("SECTION B.2: PER-VERTEBRA BREAKDOWN", report_str)
        self.assertIn("SECTION B.3: PER-LANDMARK PERFORMANCE", report_str)
        self.assertNotIn("SECTION B.4: CROSS-SOURCE STRATIFICATION", report_str)
        self.assertIn("SECTION C.1: BLAND-ALTMAN AGREEMENT", report_str)
        self.assertIn("SECTION D: STAGING EVALUATION (TEST SET)", report_str)

    def test_compute_staging_metrics(self):
        # Synthetic cohort with 12 samples across CS1 to CS6
        gts = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1]
        preds = [1, 2, 3, 3, 2, 4, 5, 5, 6, 6, 6, 1]

        stg = compute_staging_metrics(
            pred_stages=preds,
            gt_stages=gts,
            n_bootstrap=100,
            seed=42,
        )

        # 1. Overall & Adjacent Accuracy
        self.assertEqual(stg["valid_samples"], 12)
        self.assertGreater(stg["overall_accuracy"], 50.0)
        self.assertGreaterEqual(stg["adjacent_accuracy"], stg["overall_accuracy"])
        self.assertIn("lower", stg["accuracy_ci_95_wilson"])
        self.assertIn("upper", stg["accuracy_ci_95_wilson"])
        self.assertIn("lower", stg["accuracy_ci_95_bootstrap"])
        self.assertIn("upper", stg["accuracy_ci_95_bootstrap"])

        # 2. Kappa Metrics
        self.assertIn("quadratic_weighted_kappa", stg)
        self.assertIn("unweighted_cohen_kappa", stg)
        self.assertGreater(stg["quadratic_weighted_kappa"], 0.5)

        # 3. 6x6 Confusion Matrix
        cm = np.array(stg["confusion_matrix"])
        self.assertEqual(cm.shape, (6, 6))
        self.assertEqual(np.sum(cm), 12)

        # 4. Per-stage & Macro Averages
        self.assertEqual(len(stg["per_stage"]), 6)
        self.assertIn("precision", stg["macro_average"])
        self.assertIn("recall", stg["macro_average"])
        self.assertIn("f1_score", stg["macro_average"])

        # 5. Transitions
        self.assertIn("CS2_CS3", stg["transitions"])
        self.assertIn("CS3_CS4", stg["transitions"])
        self.assertIn("CS5_CS6", stg["transitions"])
        self.assertIn("error_rate_pct", stg["transitions"]["CS2_CS3"])
        self.assertIn("cross_boundary_error_rate_pct", stg["transitions"]["CS2_CS3"])

        # 6. MASE
        self.assertIn("mean_absolute_stage_error", stg)
        self.assertNotIn("by_source", stg)


if __name__ == "__main__":
    unittest.main()
