import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.data.constants import LANDMARK_CLASSES, NUM_LANDMARKS
from src.train import (
    compute_per_landmark_metrics,
    log_best_per_landmark_metrics_to_mlflow,
    validate_epoch,
)


class TestLandmarkMetricsLogging(unittest.TestCase):

    def test_compute_per_landmark_metrics_accurate_values(self):
        """Verify metric computation accuracy against synthetic predictions with known offsets."""
        img_size = 640
        num_samples = 4
        num_landmarks = NUM_LANDMARKS

        # Base GT coordinates at center (0.5, 0.5)
        gt_coords = np.full((num_samples, num_landmarks, 2), 0.5, dtype=np.float32)

        # Pred coordinates identical to GT initially
        pred_coords = gt_coords.copy()

        # Landmark 0: exactly 2.0 pixels error along X
        # 2.0 px in normalized coords: 2.0 / 640
        pred_coords[:, 0, 0] += 2.0 / img_size

        # Landmark 1: exactly 4.0 pixels error along Y
        pred_coords[:, 1, 1] += 4.0 / img_size

        metrics_list = compute_per_landmark_metrics(pred_coords, gt_coords, img_size=img_size, pixel_spacing=0.1)

        self.assertEqual(len(metrics_list), num_landmarks)

        # Test Landmark 0 (2.0 px radial error = 0.2 mm)
        lm0 = metrics_list[0]
        self.assertEqual(lm0["id"], 0)
        self.assertEqual(lm0["name"], LANDMARK_CLASSES[0])
        self.assertEqual(lm0["count"], num_samples)
        self.assertAlmostEqual(lm0["mre_px"], 2.0, places=4)
        self.assertAlmostEqual(lm0["rmse_px"], 2.0, places=4)
        self.assertAlmostEqual(lm0["mre_mm"], 0.2, places=4)
        self.assertAlmostEqual(lm0["rmse_mm"], 0.2, places=4)
        self.assertEqual(lm0["sdr_2.0mm"], 100.0)
        self.assertEqual(lm0["sdr_2.5mm"], 100.0)
        self.assertEqual(lm0["sdr_2.0px"], 100.0)
        self.assertEqual(lm0["sdr_2.5px"], 100.0)
        self.assertEqual(lm0["sdr_3.0px"], 100.0)

        # Test Landmark 1 (4.0 px radial error = 0.4 mm)
        lm1 = metrics_list[1]
        self.assertEqual(lm1["id"], 1)
        self.assertEqual(lm1["name"], LANDMARK_CLASSES[1])
        self.assertEqual(lm1["count"], num_samples)
        self.assertAlmostEqual(lm1["mre_px"], 4.0, places=4)
        self.assertAlmostEqual(lm1["rmse_px"], 4.0, places=4)
        self.assertAlmostEqual(lm1["mre_mm"], 0.4, places=4)
        self.assertAlmostEqual(lm1["rmse_mm"], 0.4, places=4)
        self.assertEqual(lm1["sdr_2.0mm"], 100.0)  # 0.4 mm <= 2.0 mm
        self.assertEqual(lm1["sdr_2.5mm"], 100.0)
        self.assertEqual(lm1["sdr_2.0px"], 0.0)    # 4.0 px > 2.0 px
        self.assertEqual(lm1["sdr_2.5px"], 0.0)
        self.assertEqual(lm1["sdr_3.0px"], 0.0)
        self.assertEqual(lm1["sdr_4.0px"], 100.0)

        # Test Landmark 2 (0 px error = 0 mm)
        lm2 = metrics_list[2]
        self.assertAlmostEqual(lm2["mre_px"], 0.0, places=4)
        self.assertAlmostEqual(lm2["mre_mm"], 0.0, places=4)
        self.assertEqual(lm2["sdr_2.0mm"], 100.0)
        self.assertEqual(lm2["sdr_2.0px"], 100.0)

    def test_compute_per_landmark_metrics_torch_tensor_input(self):
        """Verify that PyTorch Tensor inputs are supported seamlessly."""
        pred = torch.zeros((2, NUM_LANDMARKS, 2), dtype=torch.float32)
        gt = torch.zeros((2, NUM_LANDMARKS, 2), dtype=torch.float32)

        metrics_list = compute_per_landmark_metrics(pred, gt, img_size=640)
        self.assertEqual(len(metrics_list), NUM_LANDMARKS)
        for lm in metrics_list:
            self.assertAlmostEqual(lm["mre_px"], 0.0, places=4)
            self.assertEqual(lm["sdr_2.5px"], 100.0)

    def test_compute_per_landmark_metrics_masked_landmarks(self):
        """Verify handling of missing/masked landmarks (coords marked as -1.0)."""
        gt_coords = np.full((3, NUM_LANDMARKS, 2), 0.5, dtype=np.float32)
        pred_coords = gt_coords.copy()

        # Mark landmark 3 as missing across all samples
        gt_coords[:, 3, :] = -1.0

        metrics_list = compute_per_landmark_metrics(pred_coords, gt_coords, img_size=640)
        lm3 = metrics_list[3]
        self.assertEqual(lm3["count"], 0)
        self.assertEqual(lm3["mre_px"], 0.0)
        self.assertEqual(lm3["sdr_2.5px"], 0.0)

    @patch("src.train.mlflow")
    def test_log_best_per_landmark_metrics_to_mlflow(self, mock_mlflow):
        """Verify that per-landmark JSON is saved locally and logged to MLflow."""
        mock_mlflow.active_run.return_value = MagicMock(info=MagicMock(run_id="best-run-123"))

        sample_landmarks = [
            {
                "id": i,
                "name": LANDMARK_CLASSES[i],
                "count": 10,
                "mae_px": 1.2,
                "rmse_px": 1.5,
                "mre_px": 1.2,
                "medre_px": 1.1,
                "sdre_px": 0.4,
                "min_error_px": 0.5,
                "max_error_px": 2.1,
                "sdr_2.0px": 90.0,
                "sdr_2.5px": 100.0,
                "sdr_3.0px": 100.0,
                "sdr_4.0px": 100.0,
            }
            for i in range(NUM_LANDMARKS)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            summary_metrics = {
                "rmse": 1.5,
                "sdr_2_0": 90.0,
                "sdr_3_0": 100.0,
                "sdr_4_0": 100.0,
            }

            payload = log_best_per_landmark_metrics_to_mlflow(
                landmark_metrics=sample_landmarks,
                epoch=5,
                metrics=summary_metrics,
                best_val_sdr=98.5,
                best_val_mae=1.15,
                best_val_loss=0.045,
                save_dir=save_dir,
            )

            # Check JSON file written to disk
            json_file = save_dir / "best_per_landmark_metrics.json"
            self.assertTrue(json_file.exists())

            with open(json_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            self.assertEqual(saved_data["epoch"], 5)
            self.assertEqual(saved_data["summary"]["best_val_sdr_2_5"], 98.5)
            self.assertEqual(saved_data["summary"]["best_val_mae"], 1.15)
            self.assertEqual(len(saved_data["landmarks"]), NUM_LANDMARKS)
            self.assertIn("C2_PI", saved_data["by_landmark_name"])

            # Check MLflow logging calls
            mock_mlflow.log_dict.assert_called_once()
            call_dict, call_artifact = mock_mlflow.log_dict.call_args[0]
            self.assertEqual(call_artifact, "best_per_landmark_metrics.json")
            self.assertEqual(call_dict["epoch"], 5)

            mock_mlflow.log_artifact.assert_called_once_with(
                str(json_file),
                artifact_path="checkpoints",
            )

    @patch("src.train.mlflow")
    def test_log_best_per_landmark_metrics_error_tolerance(self, mock_mlflow):
        """Verify that MLflow exceptions do not raise or break execution."""
        mock_mlflow.active_run.return_value = MagicMock(info=MagicMock(run_id="best-run-456"))
        mock_mlflow.log_dict.side_effect = RuntimeError("MLflow server unavailable")
        mock_mlflow.log_artifact.side_effect = RuntimeError("Artifact upload error")

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                payload = log_best_per_landmark_metrics_to_mlflow(
                    landmark_metrics=[],
                    epoch=1,
                    metrics={},
                    best_val_sdr=80.0,
                    best_val_mae=2.0,
                    best_val_loss=0.1,
                    save_dir=tmp_dir,
                )
            except Exception as e:
                self.fail(f"log_best_per_landmark_metrics_to_mlflow raised an exception: {e}")

            self.assertEqual(payload["epoch"], 1)
            self.assertTrue((Path(tmp_dir) / "best_per_landmark_metrics.json").exists())

    def test_validate_epoch_returns_per_landmark_metrics(self):
        """Verify that validate_epoch integrates compute_per_landmark_metrics into returned metrics dict."""
        num_samples = 2
        batch_size = 2
        img_size = 640

        # Mock model returning zeros for heatmap and center coords
        mock_model = MagicMock()
        mock_pred_hm = torch.zeros((batch_size, NUM_LANDMARKS, 80, 80))
        mock_pred_coords = torch.full((batch_size, NUM_LANDMARKS, 2), 0.5)
        mock_model.return_value = (mock_pred_hm, mock_pred_coords)

        # Mock dataloader yielding 1 batch
        batch = {
            "image": torch.zeros((batch_size, 3, img_size, img_size)),
            "heatmaps": torch.zeros((batch_size, NUM_LANDMARKS, 80, 80)),
            "coords": torch.full((batch_size, NUM_LANDMARKS, 2), 0.5),
        }
        mock_dataloader = [batch]

        # Dummy loss functions
        dummy_loss = MagicMock(return_value=torch.tensor(0.1))

        val_loss, metrics = validate_epoch(
            model=mock_model,
            dataloader=mock_dataloader,
            awl_loss=dummy_loss,
            wing_loss=dummy_loss,
            graph_loss=dummy_loss,
            landmark_weights=None,
            lambda_hm=1.0,
            lambda_cd=1.0,
            lambda_graph=1.0,
            device=torch.device("cpu"),
            img_size=img_size,
        )

        self.assertIn("per_landmark", metrics)
        self.assertEqual(len(metrics["per_landmark"]), NUM_LANDMARKS)
        self.assertAlmostEqual(metrics["per_landmark"][0]["mre_px"], 0.0, places=4)
        self.assertAlmostEqual(metrics["per_landmark"][0]["mre_mm"], 0.0, places=4)
        self.assertEqual(metrics["per_landmark"][0]["sdr_2.0mm"], 100.0)
        self.assertEqual(metrics["per_landmark"][0]["sdr_2.5px"], 100.0)

    def test_eval_compute_metrics_millimeter_and_pixel_reporting(self):
        """Verify that src.eval.compute_metrics correctly computes clinical mm and canvas px metrics."""
        from src.eval import compute_metrics, format_metrics_table

        num_samples = 4
        img_size = 640
        pixel_spacing = 0.1  # 0.1 mm per pixel

        gt_coords = np.full((num_samples, NUM_LANDMARKS, 2), 0.5, dtype=np.float32)
        pred_coords = gt_coords.copy()

        # Add 10 px radial error (1.0 mm) to landmark 0
        pred_coords[:, 0, 0] += 10.0 / img_size

        summary, landmarks, errors = compute_metrics(
            pred_coords=pred_coords,
            gt_coords=gt_coords,
            img_size=img_size,
            pixel_spacing=pixel_spacing,
        )

        # Total valid landmarks = 4 samples * 13 landmarks = 52
        # 4 landmarks have 10 px (1.0 mm) error, 48 landmarks have 0 px (0.0 mm) error
        # Mean error = (4 * 1.0) / 52 = 0.07692 mm (or in px: 40/52 = 0.7692 px)
        self.assertIn("mre_mm", summary)
        self.assertIn("rmse_mm", summary)
        self.assertIn("sdr_2.0mm", summary)
        self.assertIn("sdr_2.5mm", summary)
        self.assertIn("sdr_4.0mm", summary)
        self.assertIn("mre_pixels", summary)
        self.assertIn("sdr_2.5px", summary)

        self.assertAlmostEqual(summary["mre_mm"], summary["mre_pixels"] * pixel_spacing, places=5)
        # All 52 landmarks have error <= 1.0 mm, which is <= 2.0 mm
        self.assertEqual(summary["sdr_2.0mm"], 100.0)
        self.assertEqual(summary["sdr_2.5mm"], 100.0)
        # In px: 4 landmarks have 10 px error (> 2.5 px), so 48/52 = 92.31%
        self.assertAlmostEqual(summary["sdr_2.5px"], (48.0 / 52.0) * 100.0, places=2)

        # Verify ASCII table generation contains clinical mm header
        table = format_metrics_table(summary, landmarks, run_name="test_run", weights_path="best.pth")
        self.assertIn("CLINICAL METRIC (MILLIMETERS)", table)
        self.assertIn("CANVAS METRIC (PIXELS)", table)
        self.assertIn("SDR @ 2.0 mm (Clinical Standard)", table)

    def test_sample_adaptive_pixel_spacing_computation(self):
        """Verify that compute_pixel_spacing_for_sample correctly scales based on image dimensions."""
        from src.data.loaders.dataset import compute_pixel_spacing_for_sample
        from src.eval import compute_metrics

        # 1. Standard ISBI cephalogram (1935x2400) on 640 canvas
        s_isbi = compute_pixel_spacing_for_sample(1935, 2400, canvas_size=640, base_pixel_spacing=0.1)
        self.assertAlmostEqual(s_isbi, 0.375, places=4)

        # 2. Pre-resized cephalogram (640x640) on 640 canvas (represents full 2400-tall scan)
        s_640 = compute_pixel_spacing_for_sample(640, 640, canvas_size=640, base_pixel_spacing=0.1)
        self.assertAlmostEqual(s_640, 0.375, places=4)

        # 3. TUMS landscape cephalogram (2091x1676) on 640 canvas
        s_tums = compute_pixel_spacing_for_sample(2091, 1676, canvas_size=640, base_pixel_spacing=0.1)
        self.assertAlmostEqual(s_tums, 0.1 / (640.0 / 2091.0), places=4)

        # 4. Per-sample vector spacing in compute_metrics
        num_samples = 2
        img_size = 640
        gt_coords = np.full((num_samples, NUM_LANDMARKS, 2), 0.5, dtype=np.float32)
        pred_coords = gt_coords.copy()

        # Both samples have 10 px error along X on landmark 0
        pred_coords[:, 0, 0] += 10.0 / img_size

        # Sample 0 has spacing 0.2 mm/px -> 10 px = 2.0 mm
        # Sample 1 has spacing 0.3 mm/px -> 10 px = 3.0 mm
        spacings = np.array([0.2, 0.3], dtype=np.float32)

        summary, landmarks, errors = compute_metrics(
            pred_coords=pred_coords,
            gt_coords=gt_coords,
            img_size=img_size,
            pixel_spacing=spacings,
        )

        self.assertEqual(summary["pixel_spacing_mode"], "adaptive_sample_level")
        self.assertAlmostEqual(summary["pixel_spacing_mm_per_px"], 0.25, places=4)
        # Landmark 0: sample 0 has 2.0 mm (<= 2.0mm -> True), sample 1 has 3.0 mm (<= 2.0mm -> False)
        # So SDR@2.0mm on landmark 0 should be 50.0%
        lm0 = landmarks[0]
        self.assertEqual(lm0["sdr_2.0mm"], 50.0)
        # For SDR@3.0mm, both sample 0 (2.0mm) and sample 1 (3.0mm) pass -> 100.0%
        self.assertEqual(lm0["sdr_3.0mm"], 100.0)


if __name__ == "__main__":
    unittest.main()
