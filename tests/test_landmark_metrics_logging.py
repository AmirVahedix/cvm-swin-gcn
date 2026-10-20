import json
import os
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
        # Set C3_PI (idx 5) and C3_AI (idx 7) to define known reference length (128 px)
        gt_coords[:, 5, 0] = 0.4
        gt_coords[:, 5, 1] = 0.5
        gt_coords[:, 7, 0] = 0.6
        gt_coords[:, 7, 1] = 0.5

        # Pred coordinates identical to GT initially
        pred_coords = gt_coords.copy()

        # Landmark 0: exactly 2.0 pixels error along X
        # 2.0 px error / 128.0 px ref = 1.5625%
        pred_coords[:, 0, 0] += 2.0 / img_size

        # Landmark 1: exactly 4.0 pixels error along Y
        # 4.0 px error / 128.0 px ref = 3.125%
        pred_coords[:, 1, 1] += 4.0 / img_size

        metrics_list = compute_per_landmark_metrics(pred_coords, gt_coords, img_size=img_size)

        self.assertEqual(len(metrics_list), num_landmarks)

        # Test Landmark 0 (2.0 px radial error = 1.5625% of C3 ref length)
        lm0 = metrics_list[0]
        self.assertEqual(lm0["id"], 0)
        self.assertEqual(lm0["name"], LANDMARK_CLASSES[0])
        self.assertEqual(lm0["count"], num_samples)
        self.assertAlmostEqual(lm0["mre_px"], 2.0, places=4)
        self.assertAlmostEqual(lm0["rmse_px"], 2.0, places=4)
        self.assertAlmostEqual(lm0["mean_norm_err"], 1.5625, places=4)
        self.assertEqual(lm0["share_within_2_5pct"], 100.0)
        self.assertEqual(lm0["share_within_5pct"], 100.0)
        self.assertEqual(lm0["sdr_2.0px"], 100.0)
        self.assertEqual(lm0["sdr_2.5px"], 100.0)
        self.assertEqual(lm0["sdr_3.0px"], 100.0)

        # Test Landmark 1 (4.0 px radial error = 3.125% of C3 ref length)
        lm1 = metrics_list[1]
        self.assertEqual(lm1["id"], 1)
        self.assertEqual(lm1["name"], LANDMARK_CLASSES[1])
        self.assertEqual(lm1["count"], num_samples)
        self.assertAlmostEqual(lm1["mre_px"], 4.0, places=4)
        self.assertAlmostEqual(lm1["rmse_px"], 4.0, places=4)
        self.assertAlmostEqual(lm1["mean_norm_err"], 3.125, places=4)
        self.assertEqual(lm1["share_within_2_5pct"], 0.0)
        self.assertEqual(lm1["share_within_5pct"], 100.0)
        self.assertEqual(lm1["sdr_2.0px"], 0.0)
        self.assertEqual(lm1["sdr_2.5px"], 0.0)
        self.assertEqual(lm1["sdr_3.0px"], 0.0)
        self.assertEqual(lm1["sdr_4.0px"], 100.0)

        # Test Landmark 2 (0 px error = 0% norm error)
        lm2 = metrics_list[2]
        self.assertAlmostEqual(lm2["mre_px"], 0.0, places=4)
        self.assertAlmostEqual(lm2["mean_norm_err"], 0.0, places=4)
        self.assertEqual(lm2["share_within_5pct"], 100.0)
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
                "share_within_5pct": 98.5,
                "share_within_2_5pct": 95.0,
                "share_within_7_5pct": 99.5,
                "share_within_10pct": 100.0,
                "mean_norm_err": 1.2,
                "median_norm_err": 1.1,
                "rmse_norm_err": 1.5,
                "sdr_2_0": 90.0,
                "sdr_2_5": 98.5,
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
            self.assertEqual(saved_data["summary"]["best_val_share_within_5pct"], 98.5)
            self.assertEqual(saved_data["summary"]["best_val_mae_px"], 1.15)
            self.assertEqual(saved_data["summary"]["best_val_sdr_2_5px"], 98.5)
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
        self.assertAlmostEqual(metrics["per_landmark"][0]["mean_norm_err"], 0.0, places=4)
        self.assertEqual(metrics["per_landmark"][0]["share_within_5pct"], 100.0)
        self.assertEqual(metrics["per_landmark"][0]["sdr_2.5px"], 100.0)

    def test_eval_compute_metrics_normalized_and_pixel_reporting(self):
        """Verify that src.eval.compute_metrics correctly computes normalized % and canvas px metrics."""
        from src.eval import compute_metrics, format_metrics_table

        num_samples = 4
        img_size = 640

        gt_coords = np.full((num_samples, NUM_LANDMARKS, 2), 0.5, dtype=np.float32)
        # Set C3_PI (idx 5) and C3_AI (idx 7) to define known reference length (128 px)
        gt_coords[:, 5, 0] = 0.4
        gt_coords[:, 5, 1] = 0.5
        gt_coords[:, 7, 0] = 0.6
        gt_coords[:, 7, 1] = 0.5

        pred_coords = gt_coords.copy()

        # Add 10 px radial error to landmark 0 -> 10 / 128 = 7.8125% norm error
        pred_coords[:, 0, 0] += 10.0 / img_size

        summary, landmarks, errors = compute_metrics(
            pred_coords=pred_coords,
            gt_coords=gt_coords,
            img_size=img_size,
        )

        self.assertIn("mean_norm_err", summary)
        self.assertIn("median_norm_err", summary)
        self.assertIn("rmse_norm_err", summary)
        self.assertIn("share_within_2_5pct", summary)
        self.assertIn("share_within_5pct", summary)
        self.assertIn("share_within_7_5pct", summary)
        self.assertIn("share_within_10pct", summary)
        self.assertIn("mae_pixels", summary)
        self.assertIn("rmse_pixels", summary)
        self.assertIn("sdr_2.5px", summary)

        # Total valid landmarks = 4 samples * 13 landmarks = 52
        # 4 landmarks have 10 px error (7.8125% norm error), 48 landmarks have 0 px error (0% norm error)
        # Mean norm error = (4 * 7.8125) / 52 = 0.60096%
        self.assertAlmostEqual(summary["mean_norm_err"], (4 * 7.8125) / 52, places=4)
        # For share <= 5%: 48 landmarks pass, 4 fail -> 48 / 52 = 92.31%
        self.assertAlmostEqual(summary["share_within_5pct"], (48.0 / 52.0) * 100.0, places=2)
        # For share <= 10%: all 52 pass -> 100.0%
        self.assertEqual(summary["share_within_10pct"], 100.0)

        # In px: 4 landmarks have 10 px error (> 2.5 px), so 48/52 = 92.31%
        self.assertAlmostEqual(summary["sdr_2.5px"], (48.0 / 52.0) * 100.0, places=2)

        # Verify ASCII table generation contains normalized header and canvas px
        table = format_metrics_table(summary, landmarks, run_name="test_run", weights_path="best.pth")
        self.assertIn("NORMALIZED METRIC (% OF C3 REF LENGTH)", table)
        self.assertIn("CANVAS METRIC (PIXELS)", table)
        self.assertIn("Share ≤ 5.0%", table)

    def test_reference_length_computation_and_normalization(self):
        """Verify that compute_reference_length accurately calculates C3 endplate width."""
        from src.utils.metrics import compute_reference_length

        # Batch of 2 samples
        gt_coords = np.full((2, NUM_LANDMARKS, 2), 0.5, dtype=np.float32)
        # Sample 0: C3_PI at (0.4, 0.5), C3_AI at (0.6, 0.5) -> dx = 0.2 * 640 = 128 px
        gt_coords[0, 5, 0] = 0.4
        gt_coords[0, 5, 1] = 0.5
        gt_coords[0, 7, 0] = 0.6
        gt_coords[0, 7, 1] = 0.5

        # Sample 1: C3_PI at (0.3, 0.4), C3_AI at (0.3, 0.7) -> dy = 0.3 * 640 = 192 px
        gt_coords[1, 5, 0] = 0.3
        gt_coords[1, 5, 1] = 0.4
        gt_coords[1, 7, 0] = 0.3
        gt_coords[1, 7, 1] = 0.7

        ref_lens = compute_reference_length(gt_coords, img_size=640)
        self.assertAlmostEqual(ref_lens[0], 128.0, places=4)
        self.assertAlmostEqual(ref_lens[1], 192.0, places=4)

        # Missing landmarks fallback safeguard (negative coordinates -> fallback to img_size)
        gt_missing = gt_coords.copy()
        gt_missing[0, 5, 0] = -1.0
        ref_lens_fallback = compute_reference_length(gt_missing, img_size=640)
        self.assertAlmostEqual(ref_lens_fallback[0], 640.0, places=4)

    def test_checkpoint_saving_clean_and_full(self):
        """Verify that best.pth contains only state_dict and best_full_checkpoint.pth contains optimizer state."""
        import tempfile
        import torch.nn as nn
        import torch.optim as optim

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "best.pth")
            full_ckpt_path = os.path.join(tmpdir, "best_full_checkpoint.pth")

            model = nn.Linear(4, 2)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

            # 1. Clean weights (~420MB in actual model)
            torch.save(model.state_dict(), save_path)

            # 2. Full checkpoint (~1.3GB in actual model)
            torch.save(
                {
                    "epoch": 5,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": 0.1234,
                    "val_mean_norm_err": 3.45,
                    "val_share_within_5pct": 88.5,
                },
                full_ckpt_path,
            )

            self.assertTrue(os.path.exists(save_path))
            self.assertTrue(os.path.exists(full_ckpt_path))

            # Verify clean checkpoint loads directly as state_dict
            loaded_clean = torch.load(save_path, weights_only=True)
            self.assertIn("weight", loaded_clean)
            self.assertIn("bias", loaded_clean)
            self.assertNotIn("optimizer_state_dict", loaded_clean)

            # Verify full checkpoint contains optimizer and metadata
            loaded_full = torch.load(full_ckpt_path, weights_only=False)
            self.assertIn("model_state_dict", loaded_full)
            self.assertIn("optimizer_state_dict", loaded_full)
            self.assertIn("scheduler_state_dict", loaded_full)
            self.assertEqual(loaded_full["epoch"], 5)
            self.assertEqual(loaded_full["val_mean_norm_err"], 3.45)
            self.assertEqual(loaded_full["val_share_within_5pct"], 88.5)

    def test_artifact_logging_order_charts_before_checkpoint(self):
        """Verify that evaluation/charts is logged to MLflow strictly BEFORE checkpoints/best.pth."""
        logged_artifacts = []

        def mock_log_artifact(local_path, artifact_path=None):
            logged_artifacts.append((local_path, artifact_path))

        def mock_log_artifacts(local_dir, artifact_path=None):
            logged_artifacts.append((local_dir, artifact_path))

        with patch("mlflow.log_artifact", side_effect=mock_log_artifact), \
             patch("mlflow.log_artifacts", side_effect=mock_log_artifacts):

            # Simulate Step 1 (JSON), Step 2 (Charts), Step 3 (Model)
            mock_log_artifact("evaluation/metrics.json", artifact_path="evaluation")
            mock_log_artifacts("evaluation/run_1/charts", artifact_path="evaluation/charts")
            mock_log_artifact("./artifacts/best.pth", artifact_path="checkpoints")

            paths = [a[1] for a in logged_artifacts]
            self.assertIn("evaluation/charts", paths)
            self.assertIn("checkpoints", paths)

            charts_idx = paths.index("evaluation/charts")
            ckpt_idx = paths.index("checkpoints")
            self.assertLess(charts_idx, ckpt_idx, "evaluation/charts must be logged before checkpoints/best.pth")


if __name__ == "__main__":
    unittest.main()


