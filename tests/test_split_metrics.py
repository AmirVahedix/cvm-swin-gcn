import os
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.data.constants import LANDMARK_CLASSES, NUM_LANDMARKS
from src.eval import compute_and_log_split_metrics


class TestSplitMetricsGeneration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.labels_dir = self.temp_path / "labels"
        self.output_dir = self.temp_path / "evaluation"
        self.labels_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

        # Create dummy GT label .npz files for 4 samples: 2 val, 2 train
        self.sample_ids = ["val_01", "val_02", "train_01", "train_02"]
        np.random.seed(42)
        for s_id in self.sample_ids:
            # coords in [0.2, 0.8] range
            coords = np.random.uniform(0.2, 0.8, size=(NUM_LANDMARKS, 2)).astype(np.float32)
            np.savez(self.labels_dir / f"{s_id}.npz", coords=coords)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_and_log_split_metrics(self):
        # Create mock results_dict matching batch_inference output
        results_dict = {}
        for s_id in self.sample_ids:
            split = "val" if s_id.startswith("val") else "train"
            # Read back GT and add small perturbation for prediction
            gt = np.load(self.labels_dir / f"{s_id}.npz")["coords"]
            pred_px = {}
            for k, lm in enumerate(LANDMARK_CLASSES):
                pred_px[lm] = [float(gt[k, 0] * 640.0 + 1.5), float(gt[k, 1] * 640.0 - 1.0)]

            results_dict[s_id] = {
                "id": s_id,
                "filename": f"{s_id}.png",
                "split": split,
                "image_size": {"width": 640, "height": 640},
                "landmarks": pred_px,
                "normalized_landmarks": {lm: [coords[0] / 640.0, coords[1] / 640.0] for lm, coords in pred_px.items()},
            }

        split_results = compute_and_log_split_metrics(
            results_dict=results_dict,
            labels_dir=self.labels_dir,
            output_dir=self.output_dir,
            export_json_path=None,
            img_size=640,
            log_to_mlflow=False,
            splits_to_eval=["val", "train"],
        )

        self.assertIn("val", split_results)
        self.assertIn("train", split_results)

        # Verify validation set files
        val_json = self.output_dir / "metrics_val.json"
        val_txt = self.output_dir / "summary_report_val.txt"
        val_csv = self.output_dir / "predictions_landmarks_val.csv"
        self.assertTrue(val_json.exists(), "metrics_val.json should exist")
        self.assertTrue(val_txt.exists(), "summary_report_val.txt should exist")
        self.assertTrue(val_csv.exists(), "predictions_landmarks_val.csv should exist")

        with open(val_json, "r", encoding="utf-8") as f:
            v_data = json.load(f)
            self.assertEqual(v_data["split"], "val")
            self.assertEqual(v_data["total_evaluated_samples"], 2)
            self.assertIn("comprehensive", v_data)
            self.assertIn("summary", v_data)
            self.assertIn("landmarks", v_data)
            # Check key metric values
            self.assertIn("mean_norm_err", v_data["summary"])
            self.assertIn("median_norm_err", v_data["summary"])

        # Verify train set files
        train_json = self.output_dir / "metrics_train.json"
        train_txt = self.output_dir / "summary_report_train.txt"
        train_csv = self.output_dir / "predictions_landmarks_train.csv"
        self.assertTrue(train_json.exists(), "metrics_train.json should exist")
        self.assertTrue(train_txt.exists(), "summary_report_train.txt should exist")
        self.assertTrue(train_csv.exists(), "predictions_landmarks_train.csv should exist")

        with open(train_json, "r", encoding="utf-8") as f:
            t_data = json.load(f)
            self.assertEqual(t_data["split"], "train")
            self.assertEqual(t_data["total_evaluated_samples"], 2)
            self.assertIn("comprehensive", t_data)
            self.assertIn("summary", t_data)
            self.assertIn("landmarks", t_data)


if __name__ == "__main__":
    unittest.main()
