import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import cv2

from src.models.model import CephalometricSwinGCN
from src.eval_stage2_roi import run_stage2_roi_evaluation


class TestStage2ROIEval(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.img_dir = Path(self.temp_dir) / "images"
        self.npz_dir = Path(self.temp_dir) / "labels"
        self.out_dir = Path(self.temp_dir) / "eval_out"
        self.img_dir.mkdir(parents=True)
        self.npz_dir.mkdir(parents=True)
        self.out_dir.mkdir(parents=True)

        self.device = torch.device("cpu")

        # Create 2 synthetic images and labels
        self.samples = ["sample_001.png", "sample_002.png"]
        for s in self.samples:
            img = np.ones((640, 640, 3), dtype=np.uint8) * 120
            cv2.imwrite(str(self.img_dir / s), img)

            # 13 landmarks in normalized [0, 1]
            coords = np.array([
                [0.40, 0.45], [0.44, 0.47], [0.48, 0.46],
                [0.39, 0.52], [0.49, 0.51], [0.40, 0.58],
                [0.45, 0.60], [0.50, 0.59], [0.39, 0.64],
                [0.50, 0.63], [0.40, 0.70], [0.45, 0.72],
                [0.51, 0.71],
            ], dtype=np.float32)
            np.savez(self.npz_dir / f"{Path(s).stem}.npz", coords=coords)

        # Create dummy Stage 2 weights (448)
        self.stage2_weights = Path(self.temp_dir) / "stage2_dummy.pth"
        m2 = CephalometricSwinGCN(img_size=448, pretrained=False)
        torch.save({"model_state_dict": m2.state_dict()}, str(self.stage2_weights))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stage2_roi_evaluation_run(self):
        """Tests the standalone Stage 2 GT-ROI evaluation pipeline end-to-end on synthetic samples."""
        metrics, run_folder = run_stage2_roi_evaluation(
            stage2_weights=str(self.stage2_weights),
            img_dir=str(self.img_dir),
            npz_dir=str(self.npz_dir),
            output_dir=str(self.out_dir),
            split_name="test",
            run_name="test_stage2_roi_run",
            stage2_img_size=448,
            canvas_size=640,
            margin_ratio=0.18,
            bottom_margin_ratio=0.28,
            num_visualizations=2,
            device_str="cpu",
            native_img_dir=None,
            log_to_mlflow=False,
        )

        self.assertIn("general", metrics)
        self.assertIn("by_vertebra", metrics)
        self.assertIn("by_landmark", metrics)
        self.assertIn("staging", metrics)

        run_path = Path(run_folder)
        self.assertTrue((run_path / "summary_report.txt").exists())
        self.assertTrue((run_path / "metrics.json").exists())
        self.assertTrue((run_path / "predictions_landmarks.csv").exists())
        self.assertTrue((run_path / "visualizations").exists())


if __name__ == "__main__":
    unittest.main()
