import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import cv2

from src.models.model import CephalometricSwinGCN
from src.eval_two_stage import run_two_stage_evaluation


class TestTwoStageCascade(unittest.TestCase):
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

        # Create dummy weights for Stage 1 (640) and Stage 2 (448)
        self.stage1_weights = Path(self.temp_dir) / "stage1_dummy.pth"
        self.stage2_weights = Path(self.temp_dir) / "stage2_dummy.pth"

        m1 = CephalometricSwinGCN(img_size=640, pretrained=False)
        torch.save({"model_state_dict": m1.state_dict()}, str(self.stage1_weights))

        m2 = CephalometricSwinGCN(img_size=448, pretrained=False)
        torch.save({"model_state_dict": m2.state_dict()}, str(self.stage2_weights))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_two_stage_evaluation_run(self):
        """Tests the full two-stage evaluation pipeline end-to-end on synthetic samples."""
        metrics, run_folder = run_two_stage_evaluation(
            stage1_weights=str(self.stage1_weights),
            stage2_weights=str(self.stage2_weights),
            img_dir=str(self.img_dir),
            npz_dir=str(self.npz_dir),
            output_dir=str(self.out_dir),
            run_name="test_cascade_run",
            stage1_img_size=640,
            stage2_img_size=448,
            margin_ratio=0.18,
            num_visualizations=2,
            device_str="cpu",
            log_to_mlflow=False,
        )

        self.assertIn("stage1_alone", metrics)
        self.assertIn("stage2_oracle", metrics)
        self.assertIn("two_stage_cascade", metrics)
        self.assertIn("delta_comparison", metrics)

        run_path = Path(run_folder)
        self.assertTrue((run_path / "summary_report.txt").exists())
        self.assertTrue((run_path / "metrics.json").exists())
        self.assertTrue((run_path / "predictions_landmarks.csv").exists())

        # Check report text contains all required section headers
        with open(run_path / "summary_report.txt", "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("SECTION B.1: OVERALL NORMALIZED LANDMARK PERFORMANCE", content)
        self.assertIn("SECTION B.2: PER-VERTEBRA BREAKDOWN", content)
        self.assertIn("SECTION B.3: PER-LANDMARK PERFORMANCE", content)
        self.assertIn("SECTION E: TWO-STAGE CASCADED IMPROVEMENT COMPARISON", content)
        self.assertIn("Stage 1 Coarse", content)
        self.assertIn("Stage 2 Oracle", content)
        self.assertIn("Two-Stage Cascade", content)


if __name__ == "__main__":
    unittest.main()
