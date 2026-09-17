import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scripts.generate_clinical_charts import (
    LANDMARK_CLASSES,
    generate_publication_dashboard,
    plot_ced_curve,
    plot_per_landmark_sdr_bar_chart,
    plot_radial_error_boxplot,
)


class TestClinicalCharts(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

        # Synthetic test data
        self.num_samples = 50
        self.num_landmarks = len(LANDMARK_CLASSES)
        np.random.seed(42)
        self.radial_errors_mm = np.random.gamma(shape=2.0, scale=0.6, size=(self.num_samples, self.num_landmarks))
        self.valid_mask = np.ones_like(self.radial_errors_mm, dtype=bool)

        # Synthetic metrics JSON
        self.metrics_json = {
            "run_name": "test_run",
            "summary": {
                "total_samples": self.num_samples,
                "total_valid_landmarks": self.num_samples * self.num_landmarks,
                "mre_mm": float(np.mean(self.radial_errors_mm)),
                "medre_mm": float(np.median(self.radial_errors_mm)),
                "sdr_2.0mm": float(np.mean(self.radial_errors_mm <= 2.0) * 100.0),
                "sdr_2.5mm": float(np.mean(self.radial_errors_mm <= 2.5) * 100.0),
            },
            "landmarks": [
                {
                    "id": i,
                    "name": name,
                    "mre_mm": float(np.mean(self.radial_errors_mm[:, i])),
                    "medre_mm": float(np.median(self.radial_errors_mm[:, i])),
                    "sdre_mm": float(np.std(self.radial_errors_mm[:, i])),
                    "sdr_2.0mm": float(np.mean(self.radial_errors_mm[:, i] <= 2.0) * 100.0),
                    "sdr_2.5mm": float(np.mean(self.radial_errors_mm[:, i] <= 2.5) * 100.0),
                }
                for i, name in enumerate(LANDMARK_CLASSES)
            ],
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plot_ced_curve_raw_and_json(self):
        p1 = self.out_dir / "ced_raw.png"
        plot_ced_curve(self.radial_errors_mm, self.valid_mask, self.metrics_json, output_path=p1)
        self.assertTrue(p1.exists())
        self.assertGreater(p1.stat().st_size, 1000)

        p2 = self.out_dir / "ced_json_only.png"
        plot_ced_curve(None, None, self.metrics_json, output_path=p2)
        self.assertTrue(p2.exists())
        self.assertGreater(p2.stat().st_size, 1000)

    def test_plot_per_landmark_sdr_bar_chart(self):
        p = self.out_dir / "bar_chart.png"
        plot_per_landmark_sdr_bar_chart(self.metrics_json, output_path=p)
        self.assertTrue(p.exists())
        self.assertGreater(p.stat().st_size, 1000)

    def test_plot_radial_error_boxplot(self):
        p1 = self.out_dir / "boxplot_raw.png"
        plot_radial_error_boxplot(self.radial_errors_mm, self.valid_mask, self.metrics_json, output_path=p1)
        self.assertTrue(p1.exists())
        self.assertGreater(p1.stat().st_size, 1000)

        p2 = self.out_dir / "boxplot_json_only.png"
        plot_radial_error_boxplot(None, None, self.metrics_json, output_path=p2)
        self.assertTrue(p2.exists())
        self.assertGreater(p2.stat().st_size, 1000)

    def test_generate_publication_dashboard(self):
        p = self.out_dir / "dashboard.png"
        generate_publication_dashboard(self.radial_errors_mm, self.valid_mask, self.metrics_json, output_path=p)
        self.assertTrue(p.exists())
        self.assertGreater(p.stat().st_size, 5000)


if __name__ == "__main__":
    unittest.main()
