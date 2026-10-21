import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.batch_inference import (
    build_split_mapping,
    export_lightweight_predictions,
    parse_args,
)
from src.data.constants import LANDMARK_CLASSES


class TestBatchInferenceLightweightExport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_split_mapping_with_fixed_test_ids(self):
        # Create dummy test IDs json
        test_ids_file = self.temp_path / "test_image_ids.json"
        with open(test_ids_file, "w") as f:
            json.dump([1, 2], f)

        # Create dummy dataset directory structure
        dataset_dir = self.temp_path / "dataset"
        (dataset_dir / "test" / "images").mkdir(parents=True)
        (dataset_dir / "test" / "images" / "test_sample.png").touch()

        dummy_images = [
            Path("data/images/0001.jpg"),
            Path("data/images/0002.jpg"),
            Path("data/images/test_sample.png"),
            Path("data/images/0003.jpg"),
            Path("data/images/0004.jpg"),
            Path("data/images/0005.jpg"),
        ]

        split_map = build_split_mapping(
            image_paths=dummy_images,
            dataset_dir=dataset_dir,
            test_ids_path=test_ids_file,
            val_ids_path=None,
            seed=42,
            val_ratio=0.25,
        )

        # 0001 and 0002 match test_ids (1, 2)
        self.assertEqual(split_map["0001"], "test")
        self.assertEqual(split_map["0002"], "test")
        # test_sample matches test folder
        self.assertEqual(split_map["test_sample"], "test")

        # Remaining should be partitioned into train and val
        remaining = [split_map["0003"], split_map["0004"], split_map["0005"]]
        self.assertIn("train", remaining)
        self.assertIn("val", remaining)

    def test_export_lightweight_predictions(self):
        light_out = self.temp_path / "predictions_light"

        # Mock results_dict for 2 sample images
        results_dict = {
            "img_train_01": {
                "id": "img_train_01",
                "filename": "img_train_01.png",
                "split": "train",
                "image_size": {"width": 1000, "height": 1200},
                "landmarks": {lm: [100.5, 200.5] for lm in LANDMARK_CLASSES},
                "normalized_landmarks": {lm: [0.1005, 0.16708] for lm in LANDMARK_CLASSES},
            },
            "img_test_01": {
                "id": "img_test_01",
                "filename": "img_test_01.png",
                "split": "test",
                "image_size": {"width": 800, "height": 900},
                "landmarks": {lm: [350.0, 450.0] for lm in LANDMARK_CLASSES},
                "normalized_landmarks": {lm: [0.4375, 0.500] for lm in LANDMARK_CLASSES},
            },
        }

        meta = export_lightweight_predictions(results_dict, light_output_dir=light_out)

        # 1. Check individual files
        indiv_train = light_out / "individual" / "img_train_01.json"
        indiv_test = light_out / "individual" / "img_test_01.json"
        self.assertTrue(indiv_train.exists())
        self.assertTrue(indiv_test.exists())

        with open(indiv_train, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["id"], "img_train_01")
            self.assertEqual(data["split"], "train")
            self.assertIn("C2_PI", data["landmarks"])
            self.assertEqual(data["landmarks"]["C2_PI"], [100.5, 200.5])

        # 2. Check split subfolder files
        split_train_file = light_out / "train" / "img_train_01.json"
        split_test_file = light_out / "test" / "img_test_01.json"
        self.assertTrue(split_train_file.exists())
        self.assertTrue(split_test_file.exists())

        # 3. Check by_landmark files
        for lm in LANDMARK_CLASSES:
            lm_file = light_out / "by_landmark" / f"{lm}.json"
            self.assertTrue(lm_file.exists(), f"Missing landmark file: {lm_file}")
            with open(lm_file, "r", encoding="utf-8") as f:
                lm_data = json.load(f)
                self.assertIn("img_train_01", lm_data)
                self.assertIn("img_test_01", lm_data)
                self.assertEqual(lm_data["img_train_01"]["split"], "train")
                self.assertEqual(lm_data["img_train_01"]["x"], 100.5)

        # 4. Check aggregated JSON files
        all_file = light_out / "all_landmarks.json"
        self.assertTrue(all_file.exists())
        with open(all_file, "r", encoding="utf-8") as f:
            all_data = json.load(f)
            self.assertIn("img_train_01", all_data)
            self.assertIn("img_test_01", all_data)

        train_file = light_out / "train_landmarks.json"
        test_file = light_out / "test_landmarks.json"
        self.assertTrue(train_file.exists())
        self.assertTrue(test_file.exists())

        with open(train_file, "r", encoding="utf-8") as f:
            train_data = json.load(f)
            self.assertIn("img_train_01", train_data)
            self.assertNotIn("img_test_01", train_data)

        # 5. Check CSV file
        csv_file = light_out / "predictions_light.csv"
        self.assertTrue(csv_file.exists())
        with open(csv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 1)
            # Header line: id,split,width,height,landmark,x,y,norm_x,norm_y
            self.assertIn("landmark", lines[0])
            self.assertIn("split", lines[0])


if __name__ == "__main__":
    unittest.main()
