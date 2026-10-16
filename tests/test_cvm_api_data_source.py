import os
import json
import tempfile
import unittest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.constants import LANDMARK_CLASSES, NUM_LANDMARKS
from src.data.preprocessing.download_data import (
    download_export_and_images,
    download_from_cvm_api,
    download_from_label_studio,
)
from src.data.preprocessing.resize_images import resize_images
from src.data.preprocessing.generate_labels import generate_labels, process_single_record
from src.data.preprocessing.split_dataset import load_label_studio_mapping, split_dataset
from src.data.loaders.dataloader import (
    load_stage_mapping,
    build_class_weighted_sampler,
    get_dataloaders,
)
from torch.utils.data import WeightedRandomSampler
from src.utils.verify_env import verify_env


class TestCvmApiDataSource(unittest.TestCase):

    def setUp(self):
        self.sample_landmarks = [
            {"id": "C2_PI", "x": 0.25, "y": 0.50},
            {"id": "C2_IC", "x": 0.30, "y": 0.52},
            {"id": "C2_AI", "x": 0.35, "y": 0.50},
            {"id": "C3_PS", "x": 0.26, "y": 0.60},
            {"id": "C3_AS", "x": 0.36, "y": 0.59},
            {"id": "C3_PI", "x": 0.25, "y": 0.65},
            {"id": "C3_IC", "x": 0.30, "y": 0.66},
            {"id": "C3_AI", "x": 0.35, "y": 0.65},
            {"id": "C4_PS", "x": 0.27, "y": 0.72},
            {"id": "C4_AS", "x": 0.37, "y": 0.71},
            {"id": "C4_PI", "x": 0.26, "y": 0.78},
            {"id": "C4_IC", "x": 0.31, "y": 0.79},
            {"id": "C4_AI", "x": 0.36, "y": 0.77},
        ]

    def test_verify_env_with_api_and_label_studio(self):
        """verify_env should succeed for API data source without requiring Label Studio env vars."""
        # For API data source:
        with patch("src.utils.verify_env.load_dotenv"), patch.dict(os.environ, {}, clear=True):
            try:
                verify_env(data_source="api")
            except SystemExit:
                self.fail("verify_env(data_source='api') raised SystemExit unexpectedly")

        # For Label Studio data source: should exit if missing
        with patch("src.utils.verify_env.load_dotenv"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                verify_env(data_source="label-studio")

    def test_cvm_api_download_skips_review_needed(self):
        """download_from_cvm_api should skip all tasks marked as review_needed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            export_dir = tmp_path / "exports"
            img_dir = tmp_path / "images"

            # Mock API response with 3 tasks: 1 and 2 normal, 3 marked review_needed
            mock_tasks = [
                {
                    "taskId": 1,
                    "filename": "1.jpg",
                    "imageUrl": "http://localhost:8000/images/1.jpg",
                    "reviewNeeded": False,
                    "landmarks": self.sample_landmarks,
                },
                {
                    "taskId": 2,
                    "filename": "2.jpg",
                    "imageUrl": "http://localhost:8000/images/2.jpg",
                    "reviewNeeded": False,
                    "landmarks": self.sample_landmarks,
                },
                {
                    "taskId": 3,
                    "filename": "3.jpg",
                    "imageUrl": "http://localhost:8000/images/3.jpg",
                    "reviewNeeded": True,  # MUST BE SKIPPED
                    "landmarks": self.sample_landmarks,
                },
            ]

            mock_session = MagicMock()
            # Health check 200
            mock_health = MagicMock()
            mock_health.status_code = 200
            # Export check 200
            mock_export = MagicMock()
            mock_export.status_code = 200
            mock_export.json.return_value = {"totalExported": 3, "data": mock_tasks}
            # Image download 200
            mock_img = MagicMock()
            mock_img.status_code = 200
            mock_img.iter_content.return_value = [b"mock image bytes"]

            def mock_get(url, **kwargs):
                if "/api/health" in url:
                    return mock_health
                elif "/api/training-data" in url or "/api/export" in url:
                    return mock_export
                elif "/images/" in url:
                    return mock_img
                return mock_health

            mock_session.get.side_effect = mock_get

            with patch("requests.Session", return_value=mock_session):
                count = download_from_cvm_api(
                    export_dir=str(export_dir),
                    img_dir=str(img_dir),
                    api_url="http://mock-cvm:8000",
                    skip_review_needed=True,
                )

            self.assertEqual(count, 2, "Should only download 2 non-review-needed images")
            self.assertTrue((img_dir / "1.jpg").exists(), "1.jpg should be downloaded")
            self.assertTrue((img_dir / "2.jpg").exists(), "2.jpg should be downloaded")
            self.assertFalse((img_dir / "3.jpg").exists(), "3.jpg (review_needed) MUST NOT be downloaded")

            # Check saved export JSON
            export_json_path = export_dir / "export.json"
            self.assertTrue(export_json_path.exists())
            with open(export_json_path, "r", encoding="utf-8") as f:
                saved_export = json.load(f)

            saved_tasks = saved_export.get("data", [])
            saved_ids = [t["taskId"] for t in saved_tasks]
            self.assertIn(1, saved_ids)
            self.assertIn(2, saved_ids)
            self.assertNotIn(3, saved_ids, "Task 3 (review_needed) must not be in export JSON")
            self.assertEqual(saved_export.get("skippedReviewNeeded"), 1)

    def test_api_unreachable_raises_connection_error(self):
        """When API is unreachable, download_from_cvm_api must raise ConnectionError without attempting any local fallback."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dest_export_dir = tmp_path / "exports"
            dest_img_dir = tmp_path / "raw_images"

            # Unreachable port ensures no connection can be made
            with self.assertRaises(ConnectionError) as ctx:
                download_from_cvm_api(
                    export_dir=str(dest_export_dir),
                    img_dir=str(dest_img_dir),
                    api_url="http://127.0.0.1:59999",  # unreachable port
                    skip_review_needed=True,
                )
            self.assertIn("Could not retrieve dataset from CVM Backend API", str(ctx.exception))

    def test_legacy_data_source_routing(self):
        """Calling download_export_and_images with data_source='label-studio' routes to legacy."""
        with patch("src.data.preprocessing.download_data.download_from_label_studio") as mock_ls, \
             patch("src.data.preprocessing.download_data.download_from_cvm_api") as mock_api:
            download_export_and_images(data_source="label-studio")
            mock_ls.assert_called_once()
            mock_api.assert_not_called()

        with patch("src.data.preprocessing.download_data.download_from_label_studio") as mock_ls, \
             patch("src.data.preprocessing.download_data.download_from_cvm_api") as mock_api:
            download_export_and_images(data_source="api")
            mock_api.assert_called_once()
            mock_ls.assert_not_called()

    def test_resize_images_with_cvm_api_format(self):
        """resize_images should process CVM API export, resize image with letterboxing, and skip review_needed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            img_dir = tmp_path / "raw_images"
            out_img_dir = tmp_path / "resized_images"
            img_dir.mkdir()

            # Create test image (1000 x 500)
            dummy_img = np.zeros((500, 1000, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "task1.jpg"), dummy_img)
            cv2.imwrite(str(img_dir / "task2.jpg"), dummy_img)

            export_file = tmp_path / "cvm_export.json"
            out_export_file = tmp_path / "resized_export.json"

            tasks_payload = {
                "dataSource": "cvm-backend-api",
                "data": [
                    {
                        "taskId": 1,
                        "filename": "task1.jpg",
                        "reviewNeeded": False,
                        "landmarks": [{"id": "C2_PI", "x": 0.5, "y": 0.5}],
                    },
                    {
                        "taskId": 2,
                        "filename": "task2.jpg",
                        "reviewNeeded": True,  # MUST BE SKIPPED
                        "landmarks": [{"id": "C2_PI", "x": 0.5, "y": 0.5}],
                    },
                ],
            }
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(tasks_payload, f)

            resize_images(
                img_dir=str(img_dir),
                json_path=str(export_file),
                out_img_dir=str(out_img_dir),
                out_json_path=str(out_export_file),
                target_size=640,
            )

            self.assertTrue((out_img_dir / "task1.jpg").exists())
            # Check resized dimension
            resized = cv2.imread(str(out_img_dir / "task1.jpg"))
            self.assertEqual(resized.shape[:2], (640, 640))

            # Task 2 (review_needed) must not be processed
            self.assertFalse((out_img_dir / "task2.jpg").exists())

            with open(out_export_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(len(saved["data"]), 1)
            self.assertEqual(saved["data"][0]["taskId"], 1)

    def test_generate_labels_with_cvm_api_format(self):
        """generate_labels should generate pristine NPZ heatmaps and coords from CVM API format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / "images"
            output_dir = tmp_path / "labels"
            images_dir.mkdir()

            # Create square 640x640 dummy image
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / "ceph_01.jpg"), dummy_img)
            cv2.imwrite(str(images_dir / "ceph_02.jpg"), dummy_img)

            export_file = tmp_path / "export.json"
            export_data = {
                "dataSource": "cvm-backend-api",
                "data": [
                    {
                        "taskId": 101,
                        "filename": "ceph_01.jpg",
                        "reviewNeeded": False,
                        "landmarks": self.sample_landmarks,
                    },
                    {
                        "taskId": 102,
                        "filename": "ceph_02.jpg",
                        "reviewNeeded": True,  # MUST BE SKIPPED
                        "landmarks": self.sample_landmarks,
                    },
                ],
            }
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f)

            processed = generate_labels(
                json_path=str(export_file),
                images_dir=str(images_dir),
                output_dir=str(output_dir),
                sigma=4.0,
            )

            self.assertEqual(processed, 1, "Only 1 label file should be generated")
            npz_path = output_dir / "ceph_01.npz"
            self.assertTrue(npz_path.exists())
            self.assertFalse((output_dir / "ceph_02.npz").exists(), "Review needed file must not exist")

            data = np.load(npz_path)
            heatmaps = data["heatmaps"]
            coords = data["coords"]

            self.assertEqual(heatmaps.shape, (NUM_LANDMARKS, 640, 640))
            self.assertEqual(coords.shape, (NUM_LANDMARKS, 2))
            # Landmark 0 is C2_PI at (0.25, 0.50)
            self.assertAlmostEqual(coords[0, 0], 0.25, places=4)
            self.assertAlmostEqual(coords[0, 1], 0.50, places=4)

    def test_split_dataset_mapping_with_cvm_api_format(self):
        """load_label_studio_mapping should parse CVM API export format correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            export_file = tmp_path / "export.json"
            export_data = {
                "dataSource": "cvm-backend-api",
                "data": [
                    {"taskId": 123, "filename": "123.jpg"},
                    {"taskId": 456, "filename": "tums_0001.jpg"},
                ],
            }
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f)

            mapping = load_label_studio_mapping(export_json_path=str(export_file))
            self.assertEqual(mapping.get("123.jpg"), 123)
            self.assertEqual(mapping.get("123"), 123)
            self.assertEqual(mapping.get("tums_0001.jpg"), 456)
            self.assertEqual(mapping.get("tums_0001"), 456)

    def test_split_dataset_with_test_cohort(self):
        """split_dataset should isolate images marked with isTestCohort strictly into test split."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / "images"
            labels_dir = tmp_path / "labels"
            output_dir = tmp_path / "dataset"
            images_dir.mkdir()
            labels_dir.mkdir()

            # Create 10 dummy image and label files (1.jpg .. 10.jpg, 1.npz .. 10.npz)
            dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
            for i in range(1, 11):
                img_file = images_dir / f"{i}.jpg"
                cv2.imwrite(str(img_file), dummy_img)
                label_file = labels_dir / f"{i}.npz"
                np.savez(label_file, heatmaps=np.zeros((13, 10, 10)), coords=np.zeros((13, 2)))

            # Mark tasks 1 and 2 as test_cohort
            export_file = tmp_path / "export.json"
            export_data = {
                "dataSource": "cvm-backend-api",
                "data": [
                    {
                        "taskId": i,
                        "filename": f"{i}.jpg",
                        "isTestCohort": (i in [1, 2]),
                    }
                    for i in range(1, 11)
                ],
            }
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f)

            split_res = split_dataset(
                images_dir=str(images_dir),
                labels_dir=str(labels_dir),
                output_dir=str(output_dir),
                train_ratio=0.8,
                val_ratio=0.2,
                test_ratio=0.0,
                export_json_path=str(export_file),
                use_test_cohort=True,
            )

            # Test split should contain strictly task 1 and 2
            test_images = sorted([p.name for p in (output_dir / "test" / "images").glob("*.jpg")])
            self.assertEqual(test_images, ["1.jpg", "2.jpg"])

            # Train and val splits combined should have remaining tasks 3..10
            train_images = set(p.name for p in (output_dir / "train" / "images").glob("*.jpg"))
            val_images = set(p.name for p in (output_dir / "val" / "images").glob("*.jpg"))

            # No overlap between test, train, and val
            self.assertTrue(train_images.isdisjoint(set(test_images)))
            self.assertTrue(val_images.isdisjoint(set(test_images)))
            self.assertTrue(train_images.isdisjoint(val_images))

            remaining_expected = {f"{i}.jpg" for i in range(3, 11)}
            self.assertEqual(train_images.union(val_images), remaining_expected)

            self.assertEqual(split_res["test"], 2)
            self.assertEqual(split_res["train"] + split_res["val"], 8)

    def test_class_weighted_sampling_and_stage_mapping(self):
        """build_class_weighted_sampler should calculate stage counts and assign equal sampling probability per stage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            export_file = tmp_path / "export.json"

            # Create an imbalanced dataset across all 6 CVM stages:
            # Stage 1: 10 samples
            # Stage 2: 20 samples
            # Stage 3: 5 samples
            # Stage 4: 15 samples
            # Stage 5: 50 samples
            # Stage 6: 25 samples
            stage_distribution = {1: 10, 2: 20, 3: 5, 4: 15, 5: 50, 6: 25}
            train_files = []
            tasks_data = []

            task_counter = 1
            for stage, count in stage_distribution.items():
                for _ in range(count):
                    fn = f"task_{task_counter:04d}.jpg"
                    train_files.append(fn)
                    tasks_data.append({
                        "taskId": task_counter,
                        "filename": fn,
                        "finalStage": stage,
                    })
                    task_counter += 1

            with open(export_file, "w", encoding="utf-8") as f:
                json.dump({"dataSource": "cvm-backend-api", "data": tasks_data}, f)

            # 1. Test load_stage_mapping
            mapping = load_stage_mapping(export_json_path=str(export_file))
            self.assertIn("task_0001.jpg", mapping)
            self.assertIn("task_0001", mapping)
            self.assertIn("1", mapping)
            self.assertEqual(mapping["task_0001.jpg"], 1)

            # 2. Test build_class_weighted_sampler
            sampler = build_class_weighted_sampler(
                train_files=train_files,
                export_json_path=str(export_file),
            )
            self.assertIsNotNone(sampler)
            self.assertIsInstance(sampler, WeightedRandomSampler)
            self.assertTrue(sampler.replacement)
            self.assertEqual(sampler.num_samples, len(train_files))

            # Verify sampling weights:
            # For any stage s, sum of weights across all samples in stage s must be equal (1.0)
            weights = sampler.weights.tolist()
            weights_by_stage = {s: [] for s in range(1, 7)}
            for fn, wt in zip(train_files, weights):
                st = mapping[fn]
                weights_by_stage[st].append(wt)

            for s in range(1, 7):
                count_s = stage_distribution[s]
                # Individual sample weight must be 1.0 / count_s
                expected_sample_weight = 1.0 / count_s
                for w in weights_by_stage[s]:
                    self.assertAlmostEqual(w, expected_sample_weight, places=6)
                # Sum of weights for stage s must equal 1.0
                self.assertAlmostEqual(sum(weights_by_stage[s]), 1.0, places=6)

            # Therefore, probability of picking any stage is 1.0 / 6.0 = 16.666%
            total_sum_weights = sum(weights)
            for s in range(1, 7):
                prob_s = sum(weights_by_stage[s]) / total_sum_weights
                self.assertAlmostEqual(prob_s, 1.0 / 6.0, places=6)

    def test_get_dataloaders_with_balanced_sampling(self):
        """get_dataloaders should configure WeightedRandomSampler on train_loader when use_balanced_sampling=True."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_img_dir = tmp_path / "train_imgs"
            train_npz_dir = tmp_path / "train_npz"
            val_img_dir = tmp_path / "val_imgs"
            val_npz_dir = tmp_path / "val_npz"
            for d in [train_img_dir, train_npz_dir, val_img_dir, val_npz_dir]:
                d.mkdir()

            dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
            tasks_data = []
            for i in range(1, 7):
                fn = f"{i}.jpg"
                cv2.imwrite(str(train_img_dir / fn), dummy_img)
                cv2.imwrite(str(val_img_dir / fn), dummy_img)
                np.savez(train_npz_dir / f"{i}.npz", heatmaps=np.zeros((13, 10, 10)), coords=np.zeros((13, 2)))
                np.savez(val_npz_dir / f"{i}.npz", heatmaps=np.zeros((13, 10, 10)), coords=np.zeros((13, 2)))
                tasks_data.append({"taskId": i, "filename": fn, "finalStage": i})

            export_file = tmp_path / "stage_mapping.json"
            stage_map = {f"{i}.jpg": i for i in range(1, 7)}
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(stage_map, f)

            # Balanced sampling enabled
            train_loader, val_loader = get_dataloaders(
                train_img_dir=str(train_img_dir),
                train_npz_dir=str(train_npz_dir),
                val_img_dir=str(val_img_dir),
                val_npz_dir=str(val_npz_dir),
                batch_size=2,
                num_workers=0,
                use_balanced_sampling=True,
                export_json_path=str(export_file),
            )
            self.assertIsInstance(train_loader.sampler, WeightedRandomSampler)

            # Balanced sampling disabled
            train_loader_unweighted, _ = get_dataloaders(
                train_img_dir=str(train_img_dir),
                train_npz_dir=str(train_npz_dir),
                val_img_dir=str(val_img_dir),
                val_npz_dir=str(val_npz_dir),
                batch_size=2,
                num_workers=0,
                use_balanced_sampling=False,
            )
            self.assertNotIsInstance(train_loader_unweighted.sampler, WeightedRandomSampler)


if __name__ == "__main__":
    unittest.main()


