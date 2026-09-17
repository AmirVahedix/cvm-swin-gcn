import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.preprocessing.split_dataset import split_dataset, load_label_studio_mapping, load_fixed_id_set
from src.train import log_split_image_ids_to_mlflow


class TestSplitArtifacts(unittest.TestCase):

    def test_split_dataset_generates_label_studio_ids(self):
        """Verify that split_dataset maps image files to Label Studio task IDs when export JSON is provided."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / "images"
            labels_dir = tmp_path / "labels"
            output_dir = tmp_path / "output_dataset"
            export_file = tmp_path / "export.json"

            images_dir.mkdir()
            labels_dir.mkdir()

            # Create 8 dummy image/npz pairs: 0001 to 0008
            mock_tasks = []
            for i in range(1, 9):
                stem = f"{i:04d}"
                (images_dir / f"{stem}.jpg").write_bytes(b"dummy image data")
                (labels_dir / f"{stem}.npz").write_bytes(b"dummy npz data")
                # Assign distinct Label Studio task ID: e.g. 500 + i
                mock_tasks.append({
                    "id": 500 + i,
                    "data": {"img": f"https://example.com/data/local-files/?d=cvm-images/{stem}.jpg"}
                })

            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(mock_tasks, f)

            counts = split_dataset(
                images_dir=str(images_dir),
                labels_dir=str(labels_dir),
                output_dir=str(output_dir),
                train_ratio=0.50,
                val_ratio=0.25,
                test_ratio=0.25,
                seed=42,
                clean_output=True,
                export_json_path=str(export_file),
            )

            val_json = output_dir / "val_image_ids.json"
            test_json = output_dir / "test_image_ids.json"

            self.assertTrue(val_json.exists(), "val_image_ids.json should exist")
            self.assertTrue(test_json.exists(), "test_image_ids.json should exist")

            with open(val_json, "r", encoding="utf-8") as f:
                val_ids = json.load(f)
            with open(test_json, "r", encoding="utf-8") as f:
                test_ids = json.load(f)

            # Check that each file is an array/list
            self.assertIsInstance(val_ids, list)
            self.assertIsInstance(test_ids, list)
            self.assertEqual(len(val_ids), 2)
            self.assertEqual(len(test_ids), 2)

            # Check all elements are Label Studio IDs in range [501, 508]
            for vid in val_ids:
                self.assertIn(vid, range(501, 509))
            for tid in test_ids:
                self.assertIn(tid, range(501, 509))

            # Check no overlap between val and test
            self.assertEqual(len(set(val_ids).intersection(set(test_ids))), 0)

            # Check return dictionary contains metadata
            self.assertEqual(counts["val"], 2)
            self.assertEqual(counts["test"], 2)
            self.assertEqual(counts["val_ids"], val_ids)
            self.assertEqual(counts["test_ids"], test_ids)

    @patch("src.train.mlflow")
    def test_log_split_image_ids_existing_files(self, mock_mlflow):
        """Verify that existing JSON files are loaded and logged to active MLflow run."""
        mock_mlflow.active_run.return_value = MagicMock(info=MagicMock(run_id="test-run-123"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            val_json = tmp_path / "val_image_ids.json"
            test_json = tmp_path / "test_image_ids.json"

            expected_val = [101, 102]
            expected_test = [103, 104]

            with open(val_json, "w", encoding="utf-8") as f:
                json.dump(expected_val, f)
            with open(test_json, "w", encoding="utf-8") as f:
                json.dump(expected_test, f)

            val_ids, test_ids = log_split_image_ids_to_mlflow(
                output_dir=tmp_path,
            )

            self.assertEqual(val_ids, expected_val)
            self.assertEqual(test_ids, expected_test)

            # Ensure mlflow.log_artifact was called with both file paths
            called_paths = [call.args[0] for call in mock_mlflow.log_artifact.call_args_list]
            self.assertIn(str(val_json), called_paths)
            self.assertIn(str(test_json), called_paths)

    @patch("src.train.mlflow")
    def test_log_split_image_ids_fallback_directory_scan(self, mock_mlflow):
        """Verify that missing JSON files trigger a directory scan and file generation."""
        mock_mlflow.active_run.return_value = MagicMock(info=MagicMock(run_id="test-run-456"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            val_dir = tmp_path / "val" / "images"
            test_dir = tmp_path / "test" / "images"
            val_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)

            (val_dir / "0010.jpg").write_bytes(b"data")
            (val_dir / "0011.png").write_bytes(b"data")
            (test_dir / "0020.jpeg").write_bytes(b"data")

            val_ids, test_ids = log_split_image_ids_to_mlflow(
                val_img_dir=str(val_dir),
                test_img_dir=str(test_dir),
                output_dir=tmp_path,
            )

            self.assertEqual(val_ids, [11, 12])
            self.assertEqual(test_ids, [21])

            # Check that the files were created on disk as well
            val_json = tmp_path / "val_image_ids.json"
            test_json = tmp_path / "test_image_ids.json"
            self.assertTrue(val_json.exists())
            self.assertTrue(test_json.exists())

            # Check log_artifact calls
            called_paths = [call.args[0] for call in mock_mlflow.log_artifact.call_args_list]
            self.assertIn(str(val_json), called_paths)
            self.assertIn(str(test_json), called_paths)

    @patch("src.train.mlflow")
    def test_log_split_image_ids_error_tolerance(self, mock_mlflow):
        """Verify that MLflow exceptions do not crash execution (fault tolerance)."""
        mock_mlflow.active_run.return_value = MagicMock(info=MagicMock(run_id="test-run-789"))
        mock_mlflow.log_artifact.side_effect = RuntimeError("Simulated MLflow connection timeout")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            val_json = tmp_path / "val_image_ids.json"
            test_json = tmp_path / "test_image_ids.json"

            with open(val_json, "w", encoding="utf-8") as f:
                json.dump([1], f)
            with open(test_json, "w", encoding="utf-8") as f:
                json.dump([2], f)

            # Must not raise an exception
            try:
                val_ids, test_ids = log_split_image_ids_to_mlflow(
                    output_dir=tmp_path,
                )
            except Exception as e:
                self.fail(f"log_split_image_ids_to_mlflow raised an exception: {e}")

            self.assertEqual(val_ids, [1])
            self.assertEqual(test_ids, [2])

    def test_load_fixed_id_set_scalars_and_objects(self):
        """Verify that load_fixed_id_set correctly handles scalar arrays and arrays of dicts/objects."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            scalar_json = tmp_path / "scalars.json"
            with open(scalar_json, "w", encoding="utf-8") as f:
                json.dump([10, "20", 30], f)

            obj_json = tmp_path / "objects.json"
            with open(obj_json, "w", encoding="utf-8") as f:
                json.dump([
                    {"id": 40, "name": "sample_a"},
                    {"task_id": 50, "extra": "info"},
                    {"image_id": 60},
                    {"id": "70"},
                ], f)

            scalar_ids = load_fixed_id_set(scalar_json)
            self.assertEqual(scalar_ids, {10, 20, 30})

            obj_ids = load_fixed_id_set(obj_json)
            self.assertEqual(obj_ids, {40, 50, 60, 70})

    def test_split_dataset_fixed_test_and_val(self):
        """Verify that providing both fixed test and fixed val IDs allocates train, val, and test deterministically."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / "images"
            labels_dir = tmp_path / "labels"
            output_dir = tmp_path / "output_dataset"
            export_file = tmp_path / "export.json"

            images_dir.mkdir()
            labels_dir.mkdir()

            # Create 10 dummy pairs: 0001 to 0010, tasks 601 to 610
            mock_tasks = []
            for i in range(1, 11):
                stem = f"{i:04d}"
                (images_dir / f"{stem}.jpg").write_bytes(b"dummy image")
                (labels_dir / f"{stem}.npz").write_bytes(b"dummy npz")
                mock_tasks.append({
                    "id": 600 + i,
                    "data": {"img": f"cvm-images/{stem}.jpg"}
                })

            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(mock_tasks, f)

            test_ids_file = tmp_path / "test_ids.json"
            with open(test_ids_file, "w", encoding="utf-8") as f:
                json.dump([601, 602], f)

            val_ids_file = tmp_path / "val_ids.json"
            with open(val_ids_file, "w", encoding="utf-8") as f:
                json.dump([{"id": 603}, {"id": 604}], f)

            counts = split_dataset(
                images_dir=str(images_dir),
                labels_dir=str(labels_dir),
                output_dir=str(output_dir),
                export_json_path=str(export_file),
                fixed_test_ids_path=str(test_ids_file),
                fixed_val_ids_path=str(val_ids_file),
                clean_output=True,
            )

            self.assertEqual(counts["test"], 2)
            self.assertEqual(counts["val"], 2)
            self.assertEqual(counts["train"], 6)

            self.assertEqual(counts["test_ids"], [601, 602])
            self.assertEqual(counts["val_ids"], [603, 604])

            # Check files on disk in train, val, and test dirs
            train_imgs = {f.stem for f in (output_dir / "train" / "images").iterdir()}
            val_imgs = {f.stem for f in (output_dir / "val" / "images").iterdir()}
            test_imgs = {f.stem for f in (output_dir / "test" / "images").iterdir()}

            self.assertEqual(test_imgs, {"0001", "0002"})
            self.assertEqual(val_imgs, {"0003", "0004"})
            self.assertEqual(train_imgs, {"0005", "0006", "0007", "0008", "0009", "0010"})

            # Ensure all subsets are completely disjoint
            self.assertEqual(len(test_imgs.intersection(val_imgs)), 0)
            self.assertEqual(len(train_imgs.intersection(val_imgs)), 0)
            self.assertEqual(len(train_imgs.intersection(test_imgs)), 0)

    def test_split_dataset_overlap_detection(self):
        """Verify that an overlap between fixed test and val IDs raises a ValueError data leak error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / "images"
            labels_dir = tmp_path / "labels"
            output_dir = tmp_path / "output_dataset"

            images_dir.mkdir()
            labels_dir.mkdir()

            test_ids_file = tmp_path / "test_ids.json"
            val_ids_file = tmp_path / "val_ids.json"

            with open(test_ids_file, "w", encoding="utf-8") as f:
                json.dump([1, 2, 3], f)
            with open(val_ids_file, "w", encoding="utf-8") as f:
                json.dump([3, 4, 5], f)

            with self.assertRaises(ValueError) as ctx:
                split_dataset(
                    images_dir=str(images_dir),
                    labels_dir=str(labels_dir),
                    output_dir=str(output_dir),
                    fixed_test_ids_path=str(test_ids_file),
                    fixed_val_ids_path=str(val_ids_file),
                )
            self.assertIn("Data leakage detected", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
