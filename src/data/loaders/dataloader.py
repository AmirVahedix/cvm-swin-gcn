import os
import json
import torch
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse
from .dataset import CVMDataset
from .transforms import get_transforms
from torch.utils.data import DataLoader, WeightedRandomSampler


TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_NPZ_DIR = "dataset/train/labels"

VAL_IMG_DIR = "dataset/val/images"
VAL_NPZ_DIR = "dataset/val/labels"

TEST_IMG_DIR = "dataset/test/images"
TEST_NPZ_DIR = "dataset/test/labels"


def load_stage_mapping(export_json_path=None) -> dict[str, int]:
    """
    Loads mapping of image filenames / stems / IDs to their CVM stage (1 to 6).
    Checks candidate export paths in priority order:
    1. export_json_path (if explicitly provided)
    2. dataset/stage_mapping.json
    3. data/exports/export.json
    4. data/raw/exports/*.json
    """
    candidate_paths = []
    if export_json_path:
        candidate_paths.append(Path(export_json_path))
    candidate_paths.append(Path("dataset/stage_mapping.json"))
    candidate_paths.append(Path("data/exports/export.json"))

    raw_exports_dir = Path("data/raw/exports")
    if raw_exports_dir.exists():
        raw_files = sorted(
            raw_exports_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidate_paths.extend(raw_files)

    found_path = None
    for p in candidate_paths:
        if p.exists() and p.is_file():
            found_path = p
            break

    stage_map = {}
    if not found_path:
        return stage_map

    try:
        with open(found_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # If it's already a simple key-value mapping {filename: stage}
        if isinstance(raw_data, dict) and "data" not in raw_data and not any(isinstance(v, (dict, list)) for v in raw_data.values()):
            for k, v in raw_data.items():
                try:
                    iv = int(v)
                    if 1 <= iv <= 6:
                        stage_map[str(k)] = iv
                        stage_map[Path(str(k)).stem] = iv
                except (ValueError, TypeError):
                    pass
            return stage_map

        tasks = raw_data.get("data", raw_data) if isinstance(raw_data, dict) else raw_data
        if not isinstance(tasks, list):
            return stage_map

        for task in tasks:
            if not isinstance(task, dict):
                continue

            stage = (
                task.get("finalStage")
                or task.get("final_stage")
                or task.get("cvmStage")
                or task.get("cvm_stage")
                or task.get("stage")
            )
            if stage is None and isinstance(task.get("data"), dict):
                stage = task["data"].get("finalStage") or task["data"].get("cvmStage")

            if stage is not None:
                try:
                    int_stage = int(stage)
                    if not (1 <= int_stage <= 6):
                        continue
                except (ValueError, TypeError):
                    continue

                fn = task.get("filename")
                if not fn:
                    raw_path = task.get("file_upload") or task.get("data", {}).get("img") or task.get("data", {}).get("image")
                    if raw_path:
                        fn = os.path.basename(urlparse(raw_path).path)
                if fn:
                    stage_map[fn] = int_stage
                    stage_map[Path(fn).stem] = int_stage

                task_id = task.get("taskId") or task.get("task_id") or task.get("id")
                if task_id is not None:
                    stage_map[str(task_id)] = int_stage
                    stage_map[f"{task_id}.jpg"] = int_stage
                    if str(task_id).isdigit():
                        stage_map[f"{int(task_id):04d}.jpg"] = int_stage

    except Exception as e:
        print(f"⚠️ Warning loading stage mapping from {found_path}: {e}")

    return stage_map


def build_class_weighted_sampler(
    train_files: list[str],
    export_json_path=None,
    num_samples: int | None = None,
    power: float | None = None,
) -> WeightedRandomSampler | None:
    """
    Constructs a WeightedRandomSampler based on 'finalStage' (stages 1 to 6)
    using smoothed inverse class frequency to balance gradient signals without
    over-cycling small minority cohorts.

    1. Calculates the count N_s of each stage in train_files.
    2. Calculates class weight W_s = (1.0 / N_s) ** power for each stage.
       - power=1.0: Strict 1/N balanced sampling (equal stage probability).
       - power=0.5: Smoothed 1/sqrt(N) sampling (balances gradients while preserving sample diversity).
       - power=0.0: Uniform random sampling (equivalent to unweighted).
    3. Assigns sample weight w_i = W_{s_i} for each sample i.
    4. Uses WeightedRandomSampler(replacement=True).
    """
    if not train_files:
        return None

    if power is None:
        try:
            power = float(os.getenv("BALANCED_SAMPLING_POWER", "0.5"))
        except (ValueError, TypeError):
            power = 0.5

    stage_map = load_stage_mapping(export_json_path=export_json_path)
    if not stage_map:
        print("⚠️ No stage mapping found. Class-weighted sampling will be disabled.")
        return None

    sample_stages = []
    unmapped_count = 0
    for f in train_files:
        stem = Path(f).stem
        st = stage_map.get(f) or stage_map.get(stem)
        if st is None and stem.isdigit():
            st = stage_map.get(str(int(stem)))
        sample_stages.append(st)
        if st is None:
            unmapped_count += 1

    valid_stages = [s for s in sample_stages if s is not None and 1 <= s <= 6]
    if not valid_stages:
        print("⚠️ No valid CVM stages (1-6) found among training samples. Skipping class-weighted sampler.")
        return None

    stage_counts = Counter(valid_stages)
    active_stages = sorted(stage_counts.keys())

    # Smoothed class weight W_s = (1.0 / count(s)) ** power
    class_weights = {s: (1.0 / stage_counts[s]) ** power for s in active_stages}
    mean_weight = sum(class_weights.values()) / len(class_weights)

    sample_weights = []
    for st in sample_stages:
        if st is not None and st in class_weights:
            sample_weights.append(class_weights[st])
        else:
            # For unmapped samples, assign the mean class weight
            sample_weights.append(mean_weight)

    total_samples = len(sample_weights)
    target_num_samples = num_samples if num_samples is not None else total_samples

    print(f"\n⚖️  Class-Weighted Balanced Sampling Configured (CVM Stages 1-6, Power={power:.2f}):")
    print(f"   Active stages present: {len(active_stages)} / 6 ({active_stages})")
    for s in range(1, 7):
        cnt = stage_counts.get(s, 0)
        wt = class_weights.get(s, 0.0)
        pct = (cnt / total_samples * 100.0) if total_samples > 0 else 0.0
        print(f"   • Stage {s}: {cnt:4d} images ({pct:5.1f}%) | Sampling Weight: {wt:.6f}")
    if unmapped_count > 0:
        print(f"   • Unmapped stage: {unmapped_count} images (defaulted to mean weight {mean_weight:.6f})")
    print(f"   • Total samples drawn per epoch: {target_num_samples} (with replacement=True)\n")

    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=target_num_samples,
        replacement=True,
    )
    return sampler


def build_off_midline_weighted_sampler(
    train_files: list[str],
    labels_dir: str = TRAIN_NPZ_DIR,
    off_midline_threshold: float = 0.10,
    boost_weight: float = 3.0,
    base_weight: float = 1.0,
    num_samples: int | None = None,
    include_c2: bool = False,
) -> WeightedRandomSampler | None:
    """
    Constructs a WeightedRandomSampler that boosts the sampling weight for images
    that exhibit an off-midline concavity on C3 or C4 (and optionally C2 if include_c2=True).

    Anatomical reasoning:
    - Many models develop a central bias, predicting concavity dips at the exact geometric midpoint (50%).
    - In lateral cephalograms, true anatomical concavities often develop asymmetrically along the endplate.
    - C2 naturally exhibits odontoid curvature which causes false-positive oversampling if included.
      Therefore, by default, only C3 and C4 are evaluated for off-midline boosting.

    Args:
        train_files: List of training image file paths or names.
        labels_dir: Directory containing precomputed .npz ground truth arrays.
        off_midline_threshold: Minimum absolute deviation from 0.50 (|ratio - 0.50|) to qualify as off-midline.
        boost_weight: Sampling weight assigned to images with at least one off-midline vertebra.
        base_weight: Sampling weight assigned to centered/symmetric images.
        num_samples: Total samples to draw per epoch (defaults to len(train_files)).
        include_c2: Whether to allow C2 off-midline to trigger boosting (default: False).

    Returns:
        WeightedRandomSampler configured with per-sample weights, or None if train_files is empty.
    """
    if not train_files:
        return None

    import numpy as np

    # Vertebrae triplets (PI, IC, AI)
    # Default to C3 and C4 to prevent C2 false-positive oversampling
    triplets = [("C3", 5, 6, 7), ("C4", 10, 11, 12)]
    if include_c2:
        triplets.insert(0, ("C2", 0, 1, 2))

    sample_weights = []
    boosted_count = 0
    vert_counts = {name: 0 for name, _, _, _ in triplets}

    for f in train_files:
        stem = Path(f).stem
        npz_path = os.path.join(labels_dir, f"{stem}.npz")

        is_off_midline = False
        if os.path.exists(npz_path):
            try:
                npz_data = np.load(npz_path)
                coords = npz_data["coords"]  # [13, 2]

                for vert_name, pi_idx, ic_idx, ai_idx in triplets:
                    pi = coords[pi_idx]
                    ic = coords[ic_idx]
                    ai = coords[ai_idx]

                    # Ensure landmarks are valid
                    if (pi >= 0).all() and (ic >= 0).all() and (ai >= 0).all():
                        base_v = ai - pi
                        base_len_sq = np.sum(base_v ** 2)
                        if base_len_sq > 1e-6:
                            # Tangential projection ratio: (ic - pi) . (ai - pi) / ||ai - pi||^2
                            proj_ratio = float(np.dot(ic - pi, base_v) / base_len_sq)
                            if abs(proj_ratio - 0.50) >= off_midline_threshold:
                                is_off_midline = True
                                vert_counts[vert_name] += 1
                                # If any vertebra qualifies, image is boosted
                                break
            except Exception:
                pass

        if is_off_midline:
            sample_weights.append(float(boost_weight))
            boosted_count += 1
        else:
            sample_weights.append(float(base_weight))

    total_samples = len(sample_weights)
    target_num_samples = num_samples if num_samples is not None else total_samples

    pct_boosted = (boosted_count / total_samples * 100.0) if total_samples > 0 else 0.0
    vert_trigger_str = ", ".join(f"{k}: {v}" for k, v in vert_counts.items())
    print(f"\n🎯 Off-Midline Concavity Weighted Sampling Configured:")
    print(f"   • Evaluated vertebrae:            {[t[0] for t in triplets]}")
    print(f"   • Off-midline threshold (|ratio - 0.50|): >= {off_midline_threshold:.3f}")
    print(f"   • Boosted images:                 {boosted_count}/{total_samples} ({pct_boosted:.1f}%) -> Weight: {boost_weight:.1f}x")
    print(f"   • Baseline/Centered images:       {total_samples - boosted_count}/{total_samples} ({100.0 - pct_boosted:.1f}%) -> Weight: {base_weight:.1f}x")
    print(f"   • Vertebra trigger counts:        {vert_trigger_str}")
    print(f"   • Total samples drawn per epoch:  {target_num_samples} (with replacement=True)\n")

    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=target_num_samples,
        replacement=True,
    )
    return sampler


def get_dataloaders(
    train_img_dir=TRAIN_IMG_DIR,
    train_npz_dir=TRAIN_NPZ_DIR,
    val_img_dir=VAL_IMG_DIR,
    val_npz_dir=VAL_NPZ_DIR,
    batch_size=8,
    img_size=640,
    num_workers=4,
    use_balanced_sampling=True,
    sampling_power: float | None = None,
    export_json_path=None,
):
    train_transform, val_transform = get_transforms(img_size=img_size)

    train_files = [
        f for f in os.listdir(train_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    val_files = [
        f for f in os.listdir(val_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    train_dataset = CVMDataset(
        train_img_dir,
        train_npz_dir,
        train_files,
        transform=train_transform,
        img_size=img_size,
    )
    val_dataset = CVMDataset(
        val_img_dir,
        val_npz_dir,
        val_files,
        transform=val_transform,
        img_size=img_size,
    )

    sampler = None
    if use_balanced_sampling:
        sampler = build_class_weighted_sampler(
            train_files=train_files,
            export_json_path=export_json_path,
            power=sampling_power,
        )

    pin_memory = torch.cuda.is_available() or (num_workers > 0)

    if sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader


def get_test_dataloader(
    test_img_dir=TEST_IMG_DIR,
    test_npz_dir=TEST_NPZ_DIR,
    batch_size=8,
    img_size=640,
    num_workers=4,
):
    """
    Factory function specifically for the test dataset.
    """
    _, test_transform = get_transforms(img_size=img_size)

    test_files = [
        f for f in os.listdir(test_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    test_dataset = CVMDataset(
        image_dir=test_img_dir,
        npz_dir=test_npz_dir,
        image_filenames=test_files,
        transform=test_transform,
        img_size=img_size,
    )

    pin_memory = torch.cuda.is_available() or (num_workers > 0)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return test_loader


if __name__ == "__main__":

    def get_image_files(directory):
        return [
            f for f in os.listdir(directory) if f.endswith((".png", ".jpg", ".jpeg"))
        ]

    train_files = get_image_files(TRAIN_IMG_DIR)
    val_files = get_image_files(VAL_IMG_DIR)
    test_files = get_image_files(TEST_IMG_DIR)

    print(
        f"Loaded from disk - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}"
    )

    train_loader, val_loader = get_dataloaders(
        train_img_dir=TRAIN_IMG_DIR,
        train_npz_dir=TRAIN_NPZ_DIR,
        val_img_dir=VAL_IMG_DIR,
        val_npz_dir=VAL_NPZ_DIR,
    )
    test_loader = get_test_dataloader()

    print("\nVerifying Train Loader...")
    for batch in train_loader:
        print("Batch verification successful:")
        print(f" -> Images batch tensor shape:    {batch['image'].shape}")
        print(f" -> Heatmaps batch tensor shape:  {batch['heatmaps'].shape}")
        print(f" -> Coordinates batch tensor shape: {batch['coords'].shape}")
        break
