import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.constants import NUM_LANDMARKS
from src.data.loaders.dataset import generate_gaussian_heatmaps, get_image_dimensions_map
from src.data.loaders.transforms import get_transforms
from src.utils.roi_utils import (
    ROIBoundingBox,
    compute_roi_bbox,
    crop_roi_with_padding,
    inverse_transform_coords,
    transform_coords_to_roi,
)


class CVMROIDataset(Dataset):
    """
    Stage 2 ROI Dataset for high-resolution cervical vertebrae landmark regression.
    Crops strictly square, isotropic cervical spine ROIs with boundary padding,
    dynamically mapping landmark coordinates into normalized ROI space [0, 1].
    """
    def __init__(
        self,
        image_dir: str,
        npz_dir: str,
        image_filenames: List[str],
        native_img_dir: Optional[str] = None,
        dimensions_map: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable] = None,
        margin_ratio: float = 0.18,
        min_size: int = 180,
        target_size: int = 448,
        sigma: float = 4.0,
        jitter: Optional[Dict[str, float]] = None,
        is_train: bool = True,
        canvas_size: int = 640,
    ):
        self.image_dir = image_dir
        self.npz_dir = npz_dir
        self.image_filenames = sorted(image_filenames)
        self.native_img_dir = native_img_dir
        self.dimensions_map = dimensions_map if dimensions_map is not None else get_image_dimensions_map()
        self.transform = transform
        self.margin_ratio = margin_ratio
        self.min_size = min_size
        self.target_size = target_size
        self.sigma = sigma
        self.jitter = jitter if is_train else None
        self.is_train = is_train
        self.canvas_size = canvas_size

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_name = self.image_filenames[idx]
        stem = Path(img_name).stem

        # 1. Load ground truth landmarks from .npz (normalized [0, 1] canvas space)
        npz_path = os.path.join(self.npz_dir, f"{stem}.npz")
        data = np.load(npz_path)
        gt_coords_canvas = data["coords"].astype(np.float32)  # [13, 2]

        # 2. Compute isotropic square bounding box
        bbox = compute_roi_bbox(
            coords=gt_coords_canvas,
            img_w=self.canvas_size,
            img_h=self.canvas_size,
            margin_ratio=self.margin_ratio,
            min_size=self.min_size,
            jitter=self.jitter,
        )

        # 3. Determine if a high-resolution native radiograph is available
        native_crop_used = False
        native_img_path = None
        if self.native_img_dir:
            for ext in [".png", ".jpg", ".jpeg"]:
                cand = os.path.join(self.native_img_dir, f"{stem}{ext}")
                if os.path.exists(cand):
                    native_img_path = cand
                    break

        orig_w = None
        orig_h = None
        if img_name in self.dimensions_map:
            orig_w = self.dimensions_map[img_name].get("orig_w")
            orig_h = self.dimensions_map[img_name].get("orig_h")
        elif f"{stem}.jpg" in self.dimensions_map:
            orig_w = self.dimensions_map[f"{stem}.jpg"].get("orig_w")
            orig_h = self.dimensions_map[f"{stem}.jpg"].get("orig_h")
        elif f"{stem}.png" in self.dimensions_map:
            orig_w = self.dimensions_map[f"{stem}.png"].get("orig_w")
            orig_h = self.dimensions_map[f"{stem}.png"].get("orig_h")

        # If native image exists and has higher resolution than 640 canvas, crop directly from native image
        if native_img_path and orig_w and orig_h and max(orig_w, orig_h) > self.canvas_size:
            native_bgr = cv2.imread(native_img_path)
            if native_bgr is not None:
                native_rgb = cv2.cvtColor(native_bgr, cv2.COLOR_BGR2RGB)
                # Map bbox from canvas space to native space
                scale = float(self.canvas_size) / max(orig_w, orig_h)
                new_w = int(round(orig_w * scale))
                new_h = int(round(orig_h * scale))
                pad_x = (self.canvas_size - new_w) // 2
                pad_y = (self.canvas_size - new_h) // 2

                # Convert (u1, v1, D) to native pixel coordinates
                native_u1 = int(round((bbox.u1 - pad_x) / scale))
                native_v1 = int(round((bbox.v1 - pad_y) / scale))
                native_D = int(round(bbox.D / scale))

                native_bbox = ROIBoundingBox(
                    u1=native_u1,
                    v1=native_v1,
                    u2=native_u1 + native_D,
                    v2=native_v1 + native_D,
                    D=native_D,
                    center_u=native_u1 + native_D / 2.0,
                    center_v=native_v1 + native_D / 2.0,
                    span=bbox.span / scale,
                    img_w=orig_w,
                    img_h=orig_h,
                )

                roi_patch, _ = crop_roi_with_padding(
                    image=native_rgb,
                    bbox=native_bbox,
                    target_size=self.target_size,
                    interpolation=cv2.INTER_LINEAR,
                )
                native_crop_used = True

        # Fallback to cropping from 640 canvas
        if not native_crop_used:
            canvas_path = os.path.join(self.image_dir, img_name)
            canvas_bgr = cv2.imread(canvas_path)
            if canvas_bgr is None:
                raise FileNotFoundError(f"Image not found at {canvas_path}")
            canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

            roi_patch, _ = crop_roi_with_padding(
                image=canvas_rgb,
                bbox=bbox,
                target_size=self.target_size,
                interpolation=cv2.INTER_LINEAR,
            )

        # 4. Transform ground truth coordinates into normalized ROI space [0, 1]
        # with exact OpenCV half-pixel center correction
        roi_coords_norm = transform_coords_to_roi(
            coords=gt_coords_canvas,
            bbox=bbox,
            target_size=self.target_size,
            is_normalized_input=True,
        )

        # Convert to ROI pixel space for Albumentations
        roi_coords_px = roi_coords_norm * float(self.target_size)

        # 5. Apply synchronized spatial & photometric augmentations
        if self.transform is not None:
            augmented = self.transform(
                image=roi_patch,
                keypoints=roi_coords_px,
            )
            roi_patch = augmented["image"]
            aug_kp = augmented["keypoints"]
            if len(aug_kp) == len(roi_coords_px):
                transformed_coords_px = np.array(aug_kp, dtype=np.float32)
            else:
                transformed_coords_px = roi_coords_px.copy()
        else:
            transformed_coords_px = roi_coords_px.copy()

        # Convert image to float32 tensor
        if isinstance(roi_patch, np.ndarray):
            image_tensor = torch.from_numpy(np.transpose(roi_patch, (2, 0, 1))).float()
        else:
            image_tensor = roi_patch.float()

        # 6. Clip coordinates within valid ROI bounds [0, target_size - 1]
        valid_mask = (gt_coords_canvas[:, 0] >= 0) & (gt_coords_canvas[:, 1] >= 0)
        final_coords_px = np.full_like(transformed_coords_px, -1.0)
        for i in range(len(transformed_coords_px)):
            if valid_mask[i]:
                final_coords_px[i, 0] = np.clip(
                    transformed_coords_px[i, 0], 0.0, float(self.target_size - 1)
                )
                final_coords_px[i, 1] = np.clip(
                    transformed_coords_px[i, 1], 0.0, float(self.target_size - 1)
                )

        # 7. Dynamically generate Gaussian heatmaps at target_size
        heatmaps_tensor = generate_gaussian_heatmaps(
            coords_px=final_coords_px,
            img_size=self.target_size,
            sigma=self.sigma,
        )

        # 8. Final normalized ROI coordinates in [0, 1]
        final_coords_norm = np.full_like(final_coords_px, -1.0)
        for i in range(len(final_coords_px)):
            if valid_mask[i]:
                final_coords_norm[i] = final_coords_px[i] / float(self.target_size)

        coords_tensor = torch.tensor(final_coords_norm, dtype=torch.float32)

        return {
            "image": image_tensor,
            "heatmaps": heatmaps_tensor,
            "coords": coords_tensor,
            "filename": img_name,
            "bbox_u1": float(bbox.u1),
            "bbox_v1": float(bbox.v1),
            "bbox_D": float(bbox.D),
            "canvas_size": float(self.canvas_size),
            "target_size": float(self.target_size),
        }


def get_roi_dataloaders(
    train_img_dir: str = "dataset/train/images",
    train_npz_dir: str = "dataset/train/labels",
    val_img_dir: str = "dataset/val/images",
    val_npz_dir: str = "dataset/val/labels",
    batch_size: int = 8,
    target_size: int = 448,
    sigma: float = 4.0,
    margin_ratio: float = 0.18,
    jitter: Optional[Dict[str, float]] = None,
    num_workers: int = 4,
    use_balanced_sampling: bool = True,
    sampling_power: Optional[float] = None,
    export_json_path: Optional[str] = None,
    native_img_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds training and validation DataLoaders for Stage 2 ROI training.
    """
    from src.data.loaders.dataloader import (
        build_class_weighted_sampler,
        load_stage_mapping,
    )

    train_transform, val_transform = get_transforms(img_size=target_size)

    train_files = sorted([
        f for f in os.listdir(train_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ])
    val_files = sorted([
        f for f in os.listdir(val_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ])

    default_jitter = jitter if jitter is not None else {"trans": 0.06, "scale": 0.08}

    train_dataset = CVMROIDataset(
        image_dir=train_img_dir,
        npz_dir=train_npz_dir,
        image_filenames=train_files,
        native_img_dir=native_img_dir,
        transform=train_transform,
        margin_ratio=margin_ratio,
        target_size=target_size,
        sigma=sigma,
        jitter=default_jitter,
        is_train=True,
    )

    val_dataset = CVMROIDataset(
        image_dir=val_img_dir,
        npz_dir=val_npz_dir,
        image_filenames=val_files,
        native_img_dir=native_img_dir,
        transform=val_transform,
        margin_ratio=margin_ratio,
        target_size=target_size,
        sigma=sigma,
        jitter=None,
        is_train=False,
    )

    train_sampler = None
    shuffle = True
    if use_balanced_sampling:
        train_sampler = build_class_weighted_sampler(
            train_files, export_json_path=export_json_path, power=sampling_power
        )
        if train_sampler is not None:
            shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def get_test_roi_dataloader(
    test_img_dir: str = "dataset/test/images",
    test_npz_dir: str = "dataset/test/labels",
    batch_size: int = 8,
    target_size: int = 448,
    margin_ratio: float = 0.18,
    sigma: float = 4.0,
    native_img_dir: Optional[str] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Builds DataLoader for Stage 2 test evaluation on ground-truth ROIs (oracle evaluation).
    """
    _, test_transform = get_transforms(img_size=target_size)
    test_files = sorted([
        f for f in os.listdir(test_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ])

    test_dataset = CVMROIDataset(
        image_dir=test_img_dir,
        npz_dir=test_npz_dir,
        image_filenames=test_files,
        native_img_dir=native_img_dir,
        transform=test_transform,
        margin_ratio=margin_ratio,
        target_size=target_size,
        sigma=sigma,
        jitter=None,
        is_train=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader
