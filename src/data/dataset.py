import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.transforms import get_transforms


class CVMDataset(Dataset):
    def __init__(
        self, image_dir, npz_dir, image_filenames, transform=None, img_size=640
    ):
        """
        Args:
            image_dir (str): Path to directory containing images (640x640).
            npz_dir (str): Path to directory containing .npz files.
            image_filenames (list): List of image filenames allocated for this split.
            transform (albumentations.Compose): Spatial and pixel augmentations.
            img_size (int): Expected target pixel size (640).
        """
        self.image_dir = image_dir
        self.npz_dir = npz_dir
        self.image_filenames = image_filenames
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        base_name = os.path.splitext(img_name)[0]

        # 1. Load Image
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore

        # 2. Load pre-computed targets from .npz
        npz_path = os.path.join(self.npz_dir, f"{base_name}.npz")
        data = np.load(npz_path)

        # Shape expected: [13, 640, 640]
        heatmaps = data["heatmaps"]
        # Shape expected: [13, 2] (normalized [0, 1])
        coords = data["coords"]

        # 3. Prepare data structures for Albumentations
        # Transpose heatmaps from CHW to HWC
        heatmaps_hwc = np.transpose(heatmaps, (1, 2, 0))

        # Convert normalized coordinates to absolute pixel values
        coords_pixels = coords * self.img_size

        # 4. Apply synchronized transformations
        if self.transform:
            augmented = self.transform(
                image=image, mask=heatmaps_hwc, keypoints=coords_pixels
            )
            image = augmented["image"]
            heatmaps_tensor = augmented[
                "mask"
            ]  # ToTensorV2 converts this back to [13, 640, 640]
            transformed_coords = np.array(augmented["keypoints"])
        else:
            # Fallback manual conversion if no transform provided
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
            heatmaps_tensor = torch.from_numpy(heatmaps).float()
            transformed_coords = coords_pixels

        # 5. Re-normalize coordinates back to [0, 1] for GCN network safety
        # Handle cases where keypoints might be pushed out of borders during heavy scaling/shifting
        transformed_coords = np.clip(transformed_coords, 0, self.img_size - 1)
        coords_normalized = transformed_coords / self.img_size
        coords_tensor = torch.tensor(coords_normalized, dtype=torch.float32)

        return {
            "image": image,  # torch.Tensor [3, 640, 640]
            "heatmaps": heatmaps_tensor,  # torch.Tensor [13, 640, 640]
            "coords": coords_tensor,  # torch.Tensor [13, 2]
        }


if __name__ == "__main__":
    # Mock setups for paths and file lists
    IMAGE_DIR = "path/to/640x640_images"
    NPZ_DIR = "path/to/npz_labels"

    # Example train/val split using files present in directory
    all_files = [
        f for f in os.listdir(IMAGE_DIR) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # Initialize Pipelines
    train_transform, val_transform = get_transforms(img_size=640)

    train_dataset = CVMDataset(
        image_dir=IMAGE_DIR,
        npz_dir=NPZ_DIR,
        image_filenames=train_files,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # --- SANITY CHECK ---
    print(f"Total training samples: {len(train_dataset)}")
    for batch in train_loader:
        print("Batch verification successful:")
        print(
            f" -> Images batch tensor shape:    {batch['image'].shape}"
        )  # Expected: [8, 3, 640, 640]
        print(
            f" -> Heatmaps batch tensor shape:  {batch['heatmaps'].shape}"
        )  # Expected: [8, 13, 640, 640]
        print(
            f" -> Coordinates batch tensor shape: {batch['coords'].shape}"
        )  # Expected: [8, 13, 2]
        break
