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

        heatmaps = data["heatmaps"]
        coords = data["coords"]

        # 3. Prepare data structures for Albumentations
        heatmaps_hwc = np.transpose(heatmaps, (1, 2, 0))
        coords_pixels = coords * self.img_size

        # 4. Apply synchronized transformations
        if self.transform:
            augmented = self.transform(
                image=image, mask=heatmaps_hwc, keypoints=coords_pixels
            )
            image = augmented["image"]
            heatmaps_tensor = augmented["mask"]
            transformed_coords = np.array(augmented["keypoints"])
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
            heatmaps_tensor = torch.from_numpy(heatmaps).float()
            transformed_coords = coords_pixels

        # 5. Re-normalize coordinates back to [0, 1] for network safety
        transformed_coords = np.clip(transformed_coords, 0, self.img_size - 1)
        coords_normalized = transformed_coords / self.img_size
        coords_tensor = torch.tensor(coords_normalized, dtype=torch.float32)

        return {
            "image": image,
            "heatmaps": heatmaps_tensor,
            "coords": coords_tensor,
        }


if __name__ == "__main__":
    # 1. Define base paths for your pre-split data
    # Adjust these paths to match your actual directory structure
    TRAIN_IMG_DIR = "path/to/dataset/images/train"
    TRAIN_NPZ_DIR = "path/to/dataset/labels/train"

    VAL_IMG_DIR = "path/to/dataset/images/val"
    VAL_NPZ_DIR = "path/to/dataset/labels/val"

    TEST_IMG_DIR = "path/to/dataset/images/test"
    TEST_NPZ_DIR = "path/to/dataset/labels/test"

    # 2. Extract filenames directly from the respective folders
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

    # 3. Initialize Pipelines
    train_transform, val_transform = get_transforms(img_size=640)

    # 4. Instantiate separate Datasets, passing the specific directories and files
    train_dataset = CVMDataset(
        TRAIN_IMG_DIR, TRAIN_NPZ_DIR, train_files, transform=train_transform
    )
    val_dataset = CVMDataset(
        VAL_IMG_DIR, VAL_NPZ_DIR, val_files, transform=val_transform
    )
    test_dataset = CVMDataset(
        TEST_IMG_DIR, TEST_NPZ_DIR, test_files, transform=val_transform
    )

    # 5. Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )

    # --- SANITY CHECK ---
    print("\nVerifying Train Loader...")
    for batch in train_loader:
        print("Batch verification successful:")
        print(f" -> Images batch tensor shape:    {batch['image'].shape}")
        print(f" -> Heatmaps batch tensor shape:  {batch['heatmaps'].shape}")
        print(f" -> Coordinates batch tensor shape: {batch['coords'].shape}")
        break
