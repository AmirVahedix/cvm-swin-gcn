import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


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
        # Cast to float32 to prevent OpenCV assertion errors during augmentation
        heatmaps_hwc = np.transpose(heatmaps, (1, 2, 0)).astype(np.float32)
        coords_pixels = (coords * self.img_size).astype(np.float32)

        # 4. Apply synchronized transformations
        if self.transform:
            augmented = self.transform(
                image=image, mask=heatmaps_hwc, keypoints=coords_pixels
            )
            image = augmented["image"]
            heatmaps_tensor = augmented["mask"]
            aug_kp = augmented["keypoints"]
            if len(aug_kp) == len(coords_pixels):
                transformed_coords = np.array(aug_kp, dtype=np.float32)
            else:
                transformed_coords = coords_pixels.copy().astype(np.float32)
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
            heatmaps_tensor = torch.from_numpy(heatmaps).float()
            transformed_coords = coords_pixels.copy().astype(np.float32)

        # 5. Re-normalize coordinates back to [0, 1] for network safety
        transformed_coords = np.clip(transformed_coords, 0, self.img_size - 1)
        coords_normalized = transformed_coords / self.img_size
        coords_tensor = torch.tensor(coords_normalized, dtype=torch.float32)

        return {
            "image": image,
            "heatmaps": heatmaps_tensor,
            "coords": coords_tensor,
        }
