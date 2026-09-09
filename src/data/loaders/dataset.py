import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def generate_gaussian_heatmaps(coords_px: np.ndarray, img_size: int = 640, sigma: float = 4.0) -> torch.Tensor:
    """
    Generates pristine [NUM_LANDMARKS, H, W] Gaussian heatmaps directly from
    ground-truth landmark pixel coordinates.
    """
    num_lms = len(coords_px)
    heatmaps = np.zeros((num_lms, img_size, img_size), dtype=np.float32)
    radius = int(3 * sigma + 1)
    two_sigma_sq = 2.0 * (sigma ** 2)

    for i in range(num_lms):
        cx, cy = coords_px[i]
        if cx < 0 or cy < 0 or np.isnan(cx) or np.isnan(cy):
            continue

        # Localized Gaussian calculation around peak for maximum speed & precision
        x0 = int(max(0, np.floor(cx - radius)))
        x1 = int(min(img_size, np.ceil(cx + radius + 1)))
        y0 = int(max(0, np.floor(cy - radius)))
        y1 = int(min(img_size, np.ceil(cy + radius + 1)))

        if x1 <= x0 or y1 <= y0:
            continue

        gx = np.arange(x0, x1, dtype=np.float32)
        gy = np.arange(y0, y1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(gx, gy)

        gaussian_patch = np.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / two_sigma_sq)
        heatmaps[i, y0:y1, x0:x1] = np.maximum(heatmaps[i, y0:y1, x0:x1], gaussian_patch)

    return torch.from_numpy(heatmaps).float()


class CVMDataset(Dataset):
    def __init__(
        self, image_dir, npz_dir, image_filenames, transform=None, img_size=640, sigma=4.0
    ):
        """
        Args:
            image_dir (str): Path to directory containing images.
            npz_dir (str): Path to directory containing .npz files.
            image_filenames (list): List of image filenames allocated for this split.
            transform (albumentations.Compose): Spatial and pixel augmentations.
            img_size (int): Expected target pixel size (1024).
            sigma (float): Gaussian heatmap sigma in pixels.
        """
        self.image_dir = image_dir
        self.npz_dir = npz_dir
        self.image_filenames = image_filenames
        self.transform = transform
        self.img_size = img_size
        self.sigma = sigma

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
        coords = data["coords"]  # [13, 2] in normalized scale [0, 1]

        # 3. Scale coordinates to pixel units for augmentation
        coords_pixels = (coords * self.img_size).astype(np.float32)

        # 4. Apply synchronized geometric and color transformations to Image + Keypoints
        if self.transform:
            augmented = self.transform(
                image=image, keypoints=coords_pixels
            )
            image_tensor = augmented["image"]
            aug_kp = augmented["keypoints"]
            if len(aug_kp) == len(coords_pixels):
                transformed_coords = np.array(aug_kp, dtype=np.float32)
            else:
                transformed_coords = coords_pixels.copy().astype(np.float32)
        else:
            image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
            transformed_coords = coords_pixels.copy().astype(np.float32)

        # Ensure image tensor is float32
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(np.transpose(image_tensor, (2, 0, 1))).float()
        else:
            image_tensor = image_tensor.float()

        # 5. Clip coordinates within valid image bounds
        valid_mask = (coords[:, 0] >= 0) & (coords[:, 1] >= 0)
        for i in range(len(transformed_coords)):
            if valid_mask[i]:
                transformed_coords[i, 0] = np.clip(transformed_coords[i, 0], 0.0, float(self.img_size - 1))
                transformed_coords[i, 1] = np.clip(transformed_coords[i, 1], 0.0, float(self.img_size - 1))
            else:
                transformed_coords[i] = [-1.0, -1.0]

        # 6. Dynamically generate mathematically pristine Gaussian heatmaps
        heatmaps_tensor = generate_gaussian_heatmaps(
            transformed_coords, img_size=self.img_size, sigma=self.sigma
        )

        # 7. Normalize coordinates to [0, 1] range
        coords_normalized = np.full_like(transformed_coords, -1.0)
        for i in range(len(transformed_coords)):
            if transformed_coords[i, 0] >= 0:
                coords_normalized[i] = transformed_coords[i] / float(self.img_size)

        coords_tensor = torch.tensor(coords_normalized, dtype=torch.float32)

        return {
            "image": image_tensor,
            "heatmaps": heatmaps_tensor,
            "coords": coords_tensor,
            "filename": img_name,
        }

