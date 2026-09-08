import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_transforms(img_size=1024):
    """
    Defines the training and validation augmentation pipelines.
    Note: Horizontal flip is excluded as lateral cephalometric images
    have a strict anatomical orientation.

    Augmentations included:
    - Exposure & Contrast: CLAHE, RandomGamma, RandomBrightnessContrast (simulating X-ray exposure variations)
    - Position & Tilt: Affine, Perspective (simulating patient head tilt/rotation)
    - Anatomical Deformations: ElasticTransform (simulating non-rigid anatomical variation)
    - Image Quality: GaussianBlur
    """
    train_transform = A.Compose(
        [
            # 1. Exposure & Contrast Enhancement
            A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.6),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ),
            # 2. Geometric & Positioning Transformations (Patient head rotation / tilt)
            A.Affine(
                translate_percent=0.05,
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.6,
            ),
            A.Perspective(
                scale=(0.01, 0.04),
                keep_size=True,
                p=0.3,
            ),
            # 3. Non-rigid Elastic Deformations (Anatomical shape variation)
            A.ElasticTransform(
                alpha=1,
                sigma=30,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.3,
            ),
            # 4. Noise / Blur
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # 5. Normalization & PyTorch Tensor Conversion
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,  # Crucial for GCN: maintains the 13-node sequence
        ),
    )

    val_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    return train_transform, val_transform

