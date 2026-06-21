import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_transforms(img_size=640):
    """
    Defines the training and validation augmentation pipelines.
    Note: Horizontal flip is excluded here as lateral cephalometric images
    have a strict anatomical orientation. Shift, scale, and rotate are preferred.
    """
    train_transform = A.Compose(
        [  # type: ignore[arg-type]
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,  # type: ignore
                p=0.6,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),  # Converts HWC to CHW for images and masks automatically
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,  # Crucial for GCN: maintains the 13-node sequence
        ),
    )

    val_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    return train_transform, val_transform
