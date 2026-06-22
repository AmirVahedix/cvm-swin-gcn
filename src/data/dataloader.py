import os
from src.data.dataset import CVMDataset
from src.data.transforms import get_transforms
from torch.utils.data import DataLoader


TEST_IMG_DIR = "dataset/test/images"
TEST_NPZ_DIR = "dataset/test/labels"


def get_dataloaders(
    train_img_dir,
    train_npz_dir,
    val_img_dir,
    val_npz_dir,
    batch_size=8,
    img_size=640,
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
        val_img_dir, val_npz_dir, val_files, transform=val_transform, img_size=img_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(
    test_img_dir=TEST_IMG_DIR,
    test_npz_dir=TEST_NPZ_DIR,
    batch_size=8,
    img_size=640,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader
