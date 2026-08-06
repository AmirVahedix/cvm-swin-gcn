import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2


def find_corresponding_image(base_name, images_dir):
    """
    Attempts to find the corresponding image file regardless of extension.
    """
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    for ext in extensions:
        img_path = os.path.join(images_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def visualize_gcn_coords(npz_dir, images_dir, num_samples=4):
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))

    if len(npz_files) < num_samples:
        print(f"Not enough .npz files. Found {len(npz_files)}, need {num_samples}.")
        num_samples = len(npz_files)
        if num_samples == 0:
            return

    # Pick random NPZ files
    selected_npz = random.sample(npz_files, num_samples)

    # Setup matplotlib grid
    rows = int(np.ceil(num_samples / 2))
    cols = 2 if num_samples > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(14, 7 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    # Anatomy groupings based on your 13 landmark structure
    # C2: 0-2 | C3: 3-7 | C4: 8-12
    groups = [
        {"range": range(0, 3), "color": "red", "label": "C2"},
        {"range": range(3, 8), "color": "lime", "label": "C3"},
        {"range": range(8, 13), "color": "cyan", "label": "C4"},
    ]

    for idx, npz_path in enumerate(selected_npz):
        base_name = os.path.splitext(os.path.basename(npz_path))[0]

        # 1. Find and load the original image
        img_path = find_corresponding_image(base_name, images_dir)
        if not img_path:
            axes[idx].set_title(f"Image not found for\n{base_name}")
            axes[idx].axis("off")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 2. Load the GCN coordinates
        with np.load(npz_path) as data:
            if "coords" not in data:
                axes[idx].set_title(f"'coords' key missing in\n{base_name}")
                axes[idx].axis("off")
                continue
            coords = data["coords"]  # Shape: (13, 2), values [0, 1]

        # 3. Plot original image
        axes[idx].imshow(img)

        # 4. Plot the GCN coordinates
        for group in groups:
            for i in group["range"]:
                x_norm, y_norm = coords[i]

                # Skip missing landmarks
                if x_norm == -1.0 or y_norm == -1.0:
                    continue

                # Un-normalize back to absolute pixels
                abs_x = x_norm * w
                abs_y = y_norm * h

                # Plot the point
                axes[idx].plot(
                    abs_x,
                    abs_y,
                    marker="o",
                    color=group["color"],
                    markersize=6,
                    markeredgecolor="black",
                )

        axes[idx].set_title(f"GCN Coordinates: {base_name}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Update these paths to match your directories
    NPZ_DIRECTORY = "data/labels"  
    IMAGES_DIRECTORY = "data/images"

    visualize_gcn_coords(NPZ_DIRECTORY, IMAGES_DIRECTORY, num_samples=2)
