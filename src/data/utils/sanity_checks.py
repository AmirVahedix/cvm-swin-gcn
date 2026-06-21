import os
import glob
import numpy as np
from tqdm import tqdm


def run_sanity_checks(npz_dir):
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))

    if not npz_files:
        print(f"No .npz files found in {npz_dir}")
        return

    print(f"Found {len(npz_files)} .npz files. Running checks...\n")

    # --- Tracking Variables ---
    # Heatmaps
    unique_heatmap_shapes = set()
    unique_heatmap_dtypes = set()
    missing_heatmaps_key = 0

    # GCN Coordinates
    unique_coords_shapes = set()
    unique_coords_dtypes = set()
    missing_coords_key = 0
    total_missing_landmarks = 0

    # Track min/max of VALID coordinates to ensure they are in [0, 1]
    global_coords_min = float("inf")
    global_coords_max = float("-inf")

    example_data_heatmaps = None
    example_data_coords = None
    example_file = None

    for file_path in tqdm(npz_files, desc="Checking .npz files"):
        try:
            with np.load(file_path) as data:
                # 1. Check Heatmaps
                if "heatmaps" not in data:
                    missing_heatmaps_key += 1
                else:
                    heatmaps = data["heatmaps"]
                    unique_heatmap_shapes.add(heatmaps.shape)
                    unique_heatmap_dtypes.add(heatmaps.dtype)

                # 2. Check GCN Coords
                if "coords" not in data:
                    missing_coords_key += 1
                else:
                    coords = data["coords"]
                    unique_coords_shapes.add(coords.shape)
                    unique_coords_dtypes.add(coords.dtype)

                    # Count missing landmarks (where x == -1.0)
                    total_missing_landmarks += np.sum(coords[:, 0] == -1.0)

                    # Track valid coordinate ranges (ignoring the -1.0 missing markers)
                    valid_coords = coords[coords[:, 0] != -1.0]
                    if valid_coords.size > 0:
                        global_coords_min = min(global_coords_min, np.min(valid_coords))
                        global_coords_max = max(global_coords_max, np.max(valid_coords))

                # Grab the first valid file as our example
                if example_file is None and "heatmaps" in data and "coords" in data:
                    example_data_heatmaps = heatmaps
                    example_data_coords = coords
                    example_file = os.path.basename(file_path)

        except Exception as e:
            tqdm.write(f"Error reading {file_path}: {e}")

    # --- Print Results ---
    print("\n" + "=" * 40)
    print(" SANITY CHECK RESULTS")
    print("=" * 40)
    print(f"Total files checked: {len(npz_files)}")

    print("\n--- Heatmaps (Swin) ---")
    print(f"Files missing key:  {missing_heatmaps_key}")
    print(f"Unique shapes:      {unique_heatmap_shapes}")
    print(f"Unique data types:  {unique_heatmap_dtypes}")
    if len(unique_heatmap_shapes) > 1:
        print("  -> WARNING: Inconsistent heatmap dimensions detected!")

    print("\n--- GCN Coordinates ---")
    print(f"Files missing key:  {missing_coords_key}")
    print(f"Unique shapes:      {unique_coords_shapes}")
    print(f"Unique data types:  {unique_coords_dtypes}")
    print(f"Missing landmarks:  {total_missing_landmarks} (marked as -1.0)")

    if global_coords_min != float("inf"):
        print(f"Valid value range:  [{global_coords_min:.4f}, {global_coords_max:.4f}]")
        if global_coords_min < 0.0 or global_coords_max > 1.0:
            print(
                "  -> WARNING: Valid coordinates fall outside the expected [0, 1] normalized range!"
            )
    else:
        print("Valid value range:  N/A (No valid coordinates found)")

    if len(unique_coords_shapes) > 1:
        print("  -> WARNING: Inconsistent coordinate array shapes detected!")

    # --- Example Breakdown ---
    if example_file is not None:
        print("\n" + "-" * 40)
        print(f" EXAMPLE BREAKDOWN: {example_file}")
        print("-" * 40)

        print(f"[Heatmaps]")
        print(f"  Shape: {example_data_heatmaps.shape}")
        print(f"  Type:  {example_data_heatmaps.dtype}")
        print(
            f"  Range: [{np.min(example_data_heatmaps):.4f}, {np.max(example_data_heatmaps):.4f}]"
        )
        non_zero = np.count_nonzero(example_data_heatmaps)
        print(f"  Active Pixels: {non_zero} / {example_data_heatmaps.size}")

        print(f"\n[Coordinates]")
        print(f"  Shape: {example_data_coords.shape}")
        print(f"  Type:  {example_data_coords.dtype}")
        print("  Values (first 3):")
        for i in range(min(3, len(example_data_coords))):
            print(f"    Point {i + 1}: {example_data_coords[i]}")


if __name__ == "__main__":
    NPZ_DIRECTORY = "data/labels"
    run_sanity_checks(NPZ_DIRECTORY)
