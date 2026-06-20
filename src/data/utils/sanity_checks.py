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

    unique_shapes = set()
    unique_dtypes = set()
    missing_keys = 0

    # Store one example for detailed printing
    example_data = None
    example_file = None

    # Wrap the iterable with tqdm for a visual progress bar
    for file_path in tqdm(npz_files, desc="Checking .npz files"):
        try:
            with np.load(file_path) as data:
                if "heatmaps" not in data:
                    missing_keys += 1
                    continue

                heatmaps = data["heatmaps"]
                unique_shapes.add(heatmaps.shape)
                unique_dtypes.add(heatmaps.dtype)

                # Grab the first valid file as our example
                if example_data is None:
                    example_data = heatmaps
                    example_file = os.path.basename(file_path)

        except Exception as e:
            # Use tqdm.write instead of print to prevent the progress bar from glitching
            tqdm.write(f"Error reading {file_path}: {e}")

    # --- Print Results ---
    print("\n--- Sanity Check Results ---")
    print(f"Total files checked: {len(npz_files)}")
    print(f"Files missing 'heatmaps' array: {missing_keys}")
    print(f"Unique array shapes found: {unique_shapes}")
    print(f"Unique data types found: {unique_dtypes}")

    if len(unique_shapes) > 1:
        print("\nWARNING: Inconsistent dimensions detected across your dataset!")
    elif len(unique_shapes) == 1:
        print("\nSUCCESS: All heatmap arrays have consistent dimensions.")

    if example_data is not None:
        print(f"\n--- Data Example ({example_file}) ---")
        print(f"Shape: {example_data.shape}")
        print(f"Data Type: {example_data.dtype}")
        print(f"Min Value: {np.min(example_data)}")
        print(f"Max Value: {np.max(example_data)}")
        # Check if the array is entirely empty/zeros
        non_zero = np.count_nonzero(example_data)
        print(f"Non-zero elements: {non_zero} (out of {example_data.size})")


if __name__ == "__main__":
    # Update this path to match your output directory
    NPZ_DIRECTORY = "data/heatmaps"
    run_sanity_checks(NPZ_DIRECTORY)
