#!/usr/bin/env python3
"""
Utility script to shrink heavy PyTorch training checkpoints.
Strips out optimizer_state_dict (~900MB) and training metadata to keep only model weights.
Optionally converts to FP16 for even greater storage savings (~210MB total).

Usage:
    # Strip optimizer state (reduces ~1.5GB -> ~420MB)
    python scripts/shrink_checkpoint.py --input artifacts/best.pth --inplace

    # Strip optimizer state and save to a new file
    python scripts/shrink_checkpoint.py --input artifacts/best.pth --output artifacts/best_weights.pth

    # Convert to FP16 weights (reduces ~1.5GB -> ~210MB)
    python scripts/shrink_checkpoint.py --input artifacts/best.pth --output artifacts/best_fp16.pth --fp16
"""

import argparse
import os
import sys
from pathlib import Path
import torch


def shrink_checkpoint(input_path: str, output_path: str, fp16: bool = False) -> None:
    if not os.path.exists(input_path):
        print(f"❌ Error: Input checkpoint '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    initial_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"--> Loading checkpoint '{input_path}' ({initial_size_mb:.2f} MB)...")

    checkpoint = torch.load(input_path, map_location="cpu")

    # Extract model state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("    Found 'model_state_dict' inside checkpoint dictionary.")
        if "optimizer_state_dict" in checkpoint:
            print("    Stripping 'optimizer_state_dict' (~60-70% of file size)...")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        print("    Checkpoint is already a state_dict dictionary.")
    else:
        # Full module object
        state_dict = checkpoint.state_dict()
        print("    Extracted state_dict from full PyTorch module object.")

    # Remove static non-parameter buffers if present
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.endswith("grid_x") and not k.endswith("grid_y")
    }

    # Optional FP16 casting
    if fp16:
        print("    Converting floating point tensors to FP16 (half precision)...")
        state_dict = {
            k: (v.half() if torch.is_floating_point(v) else v)
            for k, v in state_dict.items()
        }

    # Save to temp file first if in-place overwrite
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_save_path = target_path.with_suffix(".tmp.pth") if target_path == Path(input_path) else target_path

    print(f"--> Saving stripped weights to '{temp_save_path}'...")
    torch.save(state_dict, str(temp_save_path))

    if temp_save_path != target_path:
        temp_save_path.replace(target_path)

    final_size_mb = os.path.getsize(str(target_path)) / (1024 * 1024)
    reduction_pct = (1.0 - (final_size_mb / initial_size_mb)) * 100.0 if initial_size_mb > 0 else 0.0

    print(f"✅ Successfully created stripped weights file: '{target_path}'")
    print(f"    Original Size: {initial_size_mb:.2f} MB")
    print(f"    New Size:      {final_size_mb:.2f} MB")
    print(f"    Saved Space:   {initial_size_mb - final_size_mb:.2f} MB ({reduction_pct:.1f}% reduction)")


def main():
    parser = argparse.ArgumentParser(description="Shrink PyTorch checkpoints by stripping optimizer state.")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="./artifacts/best.pth",
        help="Path to the input checkpoint file (default: ./artifacts/best.pth)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output weights file. If not specified, defaults to replacing input if --inplace is set, or ./artifacts/best_weights.pth",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input checkpoint file in-place with stripped weights.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert weights to FP16 (half-precision), cutting size by another ~50% (~210MB final size).",
    )

    args = parser.parse_args()

    if args.inplace:
        output_path = args.input
    elif args.output is not None:
        output_path = args.output
    else:
        output_path = str(Path(args.input).with_name("best_weights.pth"))

    shrink_checkpoint(input_path=args.input, output_path=output_path, fp16=args.fp16)


if __name__ == "__main__":
    main()
