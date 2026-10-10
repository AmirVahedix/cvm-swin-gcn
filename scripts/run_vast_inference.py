#!/usr/bin/env python3
"""
Automated Vast.ai Cloud GPU Inference Runner for CVM X-ray Datasets.

Workflow:
1. Archives local input images into a zip package.
2. Provisions a cloud GPU instance on Vast.ai (e.g., RTX 3090 / 4090) or connects to an existing instance.
3. Uploads images, model weights, and the batch inference script.
4. Executes CUDA-accelerated batch inference with live progress streaming.
5. Automatically downloads results (CSV, JSON, and annotated visualization images) to your local machine.
6. Automatically destroys the remote instance upon completion to prevent unwanted cloud charges.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Ensure project root in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.vast_runner import (
    COLOR_BOLD,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_RESET,
    COLOR_YELLOW,
    VastAPIClient,
    get_api_key,
    get_stored_instance_id,
    set_stored_instance_id,
    clear_stored_instance_id,
    wait_for_instance_ready,
    wait_for_ssh_ready,
)


def run_cmd(cmd: list, check: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=text, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CVM Landmark Detection & Staging Inference on Vast.ai Cloud GPU")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Path to local folder containing images")
    parser.add_argument(
        "--output-dir", "-o", type=str, default="vast_inference_results", help="Local directory to save results"
    )
    parser.add_argument("--gpu", type=str, default="3090", help="GPU filter for Vast.ai (e.g. '3090', '4090', 'RTX')")
    parser.add_argument("--max-price", type=float, default=0.30, help="Maximum price in $/hr (default: 0.30)")
    parser.add_argument("--instance-id", type=int, default=None, help="Use an existing running Vast instance ID")
    parser.add_argument("--save-visualizations", action="store_true", default=False, help="Save annotated visualization images")
    parser.add_argument("--no-destroy", action="store_true", default=False, help="Keep instance running after inference")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"{COLOR_RED}❌ Error: Input directory '{input_dir}' does not exist.{COLOR_RESET}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = get_api_key()
    if not api_key:
        print(f"{COLOR_RED}❌ Error: Vast.ai API key not found in .env or environment.{COLOR_RESET}", file=sys.stderr)
        sys.exit(1)

    client = VastAPIClient(api_key)
    user_info = client.get_user()
    credit = float(user_info.get("credit", 0.0))
    print(f"{COLOR_CYAN}👤 Vast.ai Account: {user_info.get('username', 'User')} | Balance: ${credit:.2f} USD{COLOR_RESET}")

    if credit < 0.20:
        print(f"{COLOR_YELLOW}⚠️ Warning: Low Vast.ai credit (${credit:.2f}). Please top up if needed.{COLOR_RESET}")

    instance_id = get_stored_instance_id(args.instance_id)

    # 1. Zip input images for fast transfer
    print(f"\n{COLOR_CYAN}📦 Packaging input images from '{input_dir}'...{COLOR_RESET}")
    temp_zip = Path(tempfile.gettempdir()) / f"cvm_inference_images_{int(time.time())}.zip"
    shutil.make_archive(str(temp_zip.with_suffix("")), "zip", str(input_dir))
    zip_size_mb = temp_zip.stat().st_size / (1024 * 1024)
    print(f"✅ Images packaged into '{temp_zip.name}' ({zip_size_mb:.1f} MB).")

    try:
        # 2. Provision or reuse instance
        if not instance_id:
            print(f"\n{COLOR_CYAN}🔍 Searching for available {args.gpu} offers under ${args.max_price}/hr...{COLOR_RESET}")
            offers = client.search_offers(gpu_name=args.gpu, max_price=args.max_price, min_reliability=0.90, order_by="price")
            if not offers:
                print(f"{COLOR_RED}❌ No matching offers found for GPU '{args.gpu}' under ${args.max_price}/hr.{COLOR_RESET}")
                sys.exit(1)

            best_offer = offers[0]
            offer_id = best_offer["id"]
            price = best_offer.get("_computed_price", 0.0)
            gpu_name = best_offer.get("gpu_name", "GPU")
            print(f"🚀 Renting offer {offer_id}: {gpu_name} at ${price:.3f}/hr...")

            instance_id = client.create_instance(offer_id=offer_id, image="pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime", disk_gb=25.0)
            set_stored_instance_id(instance_id)
            print(f"✅ Provisioned Vast Instance ID: {COLOR_BOLD}{instance_id}{COLOR_RESET}")

        # 3. Wait for boot and SSH
        inst = wait_for_instance_ready(client, instance_id)
        ssh_host = inst.get("ssh_host") or inst.get("public_ipaddr")
        ssh_port = int(inst.get("ssh_port"))
        wait_for_ssh_ready(ssh_host, ssh_port)

        ssh_base = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-p", str(ssh_port), f"root@{ssh_host}"]
        scp_base = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-P", str(ssh_port)]

        print(f"\n{COLOR_CYAN}🚚 Deploying code and images to remote instance...{COLOR_RESET}")
        run_cmd(ssh_base + ["mkdir -p /workspace/remote_infer/images"])

        # Upload images zip
        run_cmd(scp_base + [str(temp_zip), f"root@{ssh_host}:/workspace/remote_infer/images.zip"])
        run_cmd(ssh_base + ["unzip -q -o /workspace/remote_infer/images.zip -d /workspace/remote_infer/images/"])

        # Locate scripts and weights
        symcvm_dir = REPO_ROOT.parent / "symcvm"
        batch_script = symcvm_dir / "batch_inference.py"
        src_dir = symcvm_dir / "src"
        weights_file = symcvm_dir / "model" / "weights.pth"
        if not weights_file.exists():
            weights_file = REPO_ROOT / "artifacts" / "best.pth"

        # Upload scripts and model
        run_cmd(scp_base + [str(batch_script), f"root@{ssh_host}:/workspace/remote_infer/batch_inference.py"])
        run_cmd(scp_base + ["-r", str(src_dir), f"root@{ssh_host}:/workspace/remote_infer/"])
        run_cmd(ssh_base + ["mkdir -p /workspace/remote_infer/model"])
        run_cmd(scp_base + [str(weights_file), f"root@{ssh_host}:/workspace/remote_infer/model/weights.pth"])

        # Install minimal python dependencies on remote
        print(f"\n{COLOR_CYAN}📦 Installing remote inference dependencies...{COLOR_RESET}")
        run_cmd(ssh_base + ["pip install -q timm albumentations pillow tqdm"])

        # 4. Run Batch Inference
        print(f"\n{COLOR_GREEN}⚡ Executing CUDA Batch Inference on {gpu_name}...{COLOR_RESET}")
        vis_flag = "--save-visualizations" if args.save_visualizations else ""
        remote_cmd = f"cd /workspace/remote_infer && python3 batch_inference.py -i images -o results --device cuda {vis_flag}"

        proc = subprocess.Popen(ssh_base + [remote_cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"Remote inference exited with code {proc.returncode}")

        # 5. Download results
        print(f"\n{COLOR_CYAN}📥 Downloading results to '{output_dir}'...{COLOR_RESET}")
        run_cmd(scp_base + ["-r", f"root@{ssh_host}:/workspace/remote_infer/results/*", str(output_dir)])
        print(f"{COLOR_GREEN}✅ Successfully retrieved all predictions and metrics to '{output_dir}'.{COLOR_RESET}")

    finally:
        # Cleanup local zip
        if temp_zip.exists():
            temp_zip.unlink()

        # Destroy instance
        if not args.no_destroy and instance_id:
            print(f"\n{COLOR_YELLOW}🛑 Auto-destroying Vast.ai instance {instance_id} to halt billing...{COLOR_RESET}")
            try:
                client.destroy_instance(instance_id)
                clear_stored_instance_id()
                print(f"{COLOR_GREEN}✅ Instance {instance_id} destroyed successfully.{COLOR_RESET}")
            except Exception as e:
                print(f"⚠️ Could not destroy instance {instance_id}: {e}")


if __name__ == "__main__":
    main()
