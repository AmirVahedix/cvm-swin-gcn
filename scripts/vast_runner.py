#!/usr/bin/env python3
"""
Vast.ai Automation & Instance Management Runner
Handles searching, formatting (by price, RAM, etc.), provisioning, provisioning setup,
and executing training pipelines on Vast.ai GPU cloud instances.
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Dict, List, Any, Optional

VAST_API_BASE = "https://console.vast.ai/api/v0"

# ANSI Colors
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_GREEN = "\033[32m"
COLOR_BLUE = "\033[34m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_CYAN = "\033[36m"
COLOR_DIM = "\033[2m"


def load_env_file(filepath: Path) -> Dict[str, str]:
    """Parse a .env file into a dictionary without external dependencies."""
    env_vars = {}
    if not filepath.is_file():
        return env_vars
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                env_vars[k] = v
    except Exception:
        pass
    return env_vars


def get_api_key(cli_key: Optional[str] = None) -> str:
    """Resolve Vast.ai API Key from CLI arg, environment, or .env files."""
    if cli_key:
        return cli_key.strip()

    # 1. Shell environment
    if os.getenv("VAST_API_KEY"):
        return os.getenv("VAST_API_KEY").strip()

    # 2. Check local .env files
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    for candidate in [cwd / ".env", script_dir / ".env", repo_root / ".env"]:
        env_dict = load_env_file(candidate)
        if "VAST_API_KEY" in env_dict and env_dict["VAST_API_KEY"]:
            return env_dict["VAST_API_KEY"].strip()

    # 3. Check ~/.vast_api_key
    home_key = Path.home() / ".vast_api_key"
    if home_key.is_file():
        try:
            val = home_key.read_text(encoding="utf-8").strip()
            if val:
                return val
        except Exception:
            pass

    return ""


class VastAPIClient:
    """Lightweight Vast.ai REST API client with zero third-party dependencies."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "Vast.ai API key not found. Please provide --api-key, set VAST_API_KEY in .env, "
                "or export VAST_API_KEY in your environment."
            )
        self.api_key = api_key

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        params["api_key"] = self.api_key
        url = f"{VAST_API_BASE}/{endpoint.lstrip('/')}"
        query_string = urllib.parse.urlencode(
            {
                k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                for k, v in params.items()
            }
        )
        if query_string:
            url = f"{url}?{query_string}"

        req_data = None
        headers = {"Accept": "application/json"}
        if data is not None:
            req_data = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url, data=req_data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            try:
                err_json = json.loads(err_body)
                msg = (
                    err_json.get("msg")
                    or err_json.get("error")
                    or err_json.get("message")
                    or err_body
                )
            except Exception:
                msg = err_body
            raise RuntimeError(f"Vast.ai API error ({e.code}): {msg}")
        except Exception as e:
            raise RuntimeError(f"Network error communicating with Vast.ai: {e}")

    def get_user(self) -> Dict[str, Any]:
        """Fetch current user account and balance information."""
        return self._request("users/current/")

    def search_offers(
        self,
        gpu_name: Optional[str] = None,
        num_gpus: int = 1,
        max_price: Optional[float] = None,
        min_ram_gb: Optional[float] = None,
        min_vram_gb: Optional[float] = None,
        min_disk_gb: Optional[float] = None,
        min_cuda: Optional[float] = None,
        verified_only: bool = True,
        order_by: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search available machine offers with flexible criteria."""
        query_dict: Dict[str, Any] = {
            "verified": {"eq": verified_only},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "num_gpus": {"eq": num_gpus},
        }

        if min_disk_gb is not None:
            query_dict["disk_space"] = {"gte": float(min_disk_gb)}
        if min_ram_gb is not None:
            # Vast stores cpu_ram in MB
            query_dict["cpu_ram"] = {"gte": float(min_ram_gb) * 1024}
        if min_vram_gb is not None:
            # Vast stores gpu_ram in MB
            query_dict["gpu_ram"] = {"gte": float(min_vram_gb) * 1024}
        if max_price is not None:
            query_dict["dph_total"] = {"lte": float(max_price)}
        if min_cuda is not None:
            query_dict["cuda_max_good"] = {"gte": float(min_cuda)}

        res = self._request("bundles/", method="GET", params={"q": query_dict})
        offers = res.get("offers", []) if isinstance(res, dict) else []

        # Filter by GPU name substring/case-insensitive if specified
        if gpu_name:
            gpu_query = gpu_name.lower().strip()
            offers = [
                o
                for o in offers
                if gpu_query in str(o.get("gpu_name", "")).lower()
            ]

        # Sorting
        if order_by == "price" or order_by == "dph":
            offers.sort(key=lambda o: o.get("dph_total", 999999))
        elif order_by == "ram":
            offers.sort(key=lambda o: o.get("cpu_ram", 0), reverse=True)
        elif order_by == "vram" or order_by == "gpu_ram":
            offers.sort(key=lambda o: o.get("gpu_ram", 0), reverse=True)
        elif order_by == "score" or order_by == "dlperf":
            offers.sort(key=lambda o: o.get("dlperf", 0), reverse=True)
        elif order_by == "speed" or order_by == "inet":
            offers.sort(key=lambda o: o.get("inet_down", 0), reverse=True)
        else:
            # Default Vast scoring
            offers.sort(
                key=lambda o: (
                    -o.get("score", 0),
                    o.get("dph_total", 999999),
                )
            )

        return offers[:limit]

    def create_instance(
        self,
        offer_id: int,
        image: str = "pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime",
        disk_gb: float = 40.0,
        onstart_cmd: Optional[str] = None,
    ) -> int:
        """Create and start a new instance from an offer ID."""
        payload: Dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": float(disk_gb),
            "runtype": "ssh",
            "ssh": True,
        }
        if onstart_cmd:
            payload["onstart"] = onstart_cmd

        res = self._request(f"asks/{offer_id}/", method="PUT", data=payload)
        instance_id = (
            res.get("new_contract") or res.get("instance_id") or res.get("id")
        )
        if not instance_id:
            raise RuntimeError(
                f"Failed to create instance from offer {offer_id}: {res}"
            )
        return int(instance_id)

    def list_instances(self) -> List[Dict[str, Any]]:
        """List all active and stopped user instances."""
        res = self._request("instances/", method="GET")
        if isinstance(res, dict):
            return res.get("instances", [])
        return res if isinstance(res, list) else []

    def get_instance(self, instance_id: int) -> Dict[str, Any]:
        """Fetch instance status and connection information."""
        res = self._request(f"instances/{instance_id}/", method="GET")
        if isinstance(res, dict) and "instances" in res:
            for inst in res["instances"]:
                if inst.get("id") == instance_id:
                    return inst
        return res if isinstance(res, dict) else {}

    def stop_instance(self, instance_id: int) -> Dict[str, Any]:
        """Stop a running instance to preserve disk without paying GPU runtime."""
        return self._request(
            f"instances/{instance_id}/",
            method="PUT",
            data={"state": "stopped"},
        )

    def start_instance(self, instance_id: int) -> Dict[str, Any]:
        """Resume a stopped instance."""
        return self._request(
            f"instances/{instance_id}/",
            method="PUT",
            data={"state": "running"},
        )

    def destroy_instance(self, instance_id: int) -> Dict[str, Any]:
        """Permanently delete an instance and release storage."""
        return self._request(f"instances/{instance_id}/", method="DELETE")


# --- Output Formatter ---


def format_offers_table(
    offers: List[Dict[str, Any]], sort_key: Optional[str] = None
) -> str:
    """Format offers into a clean, human-readable table."""
    if not offers:
        return "No matching offers found."

    header = (
        f"{COLOR_BOLD}{'ID':<10} {'GPU Model':<22} {'VRAM':<10} {'System RAM':<12} "
        f"{'Price ($/h)':<13} {'DLPerf':<9} {'Down/Up (Mbps)':<16} {'Direct SSH':<10}{COLOR_RESET}"
    )
    separator = "-" * 105
    lines = [header, separator]

    for o in offers:
        oid = str(o.get("id", "N/A"))
        num_gpus = o.get("num_gpus", 1)
        gpu_name = str(o.get("gpu_name", "Unknown"))
        if num_gpus > 1:
            gpu_display = f"{num_gpus}x {gpu_name}"[:21]
        else:
            gpu_display = gpu_name[:21]

        vram_mb = o.get("gpu_ram", 0)
        vram_gb = f"{vram_mb / 1024:.1f} GB" if vram_mb else "N/A"

        ram_mb = o.get("cpu_ram", 0)
        ram_gb = f"{ram_mb / 1024:.1f} GB" if ram_mb else "N/A"

        dph = o.get("dph_total", 0.0)
        price_str = f"${dph:.3f}/hr"

        dlperf = o.get("dlperf", 0.0)
        dlperf_str = f"{dlperf:.1f}" if dlperf else "N/A"

        inet_down = o.get("inet_down", 0)
        inet_up = o.get("inet_up", 0)
        inet_str = f"{int(inet_down)}/{int(inet_up)}"

        direct_ssh = "Yes" if o.get("direct_port_count", 0) > 0 else "Port-Fwd"

        # Highlight price and RAM
        if sort_key in ("price", "dph"):
            price_str = f"{COLOR_GREEN}{price_str:<13}{COLOR_RESET}"
        elif sort_key == "ram":
            ram_gb = f"{COLOR_CYAN}{ram_gb:<12}{COLOR_RESET}"
        elif sort_key in ("vram", "gpu_ram"):
            vram_gb = f"{COLOR_CYAN}{vram_gb:<10}{COLOR_RESET}"

        line = (
            f"{oid:<10} {gpu_display:<22} {vram_gb:<10} {ram_gb:<12} "
            f"{price_str:<13} {dlperf_str:<9} {inet_str:<16} {direct_ssh:<10}"
        )
        lines.append(line)

    return "\n".join(lines)


def format_instances_table(instances: List[Dict[str, Any]]) -> str:
    """Format active user instances into a clean table."""
    if not instances:
        return "No active instances."

    header = (
        f"{COLOR_BOLD}{'ID':<10} {'Status':<12} {'GPU Model':<20} {'Price ($/h)':<13} "
        f"{'SSH Host':<28} {'SSH Port':<10} {'Disk (GB)':<10}{COLOR_RESET}"
    )
    separator = "-" * 105
    lines = [header, separator]

    for inst in instances:
        iid = str(inst.get("id", "N/A"))
        status = str(inst.get("actual_status", inst.get("cur_state", "unknown")))
        if status == "running":
            status_disp = f"{COLOR_GREEN}{status:<12}{COLOR_RESET}"
        elif status == "stopped":
            status_disp = f"{COLOR_YELLOW}{status:<12}{COLOR_RESET}"
        else:
            status_disp = f"{COLOR_CYAN}{status:<12}{COLOR_RESET}"

        gpu_name = str(inst.get("gpu_name", "N/A"))[:19]
        dph = inst.get("dph_total", 0.0)
        price_str = f"${dph:.3f}/hr"

        ssh_host = str(inst.get("ssh_host", inst.get("public_ipaddr", "N/A")))[
            :27
        ]
        ssh_port = str(inst.get("ssh_port", "N/A"))
        disk_gb = f"{inst.get('disk_space', 0.0):.1f}"

        line = (
            f"{iid:<10} {status_disp} {gpu_name:<20} {price_str:<13} "
            f"{ssh_host:<28} {ssh_port:<10} {disk_gb:<10}"
        )
        lines.append(line)

    return "\n".join(lines)


# --- Remote Operations Helpers ---


def wait_for_instance_ready(
    client: VastAPIClient, instance_id: int, timeout_sec: int = 360
) -> Dict[str, Any]:
    """Poll Vast API until instance is in running state and has SSH host/port."""
    print(
        f"\n{COLOR_BLUE}⏳ Waiting for instance {instance_id} to initialize and boot...{COLOR_RESET}"
    )
    start_time = time.time()

    while time.time() - start_time < timeout_sec:
        inst = client.get_instance(instance_id)
        status = inst.get("actual_status") or inst.get("cur_state") or "starting"
        status_msg = inst.get("status_msg") or ""
        ssh_host = inst.get("ssh_host") or inst.get("public_ipaddr")
        ssh_port = inst.get("ssh_port")

        elapsed = int(time.time() - start_time)
        print(
            f"   [+{elapsed}s] Status: {COLOR_BOLD}{status}{COLOR_RESET} "
            f"{f'({status_msg})' if status_msg else ''} | SSH: {ssh_host}:{ssh_port}"
        )

        if (
            status == "running"
            and ssh_host
            and ssh_port
            and str(ssh_port).isdigit()
        ):
            print(
                f"{COLOR_GREEN}✅ Instance {instance_id} is running!{COLOR_RESET}"
            )
            return inst

        time.sleep(8)

    raise TimeoutError(
        f"Instance {instance_id} did not reach running status within {timeout_sec}s"
    )


def wait_for_ssh_ready(
    ssh_host: str, ssh_port: int, user: str = "root", timeout_sec: int = 180
) -> bool:
    """Poll SSH port with strict host checking disabled until SSH command executes."""
    print(
        f"\n{COLOR_BLUE}🔌 Probing SSH connection to {user}@{ssh_host}:{ssh_port}...{COLOR_RESET}"
    )
    start_time = time.time()

    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=5",
        "-p",
        str(ssh_port),
        f"{user}@{ssh_host}",
        "echo READY",
    ]

    while time.time() - start_time < timeout_sec:
        res = subprocess.run(
            ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if res.returncode == 0 and "READY" in res.stdout:
            print(
                f"{COLOR_GREEN}✅ SSH daemon is ready and responding!{COLOR_RESET}"
            )
            return True
        time.sleep(5)

    raise TimeoutError(
        f"SSH connection to {ssh_host}:{ssh_port} timed out after {timeout_sec}s"
    )


def upload_setup_files(
    ssh_host: str,
    ssh_port: int,
    user: str = "root",
    remote_dest: str = "/workspace",
) -> None:
    """Upload .env and setup.sh to the remote instance."""
    repo_root = Path(__file__).resolve().parent.parent
    env_file = repo_root / ".env"
    setup_script = repo_root / "scripts" / "setup.sh"

    if not env_file.is_file():
        raise FileNotFoundError(
            f".env file not found at {env_file}. Please create it."
        )
    if not setup_script.is_file():
        raise FileNotFoundError(
            f"setup.sh script not found at {setup_script}."
        )

    print(
        f"\n{COLOR_BLUE}📦 Uploading credentials and setup scripts to {user}@{ssh_host}:{ssh_port}:{remote_dest}/...{COLOR_RESET}"
    )

    # Ensure remote directory exists
    subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-p",
            str(ssh_port),
            f"{user}@{ssh_host}",
            f"mkdir -p {remote_dest}",
        ],
        check=True,
    )

    # SCP files
    scp_cmd = [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-P",
        str(ssh_port),
        str(env_file),
        str(setup_script),
        f"{user}@{ssh_host}:{remote_dest}/",
    ]
    res = subprocess.run(scp_cmd)
    if res.returncode != 0:
        raise RuntimeError("Failed to SCP files to the remote instance.")

    # chmod +x
    subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-p",
            str(ssh_port),
            f"{user}@{ssh_host}",
            f"chmod +x {remote_dest}/setup.sh",
        ],
        check=True,
    )
    print(f"{COLOR_GREEN}✅ Setup files successfully deployed.{COLOR_RESET}")


def run_remote_pipeline(
    ssh_host: str,
    ssh_port: int,
    user: str = "root",
    remote_dest: str = "/workspace",
    extra_pipeline_args: Optional[List[str]] = None,
) -> int:
    """Execute setup.sh with --run-pipeline on the instance with live output streaming."""
    extra_pipeline_args = extra_pipeline_args or []
    pipeline_args_str = " ".join(extra_pipeline_args)
    cmd_str = f"bash {remote_dest}/setup.sh -y --run-pipeline {pipeline_args_str}"

    print(f"\n{COLOR_BLUE}🚀 Executing remote setup & pipeline:{COLOR_RESET}")
    print(f"   {COLOR_BOLD}{cmd_str}{COLOR_RESET}\n" + "=" * 60 + "\n")

    ssh_cmd = [
        "ssh",
        "-t",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(ssh_port),
        f"{user}@{ssh_host}",
        cmd_str,
    ]

    proc = subprocess.run(ssh_cmd)
    return proc.returncode


def download_artifacts(
    ssh_host: str,
    ssh_port: int,
    user: str = "root",
    remote_project_dir: str = "/workspace/cvm-swin-gcn",
    local_dest: Optional[Path] = None,
) -> None:
    """Download artifacts/ and evaluation/ back to local repository."""
    local_dest = local_dest or Path(__file__).resolve().parent.parent
    print(
        f"\n{COLOR_BLUE}📥 Downloading artifacts & evaluation results to {local_dest}...{COLOR_RESET}"
    )

    for folder in ["artifacts", "evaluation"]:
        local_folder = local_dest / folder
        local_folder.mkdir(parents=True, exist_ok=True)
        scp_cmd = [
            "scp",
            "-r",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-P",
            str(ssh_port),
            f"{user}@{ssh_host}:{remote_project_dir}/{folder}/*",
            str(local_folder),
        ]
        subprocess.run(
            scp_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    print(f"{COLOR_GREEN}✅ Artifacts download complete.{COLOR_RESET}")


# --- CLI Commands ---


def cmd_search(args: argparse.Namespace, client: VastAPIClient) -> None:
    """Search and format available offers."""
    offers = client.search_offers(
        gpu_name=args.gpu,
        num_gpus=args.num_gpus,
        max_price=args.max_price,
        min_ram_gb=args.min_ram,
        min_vram_gb=args.min_vram,
        min_disk_gb=args.disk,
        min_cuda=args.min_cuda,
        verified_only=not args.unverified,
        order_by=args.sort,
        limit=args.limit,
    )

    if args.format == "json":
        print(json.dumps(offers, indent=2))
    else:
        print(
            f"\n{COLOR_BOLD}=== Vast.ai GPU Offers (Sorted by: {args.sort}) ==={COLOR_RESET}"
        )
        print(format_offers_table(offers, sort_key=args.sort))
        print(
            f"\nFound {len(offers)} matching offers. Use --sort price|ram|vram|score to change sorting."
        )


def cmd_list(args: argparse.Namespace, client: VastAPIClient) -> None:
    """List current user instances."""
    instances = client.list_instances()
    if args.format == "json":
        print(json.dumps(instances, indent=2))
    else:
        print(f"\n{COLOR_BOLD}=== Your Active Vast.ai Instances ==={COLOR_RESET}")
        print(format_instances_table(instances))


def cmd_run(args: argparse.Namespace, client: VastAPIClient) -> None:
    """Complete end-to-end automation: search/create, wait, upload, setup, train, and manage lifecycle."""
    instance_id = None
    created_new = False

    try:
        # Check if an existing instance ID was specified
        if args.instance_id:
            instance_id = int(args.instance_id)
            print(
                f"{COLOR_BLUE}Using existing instance ID: {instance_id}{COLOR_RESET}"
            )
        else:
            print(
                f"\n{COLOR_BLUE}🔍 Finding optimal offer matching criteria (Sorted by: {args.sort})...{COLOR_RESET}"
            )
            offers = client.search_offers(
                gpu_name=args.gpu,
                num_gpus=args.num_gpus,
                max_price=args.max_price,
                min_ram_gb=args.min_ram,
                min_vram_gb=args.min_vram,
                min_disk_gb=args.disk,
                min_cuda=args.min_cuda,
                verified_only=not args.unverified,
                order_by=args.sort,
                limit=5,
            )

            if not offers:
                print(
                    f"{COLOR_RED}❌ No matching offers found. Try relaxing price, RAM, or GPU filters.{COLOR_RESET}"
                )
                sys.exit(1)

            selected_offer = offers[0]
            offer_id = selected_offer["id"]
            gpu_name = selected_offer.get("gpu_name")
            dph = selected_offer.get("dph_total", 0.0)
            ram_gb = selected_offer.get("cpu_ram", 0) / 1024

            print(
                f"Selected Offer #{offer_id}: {COLOR_BOLD}{gpu_name}{COLOR_RESET} | "
                f"{ram_gb:.1f} GB RAM | ${dph:.3f}/hr"
            )

            print(
                f"\n{COLOR_BLUE}🚀 Launching new Vast.ai instance (Image: {args.image}, Disk: {args.disk}GB)...{COLOR_RESET}"
            )
            instance_id = client.create_instance(
                offer_id=offer_id, image=args.image, disk_gb=args.disk
            )
            created_new = True
            print(
                f"{COLOR_GREEN}✅ Created contract. Instance ID: {instance_id}{COLOR_RESET}"
            )

        # Wait for instance boot and SSH readiness
        inst = wait_for_instance_ready(client, instance_id)
        ssh_host = inst.get("ssh_host") or inst.get("public_ipaddr")
        ssh_port = int(inst["ssh_port"])

        wait_for_ssh_ready(ssh_host, ssh_port)

        # Upload .env and setup.sh
        upload_setup_files(ssh_host, ssh_port)

        # Collect extra pipeline arguments
        pipeline_args = []
        if args.epochs:
            pipeline_args.extend(["--epochs", str(args.epochs)])
        if args.batch_size:
            pipeline_args.extend(["--batch-size", str(args.batch_size)])
        if args.skip_download:
            pipeline_args.append("--skip-download")
        if args.train_only:
            pipeline_args.append("--train-only")
        if args.skip_eval:
            pipeline_args.append("--skip-eval")
        if args.extra_args:
            pipeline_args.extend(args.extra_args)

        # Execute training
        ret_code = run_remote_pipeline(
            ssh_host,
            ssh_port,
            extra_pipeline_args=pipeline_args,
        )

        if ret_code == 0:
            print(
                f"\n{COLOR_GREEN}🎉 Remote pipeline execution completed successfully!{COLOR_RESET}"
            )
            if args.download_artifacts:
                download_artifacts(ssh_host, ssh_port)
        else:
            print(
                f"\n{COLOR_RED}❌ Remote pipeline failed with exit code {ret_code}.{COLOR_RESET}"
            )

    except KeyboardInterrupt:
        print(f"\n{COLOR_YELLOW}⚠️ Execution interrupted by user.{COLOR_RESET}")
    except Exception as e:
        print(f"\n{COLOR_RED}❌ Error: {e}{COLOR_RESET}")
    finally:
        # Lifecycle cleanup
        if instance_id and created_new:
            if args.destroy_on_finish:
                print(
                    f"\n{COLOR_YELLOW}🗑️ Destroying instance {instance_id} as requested (--destroy-on-finish)...{COLOR_RESET}"
                )
                client.destroy_instance(instance_id)
                print(f"{COLOR_GREEN}✅ Instance {instance_id} destroyed.")
            elif args.stop_on_finish:
                print(
                    f"\n{COLOR_YELLOW}⏸️ Stopping instance {instance_id} to save costs (--stop-on-finish)...{COLOR_RESET}"
                )
                client.stop_instance(instance_id)
                print(f"{COLOR_GREEN}✅ Instance {instance_id} stopped.")
            else:
                print(
                    f"\n{COLOR_CYAN}ℹ️ Instance {instance_id} is still running. Stop or destroy it when done:{COLOR_RESET}"
                )
                print(
                    f"   python3 scripts/vast_runner.py stop {instance_id}    # pause"
                )
                print(
                    f"   python3 scripts/vast_runner.py destroy {instance_id} # delete"
                )


def cmd_stop(args: argparse.Namespace, client: VastAPIClient) -> None:
    """Stop an instance."""
    client.stop_instance(args.instance_id)
    print(
        f"{COLOR_GREEN}✅ Stop command issued for instance {args.instance_id}.{COLOR_RESET}"
    )


def cmd_destroy(args: argparse.Namespace, client: VastAPIClient) -> None:
    """Destroy an instance."""
    client.destroy_instance(args.instance_id)
    print(
        f"{COLOR_GREEN}✅ Destroy command issued for instance {args.instance_id}.{COLOR_RESET}"
    )


def cmd_ssh(args: argparse.Namespace, client: VastAPIClient) -> None:
    """SSH directly into an instance."""
    inst = client.get_instance(args.instance_id)
    ssh_host = inst.get("ssh_host") or inst.get("public_ipaddr")
    ssh_port = inst.get("ssh_port")
    if not ssh_host or not ssh_port:
        print(
            f"{COLOR_RED}❌ Instance {args.instance_id} does not have SSH host/port available yet.{COLOR_RESET}"
        )
        sys.exit(1)

    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(ssh_port),
        f"root@{ssh_host}",
    ]
    if args.command:
        ssh_cmd.append(args.command)
    subprocess.run(ssh_cmd)


def cmd_download(args: argparse.Namespace, client: VastAPIClient) -> None:
    """Download artifacts from a running instance."""
    inst = client.get_instance(args.instance_id)
    ssh_host = inst.get("ssh_host") or inst.get("public_ipaddr")
    ssh_port = inst.get("ssh_port")
    if not ssh_host or not ssh_port:
        print(
            f"{COLOR_RED}❌ Instance {args.instance_id} does not have SSH host/port available.{COLOR_RESET}"
        )
        sys.exit(1)

    download_artifacts(
        ssh_host,
        int(ssh_port),
        local_dest=Path(args.dest) if args.dest else None,
    )


# --- Main CLI Parser ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vast.ai GPU Cloud Automation & Runner for Swin-GCN Network"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Vast.ai API key (overrides VAST_API_KEY env / .env)",
    )

    subparsers = parser.add_subparsers(
        dest="subcommand", help="Action to execute"
    )

    # 1. Search
    search_p = subparsers.add_parser("search", help="Search available GPU offers")
    search_p.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU model filter (e.g., 'RTX 4090', 'RTX 3090', 'A4000')",
    )
    search_p.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs (default: 1)"
    )
    search_p.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="Maximum total price in $/hr",
    )
    search_p.add_argument(
        "--min-ram",
        type=float,
        default=None,
        help="Minimum system RAM in GB (e.g., 32)",
    )
    search_p.add_argument(
        "--min-vram",
        type=float,
        default=None,
        help="Minimum GPU VRAM in GB (e.g., 24)",
    )
    search_p.add_argument(
        "--disk",
        type=float,
        default=35.0,
        help="Required disk storage in GB (default: 35)",
    )
    search_p.add_argument(
        "--min-cuda",
        type=float,
        default=12.0,
        help="Minimum CUDA version (default: 12.0)",
    )
    search_p.add_argument(
        "--sort",
        type=str,
        default="price",
        choices=["price", "ram", "vram", "score", "speed"],
        help="Sort offers by: price, ram, vram, score (dlperf), speed (default: price)",
    )
    search_p.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Maximum offers to show (default: 15)",
    )
    search_p.add_argument(
        "--unverified",
        action="store_true",
        help="Include unverified community hosts",
    )
    search_p.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )

    # 2. List
    list_p = subparsers.add_parser(
        "list", help="List active and stopped user instances"
    )
    list_p.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )

    # 3. Run (Full End-to-End Automation)
    run_p = subparsers.add_parser(
        "run", help="Auto-provision instance, deploy code, setup, and run training"
    )
    run_p.add_argument(
        "--instance-id",
        type=int,
        default=None,
        help="Attach to existing instance ID instead of creating a new one",
    )
    run_p.add_argument(
        "--gpu",
        type=str,
        default="RTX 4090",
        help="GPU model preference (default: 'RTX 4090')",
    )
    run_p.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs (default: 1)"
    )
    run_p.add_argument(
        "--max-price",
        type=float,
        default=0.75,
        help="Max $/hr price cap (default: 0.75)",
    )
    run_p.add_argument(
        "--min-ram",
        type=float,
        default=24.0,
        help="Minimum system RAM in GB (default: 24)",
    )
    run_p.add_argument(
        "--min-vram",
        type=float,
        default=16.0,
        help="Minimum GPU VRAM in GB (default: 16)",
    )
    run_p.add_argument(
        "--disk",
        type=float,
        default=40.0,
        help="Disk storage size in GB (default: 40)",
    )
    run_p.add_argument(
        "--min-cuda",
        type=float,
        default=12.0,
        help="Minimum CUDA version (default: 12.0)",
    )
    run_p.add_argument(
        "--sort",
        type=str,
        default="price",
        choices=["price", "ram", "vram", "score", "speed"],
        help="Sort offers by: price, ram, vram, score (default: price)",
    )
    run_p.add_argument(
        "--image",
        type=str,
        default="pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime",
        help="Docker image to run",
    )
    run_p.add_argument(
        "--unverified",
        action="store_true",
        help="Include unverified community hosts",
    )
    run_p.add_argument(
        "--epochs", "-e", type=int, default=100, help="Number of training epochs"
    )
    run_p.add_argument(
        "--batch-size", "-b", type=int, default=8, help="Batch size for training"
    )
    run_p.add_argument(
        "--skip-download", action="store_true", help="Skip dataset download step"
    )
    run_p.add_argument(
        "--train-only",
        action="store_true",
        help="Skip data preprocessing and run training directly",
    )
    run_p.add_argument(
        "--skip-eval", action="store_true", help="Skip final model evaluation"
    )
    run_p.add_argument(
        "--stop-on-finish",
        action="store_true",
        help="Automatically stop the instance when pipeline finishes",
    )
    run_p.add_argument(
        "--destroy-on-finish",
        action="store_true",
        help="Automatically destroy the instance when pipeline finishes",
    )
    run_p.add_argument(
        "--download-artifacts",
        action="store_true",
        default=True,
        help="Download model artifacts and evaluation back to local disk (default: True)",
    )
    run_p.add_argument(
        "extra_args",
        nargs="*",
        help="Additional flags to pass directly to train-pipeline.py",
    )

    # 4. Stop
    stop_p = subparsers.add_parser("stop", help="Stop an instance")
    stop_p.add_argument("instance_id", type=int, help="Instance ID to stop")

    # 5. Destroy
    destroy_p = subparsers.add_parser("destroy", help="Destroy an instance")
    destroy_p.add_argument(
        "instance_id", type=int, help="Instance ID to destroy"
    )

    # 6. SSH
    ssh_p = subparsers.add_parser(
        "ssh", help="Open an interactive SSH shell or run a command"
    )
    ssh_p.add_argument("instance_id", type=int, help="Instance ID to SSH into")
    ssh_p.add_argument(
        "command",
        nargs="?",
        type=str,
        default=None,
        help="Optional remote command to run",
    )

    # 7. Download
    download_p = subparsers.add_parser(
        "download", help="Download artifacts from a running instance"
    )
    download_p.add_argument(
        "instance_id", type=int, help="Instance ID to download from"
    )
    download_p.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Local destination folder (default: project root)",
    )

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    api_key = get_api_key(args.api_key)

    if not api_key:
        print(
            f"{COLOR_RED}❌ Vast.ai API Key is required!{COLOR_RESET}\n"
            "Please provide it via one of the following methods:\n"
            "  1. Add to your .env file: VAST_API_KEY=your_key_here\n"
            "  2. Export in shell: export VAST_API_KEY=your_key_here\n"
            "  3. Pass CLI argument: python3 scripts/vast_runner.py --api-key <KEY> ...\n",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        client = VastAPIClient(api_key=api_key)
    except Exception as e:
        print(f"{COLOR_RED}Initialization error: {e}{COLOR_RESET}", file=sys.stderr)
        sys.exit(1)

    if args.subcommand == "search":
        cmd_search(args, client)
    elif args.subcommand == "list":
        cmd_list(args, client)
    elif args.subcommand == "run":
        cmd_run(args, client)
    elif args.subcommand == "stop":
        cmd_stop(args, client)
    elif args.subcommand == "destroy":
        cmd_destroy(args, client)
    elif args.subcommand == "ssh":
        cmd_ssh(args, client)
    elif args.subcommand == "download":
        cmd_download(args, client)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
