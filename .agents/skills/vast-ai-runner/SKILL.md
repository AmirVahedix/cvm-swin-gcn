---
name: vast-ai-runner
description: >-
  Search, filter, provision, and manage Vast.ai GPU cloud instances to automate training pipelines,
  remote model execution, setup scripts, and artifact synchronization. Supports formatting and sorting
  offers by price ($/hr), system RAM, GPU VRAM, DLPerf score, or network speed, with 30s auto-destroy on error.
---

# Vast.ai GPU Cloud Runner & Pipeline Automation

This skill provides full automation for discovering, provisioning, configuring, and executing deep learning workloads on [Vast.ai](https://vast.ai) GPU cloud instances.

## Prerequisites & Authentication

Ensure `VAST_API_KEY` is available via one of the following:
1. Defined in the project's `.env` file (`VAST_API_KEY=your_key_here`).
2. Exported in the shell environment (`export VAST_API_KEY=...`).
3. Stored in `~/.vast_api_key`.
4. Passed explicitly via `--api-key <KEY>`.

---

## Key Features & Safety Mechanisms

- **Dynamic `INSTANCE_ID` Tracking in `.env`**:
  - When an instance is launched, its ID is automatically stored in `.env` (`INSTANCE_ID=<id>`).
  - Subsequent commands (`stop`, `destroy`, `ssh`, `download`, `run`) automatically read `INSTANCE_ID` from `.env`—no need to type or copy instance IDs manually!
  - When an instance is destroyed, `INSTANCE_ID` is automatically cleared from `.env`.
- **Dynamic Formatting & Sorting**: Sort available offers by Auto Sort score (`--sort score`, default), raw GPU speed (`--sort dlperf`), performance per dollar (`--sort value`), lowest price (`--sort price`), highest system RAM (`--sort ram`), or GPU VRAM (`--sort vram`).
- **Host Reliability Filtering**: Filter out unreliable hosts using `--min-reliability 0.90` (default 90% reliability threshold).
- **One-Command Auto-Execution (`run`)**: Search, provision, wait for SSH, upload `.env` + `setup.sh`, execute training with live log streaming, and download checkpoints (`artifacts/`, `evaluation/`).
- **Automated Cost Protection & 30-Second Auto-Destroy**:
  - Whenever an error occurs, the training pipeline exits with an error code, or execution is interrupted (Ctrl+C), a **30-second countdown prompt** is presented asking if the instance should be destroyed.
  - If the user confirms or if the 30-second timer elapses with no input, the instance is **automatically destroyed** to prevent accidental cloud charges!
  - Users can press `n` during the countdown to keep the instance alive for debugging.
  - Set custom countdown duration with `--timeout-destroy <SECONDS>` (default: 30).

---

## Quick Reference & Commands

All operations are driven via `scripts/vast_runner.py` (or portable python invocation):

```bash
# General help
python3 scripts/vast_runner.py --help
```

### 1. Searching & Formatting GPU Offers

Search available GPU machines with custom filters and table formatting:

```bash
# Default search (Auto Sort by score & performance under host reliability >= 90%)
python3 scripts/vast_runner.py search --max-price 0.15

# Sort by raw GPU performance (DLPerf TFLOPS)
python3 scripts/vast_runner.py search --max-price 0.15 --sort dlperf

# Sort by performance per dollar (DLPerf / price)
python3 scripts/vast_runner.py search --max-price 0.15 --sort value

# Filter specifically by RTX 3090 under $0.15/hr
python3 scripts/vast_runner.py search --gpu "3090" --max-price 0.15

# Sort by lowest price ($/hr)
python3 scripts/vast_runner.py search --sort price

# Sort by highest system RAM (GB)
python3 scripts/vast_runner.py search --sort ram

# Sort by highest GPU VRAM (GB)
python3 scripts/vast_runner.py search --sort vram

# Output as JSON for programmatic inspection
python3 scripts/vast_runner.py search --gpu "RTX 3090" --format json
```

---

### 2. End-to-End Automated Training (`run`)

Automatically finds the best matching offer, rents the instance, saves `INSTANCE_ID` in `.env`, waits for boot & SSH readiness, uploads `.env` and `setup.sh`, configures dependencies, runs `train-pipeline.py`, streams logs, and downloads artifacts:

```bash
# Basic run with RTX 4090, 50 epochs, sorting by lowest price
python3 scripts/vast_runner.py run --gpu "RTX 4090" --max-price 0.70 --sort price --epochs 50

# Run with minimum 32GB system RAM, auto-stop instance when done
python3 scripts/vast_runner.py run --min-ram 32 --max-price 0.80 --stop-on-finish --epochs 100

# Run training only (skip preprocessing steps 1-4)
python3 scripts/vast_runner.py run --train-only --epochs 50 --batch-size 16

# Attach to an already running instance (or uses INSTANCE_ID from .env)
python3 scripts/vast_runner.py run --epochs 100

# Manual Attach Mode (Interactive prompt for Instance ID, SSH Host, and SSH Port)
# Deploys code/setup, streams training, downloads artifacts, and auto-destroys on finish
python3 scripts/vast_runner.py run --manual

# Manual Attach Mode with CLI flags (non-interactive)
python3 scripts/vast_runner.py run --instance-id 12345 --host 74.50.x.x --port 12345
```

---

### 3. Instance Lifecycle Management

All commands automatically use the `INSTANCE_ID` stored in `.env` if no ID is passed:

```bash
# List all active and stopped user instances
python3 scripts/vast_runner.py list

# Open interactive SSH or run a remote command (uses INSTANCE_ID from .env)
python3 scripts/vast_runner.py ssh
python3 scripts/vast_runner.py ssh "nvidia-smi"

# Download artifacts & evaluation results from instance
python3 scripts/vast_runner.py download

# Stop instance to pause billing while keeping data intact
python3 scripts/vast_runner.py stop

# Destroy instance permanently (clears INSTANCE_ID from .env)
python3 scripts/vast_runner.py destroy
```

---

## Step-by-Step Procedure for Agents

When requested by the user to train or run workloads on Vast.ai:

1. **Verify API Key**: Check if `VAST_API_KEY` is configured in `.env` or the environment. If missing, prompt the user.
2. **Search / Review Offers**:
   - Run `python3 scripts/vast_runner.py search --sort price` (or `--sort ram` if high memory is requested).
   - Verify that offers meet the required GPU, VRAM, RAM, and price constraints.
3. **Trigger Execution**:
   - Run `python3 scripts/vast_runner.py run ...` with the user's requested parameters (e.g., `--epochs`, `--batch-size`, `--stop-on-finish`).
4. **Monitor & Report**:
   - Monitor the command output for setup completion and training progress.
   - Once completed, confirm artifact synchronization (`./artifacts/best.pth`, `./evaluation/`).
5. **Cost Optimization**:
   - Remind the user if an instance is still active, or stop/destroy it using `python3 scripts/vast_runner.py destroy`.

---

## Reusing in Other Projects

To use this runner in any other project:
1. Copy `scripts/vast_runner.py` and `scripts/setup.sh` to the project's `scripts/` directory.
2. Ensure the project's `.env` contains `VAST_API_KEY` and any repository credentials.
3. The global skill is registered in `~/.gemini/config/skills/vast-ai-runner/` so Antigravity can automatically invoke it across all your repositories.
