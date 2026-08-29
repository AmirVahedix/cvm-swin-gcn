---
name: vast-ai-runner
description: >-
  Search, filter, provision, and manage Vast.ai GPU cloud instances to automate training pipelines,
  remote model execution, setup scripts, and artifact synchronization. Supports formatting and sorting
  offers by price ($/hr), system RAM, GPU VRAM, DLPerf score, or network speed, with tmux detached execution and auto-destroy on finish/error.
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

- **Decoupled `tmux` Background Execution & Laptop-Closing Safety**:
  - Remote training runs inside a persistent `tmux` session named `train`.
  - Closing your laptop, losing Wi-Fi, or disconnecting SSH will **never** stop or interrupt GPU training!
  - Real-time output is streamed to your terminal when connected.
  - Re-attach anytime with `python3 scripts/vast_runner.py attach` or stream logs with `python3 scripts/vast_runner.py logs -f`.
- **Automatic Remote Self-Destruction on Completion**:
  - Once training finishes and artifacts are logged/uploaded to MLflow/MinIO, the remote instance **automatically deletes itself** via the Vast.ai REST API.
  - Cloud billing stops immediately even if your laptop was offline when training completed!
  - Local sync occurs automatically before destruction if the laptop is connected.
  - Override with `--no-destroy` / `--keep-alive` or `--stop-on-finish`.
- **Dynamic `INSTANCE_ID` Tracking in `.env`**:
  - When an instance is launched, its ID is automatically stored in `.env` (`INSTANCE_ID=<id>`).
  - Subsequent commands (`attach`, `logs`, `stop`, `destroy`, `ssh`, `download`, `run`) automatically read `INSTANCE_ID` from `.env`—no need to type or copy instance IDs manually!
  - When an instance is destroyed, `INSTANCE_ID` is automatically cleared from `.env`.
- **Dynamic Formatting & Sorting**: Sort available offers by Auto Sort score (`--sort score`, default), raw GPU speed (`--sort dlperf`), performance per dollar (`--sort value`), lowest price (`--sort price`), highest system RAM (`--sort ram`), or GPU VRAM (`--sort vram`).
- **Host Reliability Filtering**: Filter out unreliable hosts using `--min-reliability 0.90` (default 90% reliability threshold).
- **Automated Cost Protection & 30-Second Auto-Destroy on Errors**:
  - If an error occurs during launch/setup, a **30-second countdown prompt** is presented asking if the instance should be destroyed.
  - If the user confirms or if the 30-second timer elapses with no input, the instance is **automatically destroyed** to prevent accidental cloud charges!
  - Users can press `n` during the countdown to keep the instance alive for debugging.

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

Automatically finds the best matching offer, rents the instance, saves `INSTANCE_ID` in `.env`, waits for boot & SSH readiness, uploads `.env` and `setup.sh`, configures dependencies, runs `train-pipeline.py` inside `tmux`, streams logs, and auto-destroys on completion:

```bash
# Basic run with RTX 4090, 50 epochs, sorting by lowest price (runs in tmux, auto-destroys on finish)
python3 scripts/vast_runner.py run --gpu "RTX 4090" --max-price 0.70 --sort price --epochs 50

# Run in detached mode (starts remote tmux pipeline and exits CLI immediately)
python3 scripts/vast_runner.py run --gpu "RTX 3090" --max-price 0.20 --detach

# Run training only (skip preprocessing steps 1-4)
python3 scripts/vast_runner.py run --train-only --epochs 50 --batch-size 16

# Run and keep instance alive after training (do not auto-destroy)
python3 scripts/vast_runner.py run --epochs 50 --no-destroy

# Run and auto-stop (pause) instance after training instead of destroying
python3 scripts/vast_runner.py run --epochs 50 --stop-on-finish

# Manual Attach Mode (Interactive prompt for Instance ID, SSH Host, and SSH Port)
python3 scripts/vast_runner.py run --manual

# Manual Attach Mode with CLI flags (non-interactive)
python3 scripts/vast_runner.py run --instance-id 12345 --host 74.50.x.x --port 12345
```

---

### 3. Monitoring & Instance Management

All commands automatically use the `INSTANCE_ID` stored in `.env` if no ID is passed:

```bash
# List all active and stopped user instances
python3 scripts/vast_runner.py list

# Interactively attach to the remote tmux training session
python3 scripts/vast_runner.py attach

# Stream or view live remote pipeline logs
python3 scripts/vast_runner.py logs -f

# View last 100 log lines
python3 scripts/vast_runner.py logs -n 100

# Open interactive SSH shell (or SSH into tmux)
python3 scripts/vast_runner.py ssh
python3 scripts/vast_runner.py ssh --tmux

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
   - Run `python3 scripts/vast_runner.py run ...` with the user's requested parameters (e.g., `--epochs`, `--batch-size`).
   - The job will automatically execute in `tmux` and auto-destroy upon completion.
4. **Monitor & Report**:
   - Live stream logs or use `python3 scripts/vast_runner.py logs -f` / `attach`.
   - Confirm artifact synchronization (`./artifacts/best.pth`, `./evaluation/`) and MLflow tracking.
5. **Cost Optimization**:
   - If the instance was kept alive with `--no-destroy` or `--stop-on-finish`, remind the user to stop or destroy it when done.

---

## Reusing in Other Projects

To use this runner in any other project:
1. Copy `scripts/vast_runner.py` and `scripts/setup.sh` to the project's `scripts/` directory.
2. Ensure the project's `.env` contains `VAST_API_KEY` and any repository credentials.
3. The global skill is registered in `~/.gemini/config/skills/vast-ai-runner/` so Antigravity can automatically invoke it across all your repositories.
