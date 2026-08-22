---
name: vast-ai-runner
description: >-
  Search, filter, provision, and manage Vast.ai GPU cloud instances to automate training pipelines,
  remote model execution, setup scripts, and artifact synchronization. Supports formatting and sorting
  offers by price ($/hr), system RAM, GPU VRAM, DLPerf score, or network speed.
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

## Quick Reference & Commands

All operations are driven via `scripts/vast_runner.py` (or portable python invocation):

```bash
# General help
python3 scripts/vast_runner.py --help
```

### 1. Searching & Formatting GPU Offers

Search available GPU machines with custom filters and table formatting:

```bash
# Sort by lowest price ($/hr)
python3 scripts/vast_runner.py search --sort price

# Sort by highest system RAM
python3 scripts/vast_runner.py search --sort ram

# Sort by highest GPU VRAM
python3 scripts/vast_runner.py search --sort vram

# Filter by specific GPU model, price limit, and minimum RAM
python3 scripts/vast_runner.py search --gpu "RTX 4090" --max-price 0.65 --min-ram 32 --sort price

# Output as JSON for programmatic inspection
python3 scripts/vast_runner.py search --gpu "RTX 3090" --format json
```

#### Sorting Flags:
- `--sort price`: Sort by cheapest $/hr total (ascending).
- `--sort ram`: Sort by highest system RAM in GB (descending).
- `--sort vram`: Sort by highest GPU VRAM in GB (descending).
- `--sort score`: Sort by DLPerf benchmark score (descending).
- `--sort speed`: Sort by download speed (descending).

---

### 2. End-to-End Automated Training (`run`)

Automatically finds the best matching offer, rents the instance, waits for boot & SSH readiness, uploads `.env` and `setup.sh`, configures dependencies, runs `train-pipeline.py`, streams logs, and downloads artifacts:

```bash
# Basic run with RTX 4090, 50 epochs, sorting by lowest price
python3 scripts/vast_runner.py run --gpu "RTX 4090" --max-price 0.70 --sort price --epochs 50

# Run with minimum 32GB system RAM, auto-stop instance when done
python3 scripts/vast_runner.py run --min-ram 32 --max-price 0.80 --stop-on-finish --epochs 100

# Run training only (skip preprocessing steps 1-4)
python3 scripts/vast_runner.py run --train-only --epochs 50 --batch-size 16

# Attach to an already created / running instance instead of creating a new one
python3 scripts/vast_runner.py run --instance-id 1234567 --epochs 100
```

---

### 3. Instance Lifecycle Management

```bash
# List all active and stopped user instances
python3 scripts/vast_runner.py list

# Open interactive SSH or run a remote command
python3 scripts/vast_runner.py ssh <INSTANCE_ID>
python3 scripts/vast_runner.py ssh <INSTANCE_ID> "nvidia-smi"

# Download artifacts & evaluation results from instance
python3 scripts/vast_runner.py download <INSTANCE_ID>

# Stop instance to pause billing while keeping data intact
python3 scripts/vast_runner.py stop <INSTANCE_ID>

# Destroy instance permanently to release storage
python3 scripts/vast_runner.py destroy <INSTANCE_ID>
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
   - Remind the user if an instance is still active, or stop/destroy it if requested.

---

## Reusing in Other Projects

To use this runner in any other project:
1. Copy `scripts/vast_runner.py` and `scripts/setup.sh` to the project's `scripts/` directory.
2. Ensure the project's `.env` contains `VAST_API_KEY` and any repository credentials.
3. The global skill is registered in `~/.gemini/config/skills/vast-ai-runner/` so Antigravity can automatically invoke it across all your repositories.
