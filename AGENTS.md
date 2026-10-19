# AGENTS.md — Developer & AI Agent Operations Manual

Welcome to **CVM Swin-GCN Network** (`cvm-swin-gcn`). This document serves as the operational guide and technical specification for AI agents (and human engineers) working in this repository.

---

## 1. Project Overview & Clinical Context

### Objective
This project trains a state-of-the-art deep learning model for **Cervical Vertebral Maturation (CVM)** staging on lateral cephalometric radiographs. CVM assessment evaluates the biological skeletal maturation of pediatric and orthodontic patients by classifying morphological changes in the cervical vertebrae (C2, C3, and C4) into stages CS1 through CS6—eliminating the need for supplementary ionizing hand-wrist radiographs.

### The 13 Anatomical Landmarks
The network predicts **13 anatomical landmarks** across vertebrae C2, C3, and C4 defined in `src/data/constants.py`:

| Index | Label | Vertebra | Anatomical Description |
|:---:|:---:|:---:|:---|
| **0** | `C2_PI` | C2 | Posterior-Inferior border of C2 |
| **1** | `C2_IC` | C2 | Inferior-Concavity point of C2 (deepest point of curvature) |
| **2** | `C2_AI` | C2 | Anterior-Inferior border of C2 |
| **3** | `C3_PS` | C3 | Posterior-Superior corner of C3 |
| **4** | `C3_AS` | C3 | Anterior-Superior corner of C3 |
| **5** | `C3_PI` | C3 | Posterior-Inferior corner of C3 |
| **6** | `C3_IC` | C3 | Inferior-Concavity point of C3 |
| **7** | `C3_AI` | C3 | Anterior-Inferior corner of C3 |
| **8** | `C4_PS` | C4 | Posterior-Superior corner of C4 |
| **9** | `C4_AS` | C4 | Anterior-Superior corner of C4 |
| **10** | `C4_PI` | C4 | Posterior-Inferior corner of C4 |
| **11** | `C4_IC` | C4 | Inferior-Concavity point of C4 |
| **12** | `C4_AI` | C4 | Anterior-Inferior corner of C4 |

> **CRITICAL INVARIANT**: Never alter or reorder these indices or strings. Downstream morphometric calculations (`src/utils/cvm_calculator.py`), evaluation modules (`src/utils/metrics.py`), and precomputed label arrays depend strictly on this 0–12 sequence.

---

## 2. Technical Architecture & Design

### Model Architecture (`src/models/model.py`)
The primary model is `CephalometricSwinGCN`:
1. **Backbone**: Swin Transformer Base (`swin_base_patch4_window7_224` via `timm`) operating at $640 \times 640$ resolution, outputting multi-scale hierarchical feature maps across 4 stages.
2. **Decoder**: U-Net feature pyramid decoder with transposed convolutions and skip connections (`UNetUpBlock`), projecting fused hierarchical representations to 13 heatmap channels.
3. **Heatmap Normalization**: Sigmoid activation strictly bounding output heatmaps to $[0, 1]$, matching ground-truth 2D Gaussian targets.
4. **Coordinate Head**: Fully differentiable `SoftArgmax2D` with learnable per-landmark temperature parameters $\tau_i$, computing global spatial expectations over normalized coordinate grids $[0, 1] \times [0, 1]$.
5. **Graph Adjacency Modeling**: Built-in adjacency buffer (`adj_matrix`) modeling intra-vertebra geometry (superior, inferior, anterior, posterior borders, and concavity dips) and inter-vertebra spinal alignment ($C2 \leftrightarrow C3$ and $C3 \leftrightarrow C4$).

### Loss Formulation (`src/models/losses.py`)
The total loss is a balanced composite of heatmap regression, coordinate regression, and anatomical structural consistency:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{hm}} \mathcal{L}_{\text{AWL}}(\hat{H}, H^*) + \gamma(t) \cdot \lambda_{\text{cd}} \mathcal{L}_{\text{Wing}}(\hat{C}, C^*) + \gamma(t) \cdot \lambda_{\text{graph}} \mathcal{L}_{\text{Graph}}(\hat{C}, C^*)$$

- **Adaptive Wing Loss (AWL)**: Heatmap loss adapting gradients dynamically near Gaussian peaks while smoothing background errors ($\omega=14.0$, $\theta=0.5$, $\epsilon=1.0$, $\alpha=2.1$).
- **Wing Loss**: Coordinate loss operating directly on $[0, 1]$ soft-argmax coordinates, robust to large outliers while emphasizing sub-pixel accuracy.
- **Anatomical Graph Loss**: Penalizes deviations in pairwise Euclidean distances between anatomically connected landmarks relative to ground-truth geometries.
- **Warmup Schedule $\gamma(t)$**: Ramps coordinate and graph loss weights from $0.0$ to $1.0$ over `WARMUP_COORD_EPOCHS` (default: 5) to allow heatmaps to stabilize coarse localization before strong geometric constraints engage.

### Optimization & Training Strategy (`src/train.py`)
- **Optimizer**: AdamW with **Layer-wise Learning Rate Decay (LLRD)** (`llrd_decay_rate=0.8`), applying higher learning rates to the decoder and soft-argmax head while conservatively fine-tuning Swin backbone stages.
- **Scheduler**: Cosine annealing with linear warmup.
- **Balanced Class Sampling**: Uses `torch.utils.data.WeightedRandomSampler` with configurable smoothing power (`BALANCED_SAMPLING_POWER=0.5`) across CVM stages 1–6 to counteract class imbalance.
- **Mixed Precision**: Automatic Mixed Precision (`torch.cuda.amp.autocast`) with `GradScaler` for efficient GPU utilization.

---

## 3. Directory Layout

```
swin-gcn-network/
├── AGENTS.md                  # This file (AI agent instructions & project reference)
├── README.md                  # General project readme
├── pyproject.toml             # Project dependencies and tool configurations
├── requirements.txt           # Pinned python requirements
├── .env.example               # Example environment variable template
├── .python-version            # Python version specification (3.10)
├── artifacts/                 # Saved model weights (best.pth, full checkpoints)
├── dataset/                   # Local dataset directory
│   ├── train/                 # Training images & .npz label arrays
│   ├── val/                   # Validation images & .npz label arrays
│   ├── test/                  # Test images & .npz label arrays
│   ├── val_image_ids.json     # Deterministic validation set image IDs
│   └── test_image_ids.json    # Deterministic test set image IDs
├── evaluation/                # Evaluation output reports, metric JSONs, and clinical plots
├── scripts/                   # Automation, training pipeline, and cloud execution scripts
│   ├── train-pipeline.py      # End-to-end local data download, prep, train & eval
│   ├── vast_runner.py         # Vast.ai GPU cloud discover, provision, run & auto-destroy
│   ├── setup.sh               # Environment & dependency bootstrap for remote instances
│   ├── setup-inference.sh     # Lightweight inference-only setup script
│   ├── batch_inference.py     # High-throughput batch inference CLI
│   ├── validate_dataset.py    # Dataset integrity and sanity checker
│   ├── shrink_checkpoint.py   # Strips optimizer/scheduler state to produce minimal weights
│   └── generate_roi_visualizations.py # Generates cropped vertebra ROI visualizations
├── src/                       # Core python package source code
│   ├── data/                  # Dataset loaders, transforms, preprocessing, and constants
│   │   ├── constants.py       # Landmark definitions and class constants
│   │   ├── loaders/           # PyTorch Dataset and DataLoader implementations
│   │   ├── preprocessing/     # Data ingestion (CVM API / Label Studio), resizing, splitting
│   │   └── utils/             # Dataset sanity checks and heatmap visualization
│   ├── models/                # CephalometricSwinGCN, SoftArgmax2D, and loss modules
│   │   ├── model.py           # Swin-GCN model and graph attention
│   │   └── losses.py          # AWL, Wing Loss, and Anatomical Graph Loss
│   ├── utils/                 # Metrics, CVM calculator, FTP, MLflow, and verification
│   │   ├── cvm_calculator.py  # Pure geometric percent-based CVM staging engine
│   │   ├── metrics.py         # MRE, SDR, Bland-Altman, Kappa, and clinical metrics
│   │   ├── ftp_utils.py       # Automated remote FTP backup routines
│   │   ├── test_mlflow.py     # MLflow connectivity validation
│   │   └── verify_env.py      # Environment variable validator
│   ├── train.py               # Main training loop with MLflow logging & checkpointing
│   ├── eval.py                # Comprehensive test evaluation and visualization suite
│   └── inference.py           # Single-image and directory inference script
├── tests/                     # Unit and integration test suite
│   ├── test_ablation_no_awl.py
│   ├── test_amp_training.py
│   ├── test_cvm_api_data_source.py
│   ├── test_ftp.py
│   ├── test_landmark_metrics_logging.py
│   └── test_split_artifacts.py
└── .agents/                   # Registered Antigravity skill definitions
    └── skills/vast-ai-runner/ # Automated Vast.ai management skill
```

---

## 4. Environment Setup & Configuration

### Prerequisites
- Python $\ge 3.10$
- Recommended tool: `uv` for fast package and virtual environment management
- Local hardware: CUDA-enabled GPU, Apple Silicon GPU (`mps`), or CPU

### Initial Setup
```bash
# Clone the repository and enter the directory
cd swin-gcn-network

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Environment Variables (`.env`)
Copy `.env.example` to `.env` and configure credentials:
```bash
cp .env.example .env
```

Key environment variables:
- `DATA_SOURCE`: Data ingestion backend (`api` [default] or `label-studio` [legacy]).
- `CVM_API_URL` & `CVM_API_TOKEN`: CVM Backend API endpoints.
- `LABEL_STUDIO_*`: Credentials for legacy Label Studio (used only with `--legacy-label-studio`).
- `VAST_API_KEY`: API token for Vast.ai cloud instances.
- `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`: Remote or local MLflow tracking server.
- `FTP_HOST`, `FTP_PORT`, `FTP_USER`, `FTP_PASSWORD`, `FTP_REMOTE_DIR`: Automated offsite backup for `best.pth` and `metrics.json`.
- `BALANCED_SAMPLING_POWER`: Sampling weight smoothing power (default `0.5`).

---

## 5. Development & Execution Workflows

### 1. Running Tests
Always run the test suite to verify code integrity before committing changes:
```bash
# Run all unit tests
uv run python -m unittest discover tests

# Run a specific test module
uv run python -m unittest tests/test_amp_training.py
```

### 2. End-to-End Training Pipeline (`scripts/train-pipeline.py`)
Executes full data ingestion, resizing, label generation, splitting, model training, and evaluation:
```bash
# Standard end-to-end run
python scripts/train-pipeline.py --epochs 100 --batch-size 8

# Skip downloading raw data (use existing local exports)
python scripts/train-pipeline.py --skip-download --epochs 50

# Train-only (bypass all preprocessing steps 1-4)
python scripts/train-pipeline.py --train-only --epochs 50 --batch-size 8

# Use legacy Label Studio instead of CVM Backend API
python scripts/train-pipeline.py --legacy-label-studio --epochs 100
```

### 3. Remote Cloud GPU Training via Vast.ai (`scripts/vast_runner.py`)
The repository includes a battle-tested runner for executing headless GPU training on Vast.ai:
- **Resilient execution**: Runs in a persistent `tmux` session on the remote host (laptop disconnects or sleep do not interrupt training).
- **Auto-destroy**: The remote instance automatically deletes itself upon training completion or setup failure to prevent unnecessary cloud costs.
- **Dynamic tracking**: Automatically records `INSTANCE_ID` in `.env`.

```bash
# Search available GPUs sorted by price
python3 scripts/vast_runner.py search --gpu "3090" --max-price 0.20 --sort price

# Execute full automated run on RTX 4090 (runs in tmux, auto-destroys when done)
python3 scripts/vast_runner.py run --gpu "RTX 4090" --max-price 0.70 --epochs 100

# Attach to running remote training tmux session
python3 scripts/vast_runner.py attach

# Stream remote logs
python3 scripts/vast_runner.py logs -f

# Download trained weights and evaluation artifacts
python3 scripts/vast_runner.py download

# Manually destroy the instance
python3 scripts/vast_runner.py destroy
```

### 4. Model Evaluation (`src/eval.py`)
Evaluates model weights on the test set, computing MRE, SDR thresholds, and generating visual overlay predictions:
```bash
python src/eval.py \
    --weights artifacts/best.pth \
    --test-img-dir dataset/test/images \
    --test-npz-dir dataset/test/labels \
    --output-dir evaluation/latest \
    --visualize 10
```

### 5. Inference CLI (`src/inference.py` & `scripts/batch_inference.py`)
Supports single images or batch directories, with automatic device acceleration (CUDA / MPS / CPU) and un-letterboxing to native image resolution:
```bash
# Single image inference with clinical CVM stage prediction
python src/inference.py \
    --image path/to/xray.png \
    --weights artifacts/best.pth \
    --output-dir results/ \
    --cvm-calc

# Batch folder inference
python scripts/batch_inference.py \
    --input-dir path/to/raw_images/ \
    --weights artifacts/best.pth \
    --output predictions.json \
    --tta
```

---

## 6. Key Standards & Operational Guardrails for AI Agents

When modifying or refactoring this codebase, strictly follow these rules:

1. **Preserve Anatomical Landmark Semantics**:
   - The ordering and naming of `LANDMARK_CLASSES` in `src/data/constants.py` is immutable.
   - Coordinate tensors output by the model must remain in normalized scale $[0, 1]$ with shape `[B, 13, 2]` where coordinate pair is `[x, y]`.
   - Missing or unannotated landmarks must be represented by `[-1.0, -1.0]` with valid masks checking `(target[:, :, 0] >= 0) & (target[:, :, 1] >= 0)`.

2. **Data Split Determinism & No Data Leakage**:
   - `dataset/val_image_ids.json` and `dataset/test_image_ids.json` fix the validation and test cohorts across experiments and ablations.
   - Never randomly re-partition test or validation sets without updating the tracking JSON artifacts.
   - Images flagged with `review_needed` in the API/export must always be excluded from training datasets.

3. **Loss Function Differentiability**:
   - The link between heatmaps and coordinates is `SoftArgmax2D`.
   - Do NOT use non-differentiable `argmax` (e.g. `np.unravel_index` or `torch.argmax`) during the forward training pass, as this breaks gradient backpropagation from `WingLoss` and `AnatomicalGraphLoss` to the U-Net and Swin backbone.

4. **Hardware Portability**:
   - Always resolve devices dynamically (`cuda` if available, else `mps` on macOS, else `cpu`).
   - Do not hardcode `.cuda()` calls; use `tensor.to(device)`.

5. **Cloud Cost Protection**:
   - When provisioning Vast.ai instances programmatically, always ensure error handlers preserve the auto-destroy prompt or flag to prevent orphaned billable instances.

6. **Preserve Documentation & Comments**:
   - Adhere to the repository's guidelines: maintain documentation integrity and preserve existing comments and docstrings.
