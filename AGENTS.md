# AGENTS.md — Developer & AI Agent Operations Manual

Welcome to **CVM Swin-GCN Network** (`cvm-swin-gcn`). This document serves as the operational guide and technical specification for AI agents (and human engineers) working in this repository.

---

## 1. Project Overview & Clinical Context

### Objective
This project implements and trains an advanced deep learning framework for automated **Cervical Vertebral Maturation (CVM)** staging on lateral cephalometric radiographs. CVM assessment evaluates the biological skeletal maturation of pediatric and orthodontic patients by classifying morphological changes in the cervical vertebrae (C2, C3, and C4) into stages **CS1 through CS6**—eliminating the need for supplementary ionizing hand-wrist radiographs.

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

> **CRITICAL INVARIANT**: Never alter or reorder these indices or strings. Downstream morphometric calculations (`src/utils/cvm_calculator.py`), evaluation modules (`src/utils/metrics.py`), ROI transformations (`src/utils/roi_utils.py`), and precomputed label arrays depend strictly on this 0–12 sequence.

### Clinical Morphological Features
CVM staging depends on two distinct morphometric characteristics evaluated from the 13 landmarks:
1. **Inferior Border Concavities**: Formation of concavities at the inferior borders of C2, C3, and C4 (determining stages CS1 to CS3):
   - **CS1**: Inferior borders of C2, C3, and C4 are flat (concavity $\le 2.5\%$).
   - **CS2**: Concavity present at the inferior border of C2; C3 and C4 remain flat.
   - **CS3**: Concavities present at the inferior borders of both C2 and C3; C4 remains flat.
2. **Vertebral Body Shape Ratios**: Morphological shape transitions of the bodies of C3 and C4 (determining stages CS4 to CS6):
   - **CS4**: Concavities present at C2, C3, and C4; vertebral bodies of C3 and C4 are horizontally rectangular.
   - **CS5**: Vertebral bodies of C3 and C4 are square (height-to-width ratio $\approx 1.0$).
   - **CS6**: Vertebral bodies of C3 and C4 are vertically rectangular (height exceeds width).

### Anatomical Reference Normalization ($L_{\text{ref}}$)
To eliminate scale variance across differing cephalostat sensor dimensions and patient physical sizes, landmark errors are normalized to the patient's anatomical reference length:
$$L_{\text{ref}} = \|C3\_AI - C3\_PI\|_2$$
where $C3\_AI$ (index 7) and $C3\_PI$ (index 5) define the ground-truth C3 inferior endplate width. All evaluation errors are reported as normalized percentages ($\% L_{\text{ref}}$) and successful detection rates (SDR at $\le 2\%, \le 3\%, \le 5\%, \le 8\%, \le 10\% L_{\text{ref}}$).

---

## 2. Technical Architecture & Design

The repository supports two operational paradigms: **Single-Stage Baseline** and **Two-Stage Coarse-to-Fine Pipeline** (active by default on the `two-stage` branch).

### A. Core Model (`CephalometricSwinGCN` in `src/models/model.py`)
1. **Backbone**: Swin Transformer Base (`swin_base_patch4_window7_224` via `timm`) operating at $640 \times 640$ resolution, outputting multi-scale hierarchical feature maps across 4 stages.
2. **Decoder**: U-Net feature pyramid decoder with transposed convolutions and skip connections (`UNetUpBlock`), projecting fused hierarchical representations to 13 heatmap channels.
3. **Heatmap Normalization**: Sigmoid activation strictly bounding output heatmaps to $[0, 1]$, matching ground-truth 2D Gaussian targets.
4. **Coordinate Head**: Fully differentiable `SoftArgmax2D` with learnable per-landmark temperature parameters $\tau_i$, computing global spatial expectations over normalized coordinate grids $[0, 1] \times [0, 1]$.
5. **Graph Adjacency Modeling**: Built-in adjacency buffer (`adj_matrix`) modeling intra-vertebra geometry (superior, inferior, anterior, posterior borders, and concavity dips) and inter-vertebra spinal alignment ($C2 \leftrightarrow C3$ and $C3 \leftrightarrow C4$).

### B. Two-Stage Coarse-to-Fine System (`src/models/two_stage.py`)
To achieve sub-millimeter clinical precision on high-resolution radiographs, the two-stage cascade isolates cervical vertebrae from background cranial structures:
1. **Stage 1 (Coarse Global Detector)**: Operates on full $640 \times 640$ radiographs to localize all 13 landmarks and establish the cervical spine ROI.
2. **Isotropic Square ROI Extraction (`src/utils/roi_utils.py`)**:
   - Computes an isotropic, strictly square bounding box $[u_1, v_1, u_2, v_2, D]$ bounding C2–C4 with configurable margin ratio (`margin_ratio=0.18`, minimum size $180\text{ px}$).
   - Employs zero/reflection padding (`crop_roi_with_padding`) to guarantee no aspect ratio distortion or coordinate clipping.
   - Provides coordinate forward mapping ($[0, 1] \to \text{ROI} \to [0, 1]$) and exact inverse half-pixel re-projection back to full image canvas space (`inverse_transform_coords`).
3. **Anatomical Vertebra Height Prior**:
   - Verifies physiological validity of predicted cervical geometry: $H_{C4} \ge 0.75 \cdot H_{C3}$.
   - If Stage 1 predictions collapse C4 into C3 (due to low contrast or thick soft tissue), the prior applies the reliable height of C3 downwards to ensure the ROI crop covers the complete inferior border of C4 (`apply_anatomical_height_prior`).
4. **Stage 2 (Fine Vertebrae Refiner)**:
   - High-resolution `CephalometricSwinGCN` operating directly on $448 \times 448$ (or $672 \times 672$) vertebra ROI crops.
   - Trained with empirical bounding-box jitter ($\pm 10\%\text{ shift}, \pm 5\%\text{ scale}$) simulating Stage 1 bounding box variance.
   - Heatmap Gaussian $\sigma$ sharpened to $2.5$ (in ROI space) to eliminate centroid blurring and maximize peak precision.
5. **Multi-Tier Cascaded Evaluation (`src/eval_two_stage.py`)**:
   - **Tier 1 (Stage 1 Alone)**: Standalone coarse model performance on full cephalograms.
   - **Tier 2 (Stage 2 Oracle)**: Theoretical upper bound using ground-truth bounding boxes.
   - **Tier 3 (Two-Stage Cascade)**: Full end-to-end pipeline (Stage 1 box $\to$ Stage 2 $\to$ Inverse re-projection).
   - **Tier 4 (2-Pass Iterative Refinement)**: Optional second pass re-centering the ROI around initial Stage 2 predictions.

---

## 3. Loss Formulations & Multi-Task Balancing

### Total Loss Function (`src/models/losses.py`)
$$\mathcal{L}_{\text{total}} = \lambda_{\text{hm}} \mathcal{L}_{\text{AWL}}(\hat{H}, H^*) + \gamma(t) \cdot \lambda_{\text{cd}} \mathcal{L}_{\text{Wing}}(\hat{C}, C^*) + \gamma(t) \cdot \lambda_{\text{graph}} \mathcal{L}_{\text{Graph}}(\hat{C}, C^*)$$

- **Adaptive Wing Loss (AWL)**: Heatmap loss adapting gradients dynamically near Gaussian peaks while smoothing background errors ($\omega=14.0$, $\theta=0.5$, $\epsilon=1.0$, $\alpha=2.1$).
- **Wing Loss**: Differentiable coordinate loss operating on $[0, 1]$ coordinates ($\omega=10.0$, $\epsilon=2.0$), emphasizing sub-pixel accuracy while remaining robust to large outlier errors.
- **Anatomical Graph Loss**: Penalizes deviations in pairwise Euclidean distances between anatomically connected landmarks relative to ground truth.
- **Warmup Schedule $\gamma(t)$**: Ramps coordinate and graph loss weights from $0.0$ to $1.0$ over `WARMUP_COORD_EPOCHS` (default: 5) to allow heatmaps to stabilize coarse localization before strong geometric constraints engage.
- **Default Loss Balances**:
  - Stage 1: $\lambda_{\text{hm}} = 1.0$, $\lambda_{\text{cd}} = 5.0$, $\lambda_{\text{graph}} = 1.0$, $\lambda_{\text{morph}} = 0.0$.
  - Stage 2: $\lambda_{\text{hm}} = 50.0$, $\lambda_{\text{cd}} = 2.0$, $\lambda_{\text{graph}} = 0.0$, $\lambda_{\text{morph}} = 0.0$.

### Validation-Guided Error-Adaptive Landmark Loss Weights (`src/train.py`)
To overcome heterogeneous localization difficulty across cervical landmarks, the loss function dynamically weights each landmark based on empirical residuals:
- **Tier 1 Boosted (Concavities & High-Residual Landmarks)**:
  - `C4_IC` (Index 11): **2.5×** multiplier (critical deep concavity)
  - `C4_PI` (Index 10): **2.4×** multiplier (inferior corner)
  - `C4_AI` (Index 12): **2.2×** multiplier (inferior corner)
  - `C3_IC` (Index 6):  **2.2×** multiplier (vital diagnostic concavity deciding CS2/CS3/CS4)
  - `C2_IC` (Index 1):  **2.2×** multiplier (vital diagnostic concavity deciding CS1/CS2)
- **Tier 2 Boosted (Challenging Corners)**:
  - `C4_PS` (Index 8): **1.8×** multiplier
  - `C3_PS` (Index 3): **1.8×** multiplier
  - `C3_AS` (Index 4): **1.6×** multiplier
  - `C2_AI` (Index 2): **1.5×** multiplier
- **Baseline (Well-Converged Landmarks)**:
  - `C2_PI` (0), `C3_PI` (5), `C3_AI` (7), `C4_AS` (9): **1.0×** multiplier

---

## 4. Directory Layout

```
swin-gcn-network/
├── AGENTS.md                  # This file (AI agent instructions & operational manual)
├── README.md                  # Project overview and clinical background
├── pyproject.toml             # Project build configuration and dependencies
├── requirements.txt           # Pinned python requirements
├── uv.lock                    # Locked dependency resolutions
├── .env.example               # Template for environment variables and secrets
├── .python-version            # Python version specification (3.10)
├── artifacts/                 # Trained model checkpoints
│   ├── best.pth               # Production / Stage 1 best weights
│   ├── stage1_best.pth        # Stage 1 coarse detector weights
│   └── stage2_best.pth        # Stage 2 fine ROI refiner weights
├── data/                      # Raw and intermediate data
│   ├── raw/images/            # Native-resolution source cephalograms
│   ├── resized/images/        # Preprocessed 640x640 images
│   ├── labels/                # Precomputed .npz ground truth arrays
│   └── exports/               # Ingested JSON annotations from API/Label Studio
├── dataset/                   # Local train/val/test splits
│   ├── train/                 # images/ and labels/ (.npz)
│   ├── val/                   # images/ and labels/ (.npz)
│   ├── test/                  # images/ and labels/ (.npz)
│   ├── val_image_ids.json     # Deterministic validation set image IDs
│   └── test_image_ids.json    # Deterministic test set image IDs
├── evaluation/                # Evaluation output reports, metric JSONs, and clinical plots
│   ├── predictions_light/     # Lightweight landmark JSONs for web/GUI consumption
│   └── two_stage/             # Multi-tier two-stage evaluation outputs
├── scripts/                   # Pipeline automation, Vast.ai cloud, and utility CLI scripts
│   ├── train-pipeline.py      # Master end-to-end pipeline (download -> train -> eval)
│   ├── train_two_stage_pipeline.py # Dedicated Two-Stage Coarse-to-Fine training pipeline
│   ├── vast_runner.py         # Vast.ai GPU discovery, provisioning, tmux run & auto-destroy
│   ├── download_checkpoint.py # Resumable streaming checkpoint download from MLflow/MinIO
│   ├── batch_inference.py     # High-throughput batch inference CLI (single & two-stage)
│   ├── shrink_checkpoint.py   # Strips optimizer states to create lightweight deployment weights
│   ├── validate_dataset.py    # Dataset integrity and sanity checker
│   ├── check_augmentations.py # Data augmentation visualizer
│   ├── generate_roi_visualizations.py # Generates cropped vertebra ROI visualizations
│   ├── setup.sh               # Environment bootstrap for remote training instances
│   └── setup-inference.sh     # Lightweight inference-only setup script
├── src/                       # Core python package source code
│   ├── data/
│   │   ├── constants.py       # Landmark definitions, classes, and adjacency matrices
│   │   ├── loaders/
│   │   │   ├── dataset.py     # CephalometricDataset with caching and normalization
│   │   │   ├── dataloader.py  # Dataloader factory with balanced class sampling
│   │   │   ├── roi_dataset.py # ROI crop dataset with empirical bounding-box jitter
│   │   │   └── transforms.py  # Robust Albumentations pipeline
│   │   ├── preprocessing/
│   │   │   ├── download_data.py   # CVM Backend API and Label Studio data ingestion
│   │   │   ├── resize_images.py   # High-quality aspect-ratio preserving resizing
│   │   │   ├── generate_labels.py # 2D Gaussian heatmap & GCN label generator
│   │   │   └── split_dataset.py   # Stratified splitting with fixed ID JSONs
│   │   └── utils/
│   │       ├── sanity_checks.py       # Ground truth label validation
│   │       ├── visualize_coords.py    # Coordinate overlay visualizer
│   │       └── visualize_heatmaps.py  # Multi-channel heatmap visualizer
│   ├── models/
│   │   ├── model.py           # CephalometricSwinGCN, UNetUpBlock, SoftArgmax2D
│   │   ├── two_stage.py       # TwoStageCephalometricSwinGCN cascade module
│   │   └── losses.py          # AWL, Wing Loss, AnatomicalGraphLoss, MorphometricShapeLoss
│   ├── utils/
│   │   ├── cvm_calculator.py  # Pure geometric percent-based CVM staging engine
│   │   ├── metrics.py         # Normalized error suite, SDR, Bland-Altman, Kappa, ICC(2,1)
│   │   ├── roi_utils.py       # Isotropic square ROI extraction, padding, and inverse mapping
│   │   ├── test_mlflow.py     # MLflow connection test suite
│   │   ├── ftp_utils.py       # Remote FTP artifact backup utilities
│   │   └── verify_env.py      # Environment variable validator
│   ├── train.py               # Stage 1 training engine with LLRD and MLflow logging
│   ├── train_stage2.py        # Stage 2 ROI fine refiner training engine
│   ├── eval.py                # Single-stage evaluation engine and visualization suite
│   ├── eval_two_stage.py      # 4-tier Two-Stage cascaded evaluation suite
│   └── inference.py           # Single-image inference CLI with CVM staging
├── tests/                     # 86 comprehensive unit and integration tests
└── .agents/                   # Registered Antigravity skill definitions
    └── skills/vast-ai-runner/ # Automated Vast.ai management skill
```

---

## 5. Environment Setup & Configuration

### Prerequisites
- Python $\ge 3.10$
- Recommended tool: `uv` for fast package and virtual environment management
- Local hardware: CUDA GPU, Apple Silicon GPU (`mps`), or CPU

### Initial Setup
```bash
# Clone the repository and enter the directory
cd swin-gcn-network

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate

# Install project dependencies
uv pip install -r requirements.txt
```

### Environment Variables (`.env`)
Copy `.env.example` to `.env` and configure appropriate credentials:
```bash
cp .env.example .env
```

Key environment variables:
- `DATA_SOURCE`: Data ingestion backend (`api` [default] or `label-studio` [legacy]).
- `CVM_API_URL` & `CVM_API_TOKEN`: CVM Backend API endpoints.
- `VAST_API_KEY`: API token for Vast.ai cloud instances.
- `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_EXPERIMENT_NAME`: Remote MLflow tracking server backed by MinIO.
- `FTP_HOST`, `FTP_PORT`, `FTP_USER`, `FTP_PASSWORD`, `FTP_REMOTE_DIR`: Automated offsite backup for weights and evaluation metrics.
- `BALANCED_SAMPLING_POWER`: Sampling weight smoothing power (default `0.5`).

---

## 6. Development & Execution Workflows

### 1. Running Unit & Integration Tests
Always run the test suite to verify code integrity before committing changes:
```bash
# Run all 86 unit and integration tests
uv run python -m unittest discover tests

# Run specific test modules
uv run python -m unittest tests/test_two_stage_model.py
uv run python -m unittest tests/test_two_stage_cascade.py
uv run python -m unittest tests/test_normalized_metrics_suite.py
```

### 2. End-to-End Master Training Pipeline (`scripts/train-pipeline.py`)
Executes full data ingestion, resizing, label generation, splitting, model training, and evaluation:
```bash
# Standard Two-Stage Coarse-to-Fine Pipeline (Default on two-stage branch)
python scripts/train-pipeline.py --epochs 100 --stage2-epochs 60 --batch-size 8

# Single-Stage Baseline (trains coarse model only)
python scripts/train-pipeline.py --single-stage --epochs 100 --batch-size 8

# Skip downloading raw data (use existing local exports)
python scripts/train-pipeline.py --skip-download --epochs 100 --stage2-epochs 60

# Train-only (bypass preprocessing steps 1-4)
python scripts/train-pipeline.py --train-only --epochs 100 --stage2-epochs 60

# Train Stage 2 only (using an existing Stage 1 checkpoint)
python scripts/train-pipeline.py --train-only --skip-stage1 --stage1-weights artifacts/stage1_best.pth --stage2-epochs 60
```

### 3. Dedicated Two-Stage Pipeline (`scripts/train_two_stage_pipeline.py`)
A streamlined entrypoint dedicated to Stage 1 $\to$ Stage 2 $\to$ 4-tier cascaded evaluation:
```bash
# Full two-stage training and evaluation
python scripts/train_two_stage_pipeline.py --stage1-epochs 100 --stage2-epochs 60 --batch-size 8

# Fast run skipping Stage 1 if already trained
python scripts/train_two_stage_pipeline.py --skip-stage1 --stage1-weights artifacts/stage1_best.pth --stage2-epochs 60
```

### 4. Remote Cloud GPU Training via Vast.ai (`scripts/vast_runner.py`)
Headless GPU training on Vast.ai with persistent `tmux` sessions, auto-destroy on finish/error, and automatic credential synchronization:
```bash
# Search available GPUs sorted by price ($/hr)
python3 scripts/vast_runner.py search --gpu "3090" --max-price 0.25 --sort price

# Search high-performance RTX 4090 offers
python3 scripts/vast_runner.py search --gpu "4090" --max-price 0.70 --sort dlperf

# Launch automated training run on RTX 4090 (runs in tmux, auto-destroys on finish)
python3 scripts/vast_runner.py run --gpu "RTX 4090" --max-price 0.70 --epochs 100

# Attach to running remote training tmux session
python3 scripts/vast_runner.py attach

# Stream remote training logs
python3 scripts/vast_runner.py logs -f

# Download trained weights and evaluation artifacts
python3 scripts/vast_runner.py download

# Manually destroy the instance
python3 scripts/vast_runner.py destroy
```

### 5. Checkpoint Management & Streaming Download (`scripts/download_checkpoint.py`)
Directly streams checkpoint artifacts from remote MLflow / MinIO storage with real-time transfer progress:
```bash
# Stream Stage 1 checkpoint from specific MLflow run ID
python scripts/download_checkpoint.py \
    --run-id 2d4d452398b14c688f51ac4431d40698 \
    --artifact-path checkpoints/best.pth \
    --dest artifacts/stage1_best.pth

# Strip optimizer states to produce clean lightweight weights (~100 MB)
python scripts/shrink_checkpoint.py \
    --input artifacts/best.pth \
    --output artifacts/best_clean.pth
```

### 6. Model Evaluation Suites

#### A. Two-Stage Cascaded Evaluation (`src/eval_two_stage.py`)
Runs 4-tier comparative evaluation on validation or test sets:
```bash
python src/eval_two_stage.py \
    --stage1-weights artifacts/stage1_best.pth \
    --stage2-weights artifacts/stage2_best.pth \
    --test-img-dir dataset/test/images \
    --test-npz-dir dataset/test/labels \
    --output-dir evaluation/two_stage_test \
    --visualize 10 \
    --refine-2pass
```

#### B. Single-Stage Evaluation (`src/eval.py`)
Evaluates single-stage model weights on test datasets:
```bash
python src/eval.py \
    --weights artifacts/best.pth \
    --test-img-dir dataset/test/images \
    --test-npz-dir dataset/test/labels \
    --output-dir evaluation/single_stage \
    --visualize 10
```

### 7. Inference Workflows

#### A. Batch Inference CLI (`scripts/batch_inference.py`)
Supports high-throughput directory inference with optional two-stage cascade and lightweight landmark export:
```bash
# Two-stage cascaded batch inference
python scripts/batch_inference.py \
    --images-dir data/raw/images \
    --stage1-weights artifacts/stage1_best.pth \
    --stage2-weights artifacts/stage2_best.pth \
    --output-json evaluation/predictions.json \
    --export-light \
    --compute-metrics

# Single-stage batch inference
python scripts/batch_inference.py \
    --images-dir data/raw/images \
    --weights artifacts/best.pth \
    --output-json evaluation/predictions.json
```

#### B. Single-Image Inference CLI (`src/inference.py`)
Performs inference on a single radiograph, un-letterboxes to native resolution, and prints CVM stage diagnosis:
```bash
python src/inference.py \
    --image path/to/sample.png \
    --weights artifacts/best.pth \
    --output-dir results/ \
    --cvm-calc
```

---

## 7. Evaluation Metrics & Clinical Validation Suite (`src/utils/metrics.py`)

All evaluations generate publication-grade statistical reports covering:

1. **Normalized Radial Errors**:
   - Error expressed as $\% L_{\text{ref}}$ (where $L_{\text{ref}} = \|C3\_AI - C3\_PI\|_2$).
   - Successful Detection Rate (SDR) thresholds: $\le 2.0\%, \le 2.5\%, \le 3.0\%, \le 5.0\%, \le 8.0\%, \le 10.0\% L_{\text{ref}}$.
   - Parametric and non-parametric statistics: Mean, Standard Deviation, Median, Interquartile Range (IQR: 25th–75th percentile), P90.
2. **Directional Inferior Border Projections**:
   - Decomposes error vectors for inferior border landmarks (`IC`, `AI`, `PI`) into:
     - **Tangential Component ($e_\parallel$)**: Error along the inferior endplate axis.
     - **Normal / Depth Component ($e_\perp$)**: Error perpendicular to the endplate axis (measuring true concavity depth deviation).
3. **Clinical Agreement & Reliability**:
   - **Bland-Altman Agreement**: Mean bias, 95% Limits of Agreement (LoA), Pearson $r$, and Spearman $\rho$ for Concavity percentages ($C\%$) and Shape Indices ($SI$).
   - **Intraclass Correlation Coefficient**: Two-way random single-rater absolute agreement $\text{ICC}(2, 1)$ for morphometric indices.
   - **Cohen's Kappa**: Unweighted ($\kappa$) and Quadratic Weighted ($\kappa_w$) for CVM stages and notch classifications.
   - **Paired Statistical Tests**: Wilcoxon signed-rank test and McNemar test for cascaded vs standalone comparisons.
4. **Confidence Intervals**:
   - Wilson score 95% confidence intervals for binary classification rates (SDR, notch detection).
   - Image-level bootstrap 95% confidence intervals (resampling full radiographs to account for intra-image landmark correlation).
5. **Inference Latency & Profiling**:
   - Profiling of model execution latency per image (ms) and batch throughput (FPS).

---

## 8. Key Standards & Operational Guardrails for AI Agents

When modifying or refactoring this codebase, strictly adhere to the following rules:

1. **Preserve Anatomical Landmark Semantics**:
   - The ordering and naming of `LANDMARK_CLASSES` in `src/data/constants.py` is immutable.
   - Coordinate tensors output by models must remain in normalized scale $[0, 1]$ with shape `[B, 13, 2]` where coordinate pair is `[x, y]`.
   - Missing or unannotated landmarks must be represented by `[-1.0, -1.0]` with valid masks checking `(target[:, :, 0] >= 0) & (target[:, :, 1] >= 0)`.

2. **Data Split Determinism & Zero Data Leakage**:
   - `dataset/val_image_ids.json` and `dataset/test_image_ids.json` define fixed validation and test cohorts across all experiments and ablations.
   - Never randomly re-partition test or validation sets without updating the tracking JSON artifacts.
   - Images flagged with `review_needed` or `excluded` in data exports must strictly be excluded from training datasets.

3. **Loss Function Differentiability**:
   - The link between heatmaps and coordinates is `SoftArgmax2D`.
   - Do NOT use non-differentiable `argmax` (e.g. `np.unravel_index` or `torch.argmax`) during the forward training pass, as this breaks gradient backpropagation from `WingLoss` and `AnatomicalGraphLoss` to the U-Net and Swin backbone.

4. **Half-Pixel Alignment & Coordinate Transforms**:
   - Always preserve exact half-pixel coordinate conventions between full canvas and ROI crops:
     $$x_{\text{full}} = u_1 + x_{\text{roi}} \cdot D, \quad y_{\text{full}} = v_1 + y_{\text{roi}} \cdot D$$
   - In normalized space:
     $$x_{\text{full, norm}} = \frac{u_1 + x_{\text{roi, norm}} \cdot D}{W_{\text{canvas}}}$$
   - Use `inverse_transform_coords` from `src/utils/roi_utils.py` for all inverse re-projections.

5. **Hardware Portability**:
   - Always resolve devices dynamically using `resolve_device()` (`cuda` if available, else `mps` on macOS, else `cpu`).
   - Do not hardcode `.cuda()` calls; use `.to(device)`.

6. **Cloud Cost Protection**:
   - When provisioning Vast.ai instances programmatically, always ensure error handlers preserve the auto-destroy prompt or flag to prevent orphaned billable instances.

7. **Preserve Documentation & Comments**:
   - Adhere to repository guidelines: maintain documentation integrity and preserve existing comments, docstrings, and type hints across all modifications.
