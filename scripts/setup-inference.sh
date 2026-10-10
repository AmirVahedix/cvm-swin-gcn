#!/bin/bash
# ==============================================================================
# Cephalometric Swin-GCN Network: Vast.ai & Cloud GPU Setup for Inference
#
# Functions:
# 1. Configures system dependencies (libgl1, tmux, git, curl, unzip, etc.)
# 2. Clones or safely updates the repository from GitHub using .env credentials
# 3. Detects GPU architecture & installs CUDA-matched PyTorch (CUDA 12/11/Blackwell)
# 4. Installs ALL project dependencies via uv / pip
# 5. Executes a GPU pre-flight verification test
# 6. Optionally triggers batch inference on a directory of images (--run-inference)
# ==============================================================================

set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Defaults ---
WORKSPACE="/workspace"
REPO_NAME="cvm-swin-gcn"
PROJECT_DIR="${WORKSPACE}/${REPO_NAME}"

ONLY_PULL=false
AUTO_CONFIRM=false
BRANCH_ARG=""

RUN_INFERENCE=false
IMAGES_DIR="${WORKSPACE}/images"
OUTPUT_JSON="${WORKSPACE}/predictions.json"
WEIGHTS_PATH=""
BATCH_SIZE=16
SAVE_CSV=false
EXTRA_INFERENCE_ARGS=()

# --- Parse Command-Line Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only-pull)
            ONLY_PULL=true
            shift
            ;;
        -y|--yes|--force)
            AUTO_CONFIRM=true
            shift
            ;;
        -b|--branch)
            BRANCH_ARG="$2"
            shift 2
            ;;
        --run-inference)
            RUN_INFERENCE=true
            shift
            ;;
        -i|--images-dir)
            IMAGES_DIR="$2"
            shift 2
            ;;
        -o|--output-json)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        -w|--weights)
            WEIGHTS_PATH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --save-csv)
            SAVE_CSV=true
            shift
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                EXTRA_INFERENCE_ARGS+=("$1")
                shift
            done
            break
            ;;
        *)
            shift
            ;;
    esac
done

echo -e "\n${BLUE}=================================================================${NC}"
echo -e "${BLUE}🚀 Cephalometric Swin-GCN: Inference Environment Setup${NC}"
echo -e "${BLUE}=================================================================${NC}\n"

# --- Step 1: Locate and Load .env Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE=""

for candidate in "${SCRIPT_DIR}/.env" "${WORKSPACE}/.env" "${PROJECT_DIR}/.env" "$(pwd)/.env"; do
    if [ -f "$candidate" ]; then
        ENV_FILE="$candidate"
        break
    fi
done

if [ -n "$ENV_FILE" ]; then
    echo -e "${BLUE}ℹ️ Loading environment variables from ${ENV_FILE}...${NC}"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE" 2>/dev/null || {
        while IFS= read -r line || [ -n "$line" ]; do
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${line// }" ]] && continue
            export "$line" 2>/dev/null || true
        done < "$ENV_FILE"
    }
    set +a
else
    echo -e "${YELLOW}⚠️ .env file not found. Relying on current shell environment variables...${NC}"
fi

TARGET_BRANCH="${BRANCH_ARG:-${GIT_BRANCH:-${BRANCH:-main}}}"
echo -e "${BLUE}🌿 Target Git Branch: ${TARGET_BRANCH}${NC}"

# --- Mode: Only Pull Changes ---
if [ "$ONLY_PULL" = true ]; then
    echo -e "${BLUE}=== Mode: Pulling Latest Repository Changes ===${NC}"
    TARGET_DIR="$PROJECT_DIR"
    if [ ! -d "$TARGET_DIR" ] && [ -d "${SCRIPT_DIR}/../.git" ]; then
        TARGET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi

    if [ -d "$TARGET_DIR" ]; then
        cd "$TARGET_DIR" || exit 1
        echo -e "${BLUE}🔄 Pulling from origin/${TARGET_BRANCH}...${NC}"
        git fetch origin
        git checkout "$TARGET_BRANCH" 2>/dev/null || git checkout -b "$TARGET_BRANCH" "origin/$TARGET_BRANCH"
        git pull origin "$TARGET_BRANCH"
        echo -e "${GREEN}✅ Repository updated successfully (user-uploaded files preserved).${NC}"
    else
        echo -e "${RED}❌ Project directory not found at ${TARGET_DIR}.${NC}"
        exit 1
    fi
    exit 0
fi

# --- Step 2: System Base Dependencies (apt-get) ---
echo -e "\n${BLUE}⚙️ Step 1/5: Installing system libraries (OpenCV, git, tmux, curl)...${NC}"
if command -v apt-get &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq \
        tmux \
        sshpass \
        zip \
        unzip \
        pv \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libxcb1 \
        libx11-xcb1 \
        build-essential > /dev/null 2>&1 || true
    echo -e "${GREEN}✅ System dependencies installed.${NC}"
fi

# --- Step 3: Git Clone or Safe Update ---
echo -e "\n${BLUE}📦 Step 2/5: Setting up repository code...${NC}"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE" || exit 1

if [ -z "${GITHUB_TOKEN// }" ] || [ -z "${GITHUB_USER// }" ]; then
    echo -e "${RED}❌ GITHUB_TOKEN or GITHUB_USER missing in environment/.env${NC}"
    echo -e "${YELLOW}Please ensure GITHUB_TOKEN and GITHUB_USER are defined in .env before running.${NC}"
    exit 1
fi

REPO_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

if [ -d "$PROJECT_DIR" ]; then
    echo -e "${GREEN}ℹ️ Existing repository found at ${PROJECT_DIR}.${NC}"
    cd "$PROJECT_DIR" || exit 1
    echo -e "${BLUE}🔄 Fetching updates for branch '${TARGET_BRANCH}' (preserving local images & weights)...${NC}"
    git fetch origin
    git checkout "$TARGET_BRANCH" 2>/dev/null || git checkout -b "$TARGET_BRANCH" "origin/$TARGET_BRANCH"
    # Pull without hard reset so user-uploaded weights and images in artifacts/ or workspace are NOT deleted!
    git pull origin "$TARGET_BRANCH" || true
else
    echo -e "${BLUE}📥 Cloning repository from GitHub on branch '${TARGET_BRANCH}'...${NC}"
    if ! git clone -b "$TARGET_BRANCH" "$REPO_URL" "$PROJECT_DIR" 2>&1 | sed "s|${GITHUB_TOKEN}|***HIDDEN_TOKEN***|g"; then
        echo -e "${RED}❌ Failed to clone repository. Check your GitHub credentials and repo permissions.${NC}"
        exit 1
    fi
    cd "$PROJECT_DIR" || exit 1
fi

# Place .env inside project directory for internal scripts
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ] && [ "$ENV_FILE" != "${PROJECT_DIR}/.env" ]; then
    cp "$ENV_FILE" "${PROJECT_DIR}/.env" 2>/dev/null || true
fi

# Clean up any pinned python version in pyproject that may conflict with system Python
rm -f "${PROJECT_DIR}/.python-version" 2>/dev/null || true
if [ -f "${PROJECT_DIR}/pyproject.toml" ]; then
    sed -i 's/requires-python = ">=[0-9]*\.[0-9]*"/requires-python = ">=3.8"/g' "${PROJECT_DIR}/pyproject.toml" 2>/dev/null || true
    sed -i 's/torch>=[0-9]*\.[0-9]*\.[0-9]*/torch>=2.0.0/g' "${PROJECT_DIR}/pyproject.toml" 2>/dev/null || true
fi

# --- Step 4: Python, UV, and CUDA-Matched PyTorch Setup ---
echo -e "\n${BLUE}🐍 Step 3/5: Setting up Python and UV Package Manager...${NC}"
export UV_PYTHON_DOWNLOADS=never
curl -LSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1 || true
export PATH="$HOME/.cargo/bin:$PATH"
grep -qxF 'export PATH="$HOME/.cargo/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null || echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bashrc" 2>/dev/null || true

PYTHON_BIN=""
if command -v python3 &> /dev/null; then
    PYTHON_BIN=$(command -v python3)
elif command -v python &> /dev/null; then
    PYTHON_BIN=$(command -v python)
else
    echo -e "${RED}❌ Python binary not found in PATH!${NC}"
    exit 1
fi
echo -e "${BLUE}Using Python: ${PYTHON_BIN} ($("$PYTHON_BIN" --version 2>&1))${NC}"

# Detect GPU & CUDA
echo -e "\n${BLUE}🔍 Step 4/5: Detecting GPU hardware and CUDA runtime...${NC}"
GPU_MODEL=""
CUDA_VERSION=""
if command -v nvidia-smi &> /dev/null; then
    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -i "CUDA Version" | sed -E 's/.*CUDA Version:[[:space:]]*([0-9]+\.[0-9]+).*/\1/' | head -n 1)
fi

if [ -z "$CUDA_VERSION" ] && command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -i "release" | sed -E 's/.*release ([0-9]+\.[0-9]+).*/\1/' | head -n 1)
fi

if [[ "$GPU_MODEL" =~ "5090" ]] || [[ "$GPU_MODEL" =~ "RTX 50" ]]; then
    echo -e "${GREEN}⚡ Detected NVIDIA RTX 50-series: ${GPU_MODEL} (Blackwell)${NC}"
    CUDA_TAG="cu128"
elif [ -n "$CUDA_VERSION" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d'.' -f2)
    echo -e "${GREEN}✅ Detected GPU: ${GPU_MODEL:-Unknown} | Host CUDA: ${CUDA_VERSION}${NC}"

    if [ "$CUDA_MAJOR" -ge 12 ]; then
        if [ "$CUDA_MINOR" -ge 8 ]; then
            CUDA_TAG="cu128"
        elif [ "$CUDA_MINOR" -ge 6 ]; then
            CUDA_TAG="cu126"
        elif [ "$CUDA_MINOR" -ge 4 ]; then
            CUDA_TAG="cu124"
        else
            CUDA_TAG="cu121"
        fi
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        CUDA_TAG="cu118"
    else
        CUDA_TAG="cpu"
    fi
else
    echo -e "${YELLOW}⚠️ No NVIDIA GPU detected. Falling back to CPU wheel.${NC}"
    CUDA_TAG="cpu"
fi

PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
echo -e "${BLUE}🎯 Selected PyTorch Index: ${PYTORCH_INDEX_URL}${NC}"

# Install CUDA-matched PyTorch if not already matching
echo -e "${BLUE}⬇️ Verifying / Installing PyTorch and torchvision (${CUDA_TAG})...${NC}"
if ! "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${BLUE}Installing PyTorch from ${PYTORCH_INDEX_URL}...${NC}"
    uv pip install --system --python "$PYTHON_BIN" --no-python-downloads --break-system-packages torch torchvision "numpy<2" --index-url "${PYTORCH_INDEX_URL}" 2>/dev/null || {
        "$PYTHON_BIN" -m pip install --break-system-packages torch torchvision "numpy<2" --index-url "${PYTORCH_INDEX_URL}"
    }
fi

# --- Install ALL Project Dependencies ---
echo -e "\n${BLUE}📦 Installing all project dependencies (timm, albumentations, opencv, etc.)...${NC}"
if [ -f "${PROJECT_DIR}/pyproject.toml" ]; then
    cd "$PROJECT_DIR" || exit 1
    if ! uv pip install --system --python "$PYTHON_BIN" --no-python-downloads --break-system-packages --extra-index-url "${PYTORCH_INDEX_URL}" .; then
        "$PYTHON_BIN" -m pip install --break-system-packages --extra-index-url "${PYTORCH_INDEX_URL}" .
    fi
elif [ -f "${PROJECT_DIR}/requirements.txt" ]; then
    uv pip install --system --python "$PYTHON_BIN" --break-system-packages -r "${PROJECT_DIR}/requirements.txt" || {
        "$PYTHON_BIN" -m pip install -r "${PROJECT_DIR}/requirements.txt"
    }
fi

# Ensure batch inference script is executable
if [ -f "${PROJECT_DIR}/scripts/batch_inference.py" ]; then
    chmod +x "${PROJECT_DIR}/scripts/batch_inference.py"
fi

# --- Step 5: GPU Pre-Flight CUDA Verification ---
echo -e "\n${BLUE}🧪 Step 5/5: Running GPU Pre-Flight CUDA Test...${NC}"
PREFLIGHT_OUTPUT=$("$PYTHON_BIN" -c '
import sys
import torch

print(f"  - PyTorch Version: {torch.__version__}")
print(f"  - CUDA Available:  {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ ERROR: PyTorch cannot detect CUDA on this host.")
    sys.exit(1)

dev_name = torch.cuda.get_device_name(0)
print(f"  - Device Name:     {dev_name}")

try:
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print("  - CUDA Tensor Execution: SUCCESS")
except Exception as e:
    print(f"❌ CUDA Matrix Multiplication Failed: {e}")
    sys.exit(1)
' 2>&1)

PREFLIGHT_EXIT=$?
echo "$PREFLIGHT_OUTPUT"

if [ $PREFLIGHT_EXIT -ne 0 ]; then
    echo -e "\n${RED}❌ Pre-flight GPU test failed! Please check your NVIDIA drivers.${NC}"
    exit 1
fi

echo -e "\n${GREEN}=================================================================${NC}"
echo -e "${GREEN}🎉 Inference Environment Setup Complete!${NC}"
echo -e "${GREEN}=================================================================${NC}"
echo -e "Project location: ${CYAN}${PROJECT_DIR}${NC}"
echo -e "\nNext steps:"
echo -e "1. Upload your model weights to: ${BOLD}${PROJECT_DIR}/artifacts/best.pth${NC}"
echo -e "2. Upload your images directory to:  ${BOLD}/workspace/images/${NC}"
echo -e "3. Run batch inference:"
echo -e "   ${BOLD}python3 scripts/batch_inference.py -i /workspace/images -o /workspace/predictions.json --device cuda${NC}\n"

# --- Optional Automated Inference Execution ---
if [ "$RUN_INFERENCE" = true ]; then
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${BLUE}🚀 Auto-Launching Batch Inference Pipeline...${NC}"
    echo -e "${BLUE}=================================================================${NC}\n"
    cd "$PROJECT_DIR" || exit 1

    # Resolve Weights Path
    RESOLVED_WEIGHTS="${WEIGHTS_PATH}"
    if [ -z "$RESOLVED_WEIGHTS" ]; then
        if [ -f "${PROJECT_DIR}/artifacts/best.pth" ]; then
            RESOLVED_WEIGHTS="${PROJECT_DIR}/artifacts/best.pth"
        elif [ -f "${WORKSPACE}/best.pth" ]; then
            RESOLVED_WEIGHTS="${WORKSPACE}/best.pth"
        elif [ -f "${WORKSPACE}/weights.pth" ]; then
            RESOLVED_WEIGHTS="${WORKSPACE}/weights.pth"
        fi
    fi

    # Check Weights Existence
    if [ -z "$RESOLVED_WEIGHTS" ] || [ ! -f "$RESOLVED_WEIGHTS" ]; then
        echo -e "${RED}❌ Model weights not found!${NC}"
        echo -e "Please upload your weights to ${BOLD}${PROJECT_DIR}/artifacts/best.pth${NC} or specify --weights <path>"
        exit 1
    fi

    # Check Images Directory Existence
    if [ ! -d "$IMAGES_DIR" ]; then
        echo -e "${RED}❌ Images directory not found at '${IMAGES_DIR}'!${NC}"
        echo -e "Please upload your images to ${BOLD}${IMAGES_DIR}${NC} or specify -i /path/to/images"
        exit 1
    fi

    INFER_ARGS=(
        "--images-dir" "$IMAGES_DIR"
        "--output-json" "$OUTPUT_JSON"
        "--weights" "$RESOLVED_WEIGHTS"
        "--device" "cuda"
        "--batch-size" "$BATCH_SIZE"
    )

    if [ "$SAVE_CSV" = true ]; then
        INFER_ARGS+=("--save-csv")
    fi

    if [ ${#EXTRA_INFERENCE_ARGS[@]} -gt 0 ]; then
        INFER_ARGS+=("${EXTRA_INFERENCE_ARGS[@]}")
    fi

    echo -e "${BLUE}Executing: ${PYTHON_BIN} scripts/batch_inference.py ${INFER_ARGS[*]}${NC}\n"
    "$PYTHON_BIN" scripts/batch_inference.py "${INFER_ARGS[@]}"
    INFER_EXIT=$?

    if [ $INFER_EXIT -eq 0 ]; then
        echo -e "\n${GREEN}🎉 Batch inference finished successfully! Results saved to ${OUTPUT_JSON}${NC}"
    else
        echo -e "\n${RED}❌ Batch inference failed with exit code ${INFER_EXIT}${NC}"
        exit $INFER_EXIT
    fi
fi
