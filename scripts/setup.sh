#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Command-line arguments parsing ---
ONLY_PULL=false
AUTO_CONFIRM=false
RUN_PIPELINE=false
BRANCH_ARG=""
PIPELINE_ARGS=()

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
        --run-pipeline)
            RUN_PIPELINE=true
            shift
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                PIPELINE_ARGS+=("$1")
                shift
            done
            break
            ;;
        *)
            if [ "$RUN_PIPELINE" = true ]; then
                PIPELINE_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

echo -e "${BLUE}=== Step 1: Environment Setup ===${NC}"

WORKSPACE="/workspace"
REPO_NAME="cvm-swin-gcn"
PROJECT_DIR="${WORKSPACE}/${REPO_NAME}"

# --- Locate and Load .env file ---
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
    echo -e "${YELLOW}⚠️ .env file not found. Relying on system env vars...${NC}"
fi
# ----------------------
# Priority: CLI argument (-b/--branch) > GIT_BRANCH / BRANCH env var > default: main
TARGET_BRANCH="${BRANCH_ARG:-${GIT_BRANCH:-${BRANCH:-main}}}"
echo -e "${BLUE}🌿 Target Git Branch: ${TARGET_BRANCH}${NC}"

# --- Only Pull Mode ---
if [ "$ONLY_PULL" = true ]; then
    echo -e "${BLUE}=== Mode: Only Pull Repository Changes ===${NC}"
    TARGET_DIR="$PROJECT_DIR"
    if [ ! -d "$TARGET_DIR" ] && [ -d "${SCRIPT_DIR}/../.git" ]; then
        TARGET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi

    if [ -d "$TARGET_DIR" ]; then
        echo -e "${BLUE}🔄 Pulling latest changes from branch '${TARGET_BRANCH}' into ${TARGET_DIR}...${NC}"
        cd "$TARGET_DIR" || exit 1
        git fetch origin
        git checkout "$TARGET_BRANCH" 2>/dev/null || git checkout -b "$TARGET_BRANCH" "origin/$TARGET_BRANCH"
        git pull origin "$TARGET_BRANCH"
        echo -e "${GREEN}✅ Successfully pulled latest repository changes on branch '${TARGET_BRANCH}' without touching existing data.${NC}"
    else
        echo -e "${BLUE}📥 Repository not found. Cloning branch '${TARGET_BRANCH}' into ${PROJECT_DIR}...${NC}"
        if [ -z "${GITHUB_TOKEN// }" ] || [ -z "${GITHUB_USER// }" ]; then
            echo -e "${RED}❌ GITHUB_TOKEN or GITHUB_USER missing in environment variables${NC}"
            exit 1
        fi
        REPO_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
        mkdir -p "$WORKSPACE"
        cd "$WORKSPACE" || exit 1
        if ! git clone -b "$TARGET_BRANCH" "$REPO_URL" "$PROJECT_DIR" 2>&1 | sed "s|${GITHUB_TOKEN}|***HIDDEN_TOKEN***|g"; then
            echo -e "${RED}❌ Failed to clone repository.${NC}"
            exit 1
        fi
        echo -e "${GREEN}✅ Successfully cloned repository on branch '${TARGET_BRANCH}'.${NC}"
    fi

    if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ] && [ "$ENV_FILE" != "${TARGET_DIR}/.env" ]; then
        cp "$ENV_FILE" "${TARGET_DIR}/.env" 2>/dev/null || true
    fi

    exit 0
fi

# 1. Check if repository already exists
if [ -d "$PROJECT_DIR" ]; then
    echo -e "${GREEN}✅ Repository already exists at ${PROJECT_DIR}.${NC}"
    if [ "$AUTO_CONFIRM" = false ]; then
        read -p "Force update repository and reinstall dependencies? [y/N]: " -n 1 -r
        echo "" # Move to a new line
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}⏭️ Skipping repo update & base install.${NC}"
        fi
    fi
fi

# 2. Install base dependencies
echo -e "${BLUE}⚙️ Installing base dependencies (including tmux)...${NC}"
apt-get update && apt-get install -y tmux sshpass zip unzip pv curl git libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 libxcb1 libx11-xcb1 2>/dev/null || apt-get install -y tmux sshpass zip unzip pv curl git libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libxcb1 libx11-xcb1

cd "$WORKSPACE" || exit

# 3. Check for GitHub credentials
if [ -z "${GITHUB_TOKEN// }" ] || [ -z "${GITHUB_USER// }" ]; then
    echo -e "${RED}❌ GITHUB_TOKEN or GITHUB_USER missing in environment variables${NC}"
    exit 1
fi

REPO_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

# Prevent uv from downloading an external Python version
export UV_PYTHON_DOWNLOADS=never

# 4. Clone or update the repository
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR" || exit
    echo -e "${BLUE}🔄 Fetching and resetting repository to 'origin/${TARGET_BRANCH}'...${NC}"
    git fetch origin
    git checkout "$TARGET_BRANCH" 2>/dev/null || git checkout -b "$TARGET_BRANCH" "origin/$TARGET_BRANCH"
    git reset --hard "origin/${TARGET_BRANCH}" && git clean -fd
else
    echo -e "${BLUE}📥 Cloning repository on branch '${TARGET_BRANCH}'...${NC}"
    if ! git clone -b "$TARGET_BRANCH" "$REPO_URL" "$PROJECT_DIR" 2>&1 | sed "s|${GITHUB_TOKEN}|***HIDDEN_TOKEN***|g"; then
        echo -e "${RED}❌ Failed to clone repository. Check your token and permissions.${NC}"
        exit 1
    fi
    cd "$PROJECT_DIR" || exit
fi

# Ensure .env is placed inside the project root for scripts
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "${PROJECT_DIR}/.env" 2>/dev/null || true
fi

# Clean up any cloned .python-version or strict requires-python that force uv to switch Python runtimes
rm -f "${PROJECT_DIR}/.python-version" 2>/dev/null || true
if [ -f "${PROJECT_DIR}/pyproject.toml" ]; then
    sed -i 's/requires-python = ">=[0-9]*\.[0-9]*"/requires-python = ">=3.8"/g' "${PROJECT_DIR}/pyproject.toml" 2>/dev/null || true
    sed -i 's/torch>=[0-9]*\.[0-9]*\.[0-9]*/torch>=2.0.0/g' "${PROJECT_DIR}/pyproject.toml" 2>/dev/null || true
fi

# 5. Install uv and detect CUDA / PyTorch configuration
echo -e "\n${BLUE}📦 Setting up UV and PyTorch environment...${NC}"
curl -LSf https://astral.sh/uv/install.sh | sh

# Ensure the cargo bin directory is in PATH
export PATH="$HOME/.cargo/bin:$PATH"
grep -qxF 'export PATH="$HOME/.cargo/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null || echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bashrc" 2>/dev/null || true

# --- Locate Python Executable ---
PYTHON_BIN=""
if command -v python3 &> /dev/null; then
    PYTHON_BIN=$(command -v python3)
elif command -v python &> /dev/null; then
    PYTHON_BIN=$(command -v python)
else
    echo -e "${RED}❌ Python binary not found in PATH!${NC}"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1)
echo -e "${BLUE}🐍 Using Python binary: ${PYTHON_BIN} (${PYTHON_VERSION})${NC}"

# --- Automatic CUDA & GPU Detection ---
echo -e "${BLUE}🔍 Detecting GPU and CUDA version...${NC}"

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
    echo -e "${GREEN}⚡ Detected NVIDIA RTX 50-series GPU: ${GPU_MODEL} (Blackwell / sm_120)${NC}"
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
    echo -e "${YELLOW}⚠️ No NVIDIA GPU / CUDA detected. Falling back to CPU PyTorch index.${NC}"
    CUDA_TAG="cpu"
fi

PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
echo -e "${BLUE}🎯 Selected PyTorch Index: ${PYTORCH_INDEX_URL}${NC}"

# Pre-install CUDA-matched PyTorch & torchvision into detected Python environment
echo -e "${BLUE}⬇️ Installing PyTorch and torchvision (${CUDA_TAG})...${NC}"
if [ "$CUDA_TAG" = "cu128" ]; then
    echo -e "${BLUE}ℹ️ CUDA 12.8 / RTX 5090 detected. Installing PyTorch nightly with cu128 support (for Blackwell / sm_120)...${NC}"
    if ! uv pip install --system --python "$PYTHON_BIN" --no-python-downloads --break-system-packages --pre torch torchvision "numpy<2" --index-url "https://download.pytorch.org/whl/nightly/cu128" --extra-index-url "https://pypi.org/simple"; then
        echo -e "${YELLOW}⚠️ uv pip install nightly cu128 failed, retrying with python -m pip...${NC}"
        if ! "$PYTHON_BIN" -m pip install --break-system-packages --pre torch torchvision "numpy<2" --index-url "https://download.pytorch.org/whl/nightly/cu128" --extra-index-url "https://pypi.org/simple"; then
            echo -e "${YELLOW}⚠️ Nightly cu128 pip install failed, falling back to stable cu126...${NC}"
            CUDA_TAG="cu126"
            PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
            uv pip install --system --python "$PYTHON_BIN" --no-python-downloads --break-system-packages torch torchvision "numpy<2" --index-url "${PYTORCH_INDEX_URL}"
        fi
    fi
else
    if ! uv pip install --system --python "$PYTHON_BIN" --no-python-downloads --break-system-packages torch torchvision "numpy<2" --index-url "${PYTORCH_INDEX_URL}"; then
        echo -e "${YELLOW}⚠️ Failed with ${PYTORCH_INDEX_URL}, retrying with cu126 index...${NC}"
        if ! uv pip install --system --python "$PYTHON_BIN" --no-python-downloads --break-system-packages torch torchvision "numpy<2" --index-url "https://download.pytorch.org/whl/cu126"; then
            echo -e "${YELLOW}⚠️ Fallback to standard PyPI for PyTorch...${NC}"
            "$PYTHON_BIN" -m pip install --break-system-packages torch torchvision "numpy<2"
        fi
    fi
fi

# Install project dependencies using extra index url
echo -e "${BLUE}📦 Syncing remaining project dependencies...${NC}"
if ! uv pip install --system --python "$PYTHON_BIN" --no-python-downloads --break-system-packages --extra-index-url "${PYTORCH_INDEX_URL}" --extra-index-url "https://download.pytorch.org/whl/cu126" --extra-index-url "https://download.pytorch.org/whl/nightly/cu128" .; then
    echo -e "${YELLOW}⚠️ uv pip install . failed, attempting fallback via $PYTHON_BIN -m pip...${NC}"
    "$PYTHON_BIN" -m pip install --break-system-packages --ignore-installed --extra-index-url "${PYTORCH_INDEX_URL}" --extra-index-url "https://download.pytorch.org/whl/cu126" --extra-index-url "https://download.pytorch.org/whl/nightly/cu128" .
fi

# --- Step 5: GPU Pre-Flight CUDA Test ---
echo -e "\n${BLUE}🧪 Running GPU Pre-Flight CUDA Kernel Execution Test...${NC}"

PREFLIGHT_OUTPUT=$("$PYTHON_BIN" -c '
import sys
import torch

print(f"  - PyTorch version: {torch.__version__}")
print(f"  - CUDA Available:  {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ ERROR: CUDA is NOT available to PyTorch.")
    sys.exit(1)

dev_name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
print(f"  - Device Count:    {torch.cuda.device_count()}")
print(f"  - Device Name:     {dev_name} (compute capability {arch})")

arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
if arch_list:
    arch_str = " ".join(arch_list)
    print(f"  - PyTorch Archs:   {arch_str}")

# Execute an actual CUDA kernel operation on GPU to verify kernel compilation & driver support
try:
    a = torch.randn(128, 128, device="cuda")
    b = torch.randn(128, 128, device="cuda")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print("  - CUDA Kernel Exec: SUCCESS (Tensor matrix multiplication verified on GPU)")
except Exception as e:
    print(f"❌ PRE-FLIGHT TEST FAILED: CUDA kernel execution error on {dev_name} ({arch}): {e}")
    sys.exit(1)

# Verify Triton compiler & torch.compile support
try:
    import triton
    if hasattr(torch, "compile"):
        @torch.compile
        def _probe(x):
            return x + 1
        _probe(torch.zeros(1, device="cuda"))
        print("  - Triton Compiler: AVAILABLE (torch.compile acceleration supported)")
    else:
        print("  - Triton Compiler: PyTorch < 2.0 (compile not supported)")
except Exception:
    print("  - Triton Compiler: NOT AVAILABLE (eager execution mode)")
' 2>&1)

PREFLIGHT_EXIT=$?
echo "$PREFLIGHT_OUTPUT"

if [ $PREFLIGHT_EXIT -ne 0 ]; then
    echo -e "\n${RED}❌ Pre-flight GPU CUDA test failed! Aborting before downloading data or launching training pipeline.${NC}"
    exit 1
fi

echo -e "\n${GREEN}✅ Pre-flight GPU test passed! GPU and CUDA execution environment verified.${NC}"
echo -e "\n${GREEN}✅ Setup Complete!${NC}"

# --- Optional Pipeline Execution ---
if [ "$RUN_PIPELINE" = true ]; then
    echo -e "\n${BLUE}==========================================${NC}"
    echo -e "${BLUE}🚀 Auto-launching Training Pipeline...${NC}"
    echo -e "${BLUE}==========================================${NC}\n"
    cd "$PROJECT_DIR" || exit 1

    # Check if --compile or --no-compile was already explicitly passed
    COMPILE_FLAG_PASSED=false
    for arg in "${PIPELINE_ARGS[@]}"; do
        if [ "$arg" = "--compile" ] || [ "$arg" = "--no-compile" ]; then
            COMPILE_FLAG_PASSED=true
            break
        fi
    done

    # Automatically probe and enable --compile if supported and not explicitly passed
    if [ "$COMPILE_FLAG_PASSED" = false ]; then
        if "$PYTHON_BIN" -c '
import sys
try:
    import triton
    import torch
    if hasattr(torch, "compile"):
        @torch.compile
        def _probe(x):
            return x + 1
        _probe(torch.zeros(1))
        sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
' 2>/dev/null; then
            echo -e "${GREEN}⚡ Triton compiler supported in this environment! Automatically adding --compile flag.${NC}"
            PIPELINE_ARGS+=("--compile")
        else
            echo -e "${YELLOW}ℹ️ Triton compiler not available or not supported on this environment. Running in standard eager mode.${NC}"
        fi
    fi

    echo -e "${BLUE}Running: ${PYTHON_BIN} scripts/train-pipeline.py ${PIPELINE_ARGS[*]}${NC}\n"
    "$PYTHON_BIN" scripts/train-pipeline.py "${PIPELINE_ARGS[@]}"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "\n${GREEN}🎉 Training pipeline finished successfully!${NC}"
    else
        echo -e "\n${RED}❌ Training pipeline exited with code ${EXIT_CODE}.${NC}"
        exit $EXIT_CODE
    fi
fi