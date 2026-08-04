#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Step 1: Environment Setup ===${NC}"

WORKSPACE="/workspace"
REPO_NAME="cvm-swin-gcn"
PROJECT_DIR="${WORKSPACE}/${REPO_NAME}"

# --- Load .env file ---
# Looks for .env in the same directory as this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [ -f "$ENV_FILE" ]; then
    echo -e "${BLUE}ℹ️ Loading environment variables from .env...${NC}"
    # Export variables, ignoring comments and blank lines
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo -e "${RED}⚠️ .env file not found at $ENV_FILE. Relying on system env vars...${NC}"
fi
# ----------------------

# 1. Check if repository already exists and prompt for update
if [ -d "$PROJECT_DIR" ]; then
    echo -e "${GREEN}✅ Repository already cloned.${NC}"
    read -p "Force update repository and reinstall dependencies? [y/N]: " -n 1 -r
    echo "" # Move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}⏭️ Skipping setup.${NC}"
        exit 0
    fi
fi

# 2. Install base dependencies
echo -e "${BLUE}⚙️ Installing base dependencies...${NC}"
# Removed -qq to show apt-get output
apt-get update && apt-get install -y sshpass zip unzip pv curl git

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
    echo -e "${BLUE}🔄 Fetching and resetting existing repository...${NC}"
    git fetch origin && git reset --hard origin/main && git clean -fd
else
    echo -e "${BLUE}📥 Cloning repository...${NC}"
    # Removed /dev/null redirection to show output. 
    # Piped through sed to safely mask the token in the terminal logs.
    if ! git clone "$REPO_URL" "$PROJECT_DIR" 2>&1 | sed "s|${GITHUB_TOKEN}|***HIDDEN_TOKEN***|g"; then
        echo -e "${RED}❌ Failed to clone repository. Check your token and permissions.${NC}"
        exit 1
    fi
    cd "$PROJECT_DIR" || exit
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

# Ensure the cargo bin directory is in PATH for current script execution
export PATH="$HOME/.cargo/bin:$PATH"

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

CUDA_VERSION=""
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -i "CUDA Version" | sed -E 's/.*CUDA Version:[[:space:]]*([0-9]+\.[0-9]+).*/\1/' | head -n 1)
fi

if [ -z "$CUDA_VERSION" ] && command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -i "release" | sed -E 's/.*release ([0-9]+\.[0-9]+).*/\1/' | head -n 1)
fi

if [ -n "$CUDA_VERSION" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d'.' -f2)
    echo -e "${GREEN}✅ Detected host CUDA version: ${CUDA_VERSION}${NC}"

    if [ "$CUDA_MAJOR" -ge 12 ]; then
        if [ "$CUDA_MINOR" -ge 6 ]; then
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
    echo -e "${RED}⚠️ No NVIDIA GPU / CUDA detected. Falling back to CPU PyTorch index.${NC}"
    CUDA_TAG="cpu"
fi

PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
echo -e "${BLUE}🎯 Selected PyTorch Index: ${PYTORCH_INDEX_URL}${NC}"

# Pre-install CUDA-matched PyTorch & torchvision into detected Python environment
echo -e "${BLUE}⬇️ Installing PyTorch and torchvision (${CUDA_TAG})...${NC}"
if ! uv pip install --system --python "$PYTHON_BIN" --no-python-downloads --break-system-packages torch torchvision torchaudio --index-url "${PYTORCH_INDEX_URL}"; then
    echo -e "${RED}⚠️ uv pip install failed, attempting fallback via $PYTHON_BIN -m pip...${NC}"
    "$PYTHON_BIN" -m pip install --break-system-packages torch torchvision torchaudio --index-url "${PYTORCH_INDEX_URL}"
fi

# Install project dependencies using extra index url
echo -e "${BLUE}📦 Syncing remaining project dependencies...${NC}"
if ! uv pip install --system --python "$PYTHON_BIN" --no-python-downloads --break-system-packages --extra-index-url "${PYTORCH_INDEX_URL}" .; then
    echo -e "${RED}⚠️ uv pip install . failed, attempting fallback via $PYTHON_BIN -m pip...${NC}"
    "$PYTHON_BIN" -m pip install --break-system-packages --extra-index-url "${PYTORCH_INDEX_URL}" .
fi

# --- Verify PyTorch & CUDA installation ---
echo -e "\n${BLUE}🧪 Verifying PyTorch GPU / CUDA installation...${NC}"
"$PYTHON_BIN" -c '
import torch
print(f"  - PyTorch version: {torch.__version__}")
print(f"  - CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - Device Count:    {torch.cuda.device_count()}")
    print(f"  - Device Name:     {torch.cuda.get_device_name(0)}")
else:
    print("  - WARNING: CUDA is NOT available to PyTorch! Training will use CPU.")
'

echo -e "\n${GREEN}✅ Setup Complete!${NC}"