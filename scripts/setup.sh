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

# --- Only Pull Mode ---
if [ "$ONLY_PULL" = true ]; then
    echo -e "${BLUE}=== Mode: Only Pull Repository Changes ===${NC}"
    TARGET_DIR="$PROJECT_DIR"
    if [ ! -d "$TARGET_DIR" ] && [ -d "${SCRIPT_DIR}/../.git" ]; then
        TARGET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi

    if [ -d "$TARGET_DIR" ]; then
        echo -e "${BLUE}🔄 Pulling latest changes into ${TARGET_DIR}...${NC}"
        cd "$TARGET_DIR" || exit 1
        git pull
        echo -e "${GREEN}✅ Successfully pulled latest repository changes without touching existing data.${NC}"
    else
        echo -e "${BLUE}📥 Repository not found. Cloning repository into ${PROJECT_DIR}...${NC}"
        if [ -z "${GITHUB_TOKEN// }" ] || [ -z "${GITHUB_USER// }" ]; then
            echo -e "${RED}❌ GITHUB_TOKEN or GITHUB_USER missing in environment variables${NC}"
            exit 1
        fi
        REPO_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
        mkdir -p "$WORKSPACE"
        cd "$WORKSPACE" || exit 1
        if ! git clone "$REPO_URL" "$PROJECT_DIR" 2>&1 | sed "s|${GITHUB_TOKEN}|***HIDDEN_TOKEN***|g"; then
            echo -e "${RED}❌ Failed to clone repository.${NC}"
            exit 1
        fi
        echo -e "${GREEN}✅ Successfully cloned repository.${NC}"
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
echo -e "${BLUE}⚙️ Installing base dependencies...${NC}"
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
    if ! git clone "$REPO_URL" "$PROJECT_DIR" 2>&1 | sed "s|${GITHUB_TOKEN}|***HIDDEN_TOKEN***|g"; then
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
    echo -e "${YELLOW}⚠️ No NVIDIA GPU / CUDA detected. Falling back to CPU PyTorch index.${NC}"
    CUDA_TAG="cpu"
fi

PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
echo -e "${BLUE}🎯 Selected PyTorch Index: ${PYTORCH_INDEX_URL}${NC}"

# Pre-install CUDA-matched PyTorch & torchvision into detected Python environment
echo -e "${BLUE}⬇️ Installing PyTorch and torchvision (${CUDA_TAG})...${NC}"
if ! uv pip install --system --python "$PYTHON_BIN" --no-python-downloads --break-system-packages torch torchvision torchaudio --index-url "${PYTORCH_INDEX_URL}"; then
    echo -e "${YELLOW}⚠️ uv pip install failed, attempting fallback via $PYTHON_BIN -m pip...${NC}"
    "$PYTHON_BIN" -m pip install --break-system-packages torch torchvision torchaudio --index-url "${PYTORCH_INDEX_URL}"
fi

# Install project dependencies using extra index url
echo -e "${BLUE}📦 Syncing remaining project dependencies...${NC}"
if ! uv pip install --system --python "$PYTHON_BIN" --no-python-downloads --break-system-packages --extra-index-url "${PYTORCH_INDEX_URL}" .; then
    echo -e "${YELLOW}⚠️ uv pip install . failed, attempting fallback via $PYTHON_BIN -m pip...${NC}"
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

# --- Optional Pipeline Execution ---
if [ "$RUN_PIPELINE" = true ]; then
    echo -e "\n${BLUE}==========================================${NC}"
    echo -e "${BLUE}🚀 Auto-launching Training Pipeline...${NC}"
    echo -e "${BLUE}==========================================${NC}\n"
    cd "$PROJECT_DIR" || exit 1

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