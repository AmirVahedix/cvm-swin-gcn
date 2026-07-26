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

# 5. Install uv and sync dependencies
echo -e "\n${BLUE}📦 Syncing UV dependencies...${NC}"
# Removed 's' (silent) from curl flags to show download progress
curl -LSf https://astral.sh/uv/install.sh | sh

# Ensure the cargo bin directory is in PATH for the current script execution
export PATH="$HOME/.cargo/bin:$PATH"

# Run uv sync (output is already visible by default)
uv pip install --system .

echo -e "${GREEN}✅ Setup Complete!${NC}"