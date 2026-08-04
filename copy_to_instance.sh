#!/bin/bash
set -e

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine repository root relative to script position
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/setup.sh" ]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    REPO_ROOT="${SCRIPT_DIR}"
fi

ENV_FILE="${REPO_ROOT}/.env"
SETUP_SCRIPT="${REPO_ROOT}/scripts/setup.sh"

echo -e "${BLUE}=== Cloud GPU Instance File Uploader ===${NC}\n"


# Prompt for IP address if not supplied as 1st argument
INSTANCE_IP="${1:-}"
if [ -z "$INSTANCE_IP" ]; then
    read -p "Enter Cloud GPU Instance IP (e.g. 192.168.1.100 or root@192.168.1.100): " INSTANCE_IP
fi

if [ -z "$INSTANCE_IP" ]; then
    echo -e "${RED}❌ Error: IP address cannot be empty.${NC}"
    exit 1
fi

# Parse SSH user if included in input (default to 'root')
if [[ "$INSTANCE_IP" == *"@"* ]]; then
    SSH_USER="${INSTANCE_IP%%@*}"
    INSTANCE_IP="${INSTANCE_IP#*@}"
else
    SSH_USER="root"
fi

# Prompt for SSH Port if not supplied as 2nd argument
INSTANCE_PORT="${2:-}"
if [ -z "$INSTANCE_PORT" ]; then
    read -p "Enter SSH Port [default: 22]: " INPUT_PORT
    INSTANCE_PORT="${INPUT_PORT:-22}"
fi

# Remote destination directory (defaults to /workspace)
REMOTE_DEST="${3:-/workspace}"

echo -e "\n${BLUE}🚀 Uploading files to ${SSH_USER}@${INSTANCE_IP}:${INSTANCE_PORT} (${REMOTE_DEST}/)...${NC}"
echo -e "   📄 .env             -> ${ENV_FILE}"
echo -e "   📜 scripts/setup.sh -> ${SETUP_SCRIPT}\n"

# Execute scp
if scp -P "$INSTANCE_PORT" "$ENV_FILE" "$SETUP_SCRIPT" "${SSH_USER}@${INSTANCE_IP}:${REMOTE_DEST}/"; then
    echo -e "\n${GREEN}✅ Files copied successfully to ${SSH_USER}@${INSTANCE_IP}:${INSTANCE_PORT}:${REMOTE_DEST}/${NC}"
    
    echo -e "${BLUE}🔧 Making setup.sh executable...${NC}"
    if ssh -p "$INSTANCE_PORT" "${SSH_USER}@${INSTANCE_IP}" "chmod +x ${REMOTE_DEST}/setup.sh"; then
        echo -e "${GREEN}✅ setup.sh is now executable.${NC}"
    else
        echo -e "${RED}❌ Failed to set executable permissions for setup.sh via ssh.${NC}"
        exit 1
    fi
else
    echo -e "\n${RED}❌ Failed to copy files via scp.${NC}"
    exit 1
fi
