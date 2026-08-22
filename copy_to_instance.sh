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

INSTANCE_IP=""
INSTANCE_PORT="22"
REMOTE_DEST="/workspace"
EXEC_PIPELINE=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run|--run-pipeline)
            EXEC_PIPELINE=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./copy_to_instance.sh [IP/HOST] [PORT] [REMOTE_DEST] [--run]"
            echo "Example: ./copy_to_instance.sh 123.45.67.89 12345 /workspace --run"
            exit 0
            ;;
        *)
            if [ -z "$INSTANCE_IP" ]; then
                INSTANCE_IP="$1"
            elif [ "$INSTANCE_PORT" = "22" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
                INSTANCE_PORT="$1"
            elif [ "$REMOTE_DEST" = "/workspace" ] && [[ "$1" == /* ]]; then
                REMOTE_DEST="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Prompt for IP address if not supplied
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

echo -e "\n${BLUE}🚀 Uploading files to ${SSH_USER}@${INSTANCE_IP}:${INSTANCE_PORT} (${REMOTE_DEST}/)...${NC}"
echo -e "   📄 .env             -> ${ENV_FILE}"
echo -e "   📜 scripts/setup.sh -> ${SETUP_SCRIPT}\n"

# Execute scp with disabled host key check for transient GPU instances
if scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$INSTANCE_PORT" "$ENV_FILE" "$SETUP_SCRIPT" "${SSH_USER}@${INSTANCE_IP}:${REMOTE_DEST}/"; then
    echo -e "\n${GREEN}✅ Files copied successfully to ${SSH_USER}@${INSTANCE_IP}:${INSTANCE_PORT}:${REMOTE_DEST}/${NC}"
    
    echo -e "${BLUE}🔧 Making setup.sh executable...${NC}"
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$INSTANCE_PORT" "${SSH_USER}@${INSTANCE_IP}" "chmod +x ${REMOTE_DEST}/setup.sh"; then
        echo -e "${GREEN}✅ setup.sh is now executable.${NC}"
    else
        echo -e "${RED}❌ Failed to set executable permissions for setup.sh via ssh.${NC}"
        exit 1
    fi
else
    echo -e "\n${RED}❌ Failed to copy files via scp.${NC}"
    exit 1
fi

if [ "$EXEC_PIPELINE" = true ]; then
    echo -e "\n${BLUE}🚀 Launching setup and training pipeline on remote instance...${NC}"
    ssh -t -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$INSTANCE_PORT" "${SSH_USER}@${INSTANCE_IP}" \
        "bash ${REMOTE_DEST}/setup.sh -y --run-pipeline ${EXTRA_ARGS[*]}"
fi
