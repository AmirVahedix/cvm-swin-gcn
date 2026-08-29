#!/bin/bash
# ==============================================================================
# Copy to Instance Script
# Copies only .env and setup.sh to the remote instance (e.g., Vast.ai / cloud host),
# makes setup.sh executable on the host, and exits cleanly.
# ==============================================================================

set -e

# --- Color Formatting ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Defaults ---
SSH_USER="root"
SSH_HOST=""
SSH_PORT=""
REMOTE_DEST="/workspace"
SSH_KEY=""
POSITIONAL_ARGS=()

# --- Script Location Resolution ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." 2>/dev/null && pwd || pwd)"

# --- Helper Functions ---
usage() {
    echo -e "${BOLD}Usage:${NC}"
    echo -e "  $(basename "$0") [OPTIONS] [HOST] [PORT]"
    echo -e "  $(basename "$0") root@<host>:<port>"
    echo -e "  $(basename "$0") \"ssh -p <port> root@<host>\"\n"
    echo -e "${BOLD}Options:${NC}"
    echo -e "  -H, --host <host>       Remote SSH host / IP address"
    echo -e "  -p, --port <port>       Remote SSH port"
    echo -e "  -u, --user <user>       Remote SSH user (default: root)"
    echo -e "  -d, --dest <path>       Remote destination directory (default: /workspace)"
    echo -e "  -i, --identity <file>   SSH private key file (optional)"
    echo -e "  -h, --help              Show this help message and exit\n"
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  $(basename "$0") ssh4.vast.ai 12345"
    echo -e "  $(basename "$0") root@ssh4.vast.ai:12345"
    echo -e "  $(basename "$0") --host 74.50.x.x --port 12345"
    echo -e "  $(basename "$0") \"ssh -p 12345 root@ssh4.vast.ai\"\n"
    exit 0
}

# --- Locate .env file ---
ENV_FILE=""
for candidate in "${REPO_ROOT}/.env" "${SCRIPT_DIR}/.env" "$(pwd)/.env"; do
    if [ -f "$candidate" ]; then
        ENV_FILE="$candidate"
        break
    fi
done

# Load environment variables if available
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
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
fi

# --- Locate setup.sh script ---
SETUP_SCRIPT=""
for candidate in "${SCRIPT_DIR}/setup.sh" "${REPO_ROOT}/scripts/setup.sh" "${REPO_ROOT}/setup.sh" "$(pwd)/setup.sh"; do
    if [ -f "$candidate" ]; then
        SETUP_SCRIPT="$candidate"
        break
    fi
done

# --- Parse Arguments ---
parse_connection_string() {
    local raw="$1"
    
    # Strip leading 'ssh ' if present
    raw="${raw#ssh }"
    
    # Match patterns like: -p 12345 root@host or root@host -p 12345
    if [[ "$raw" =~ -p[[:space:]]*([0-9]+) ]]; then
        SSH_PORT="${BASH_REMATCH[1]}"
        raw=$(echo "$raw" | sed -E 's/-p[[:space:]]*[0-9]+//g')
    fi
    
    # Clean whitespace
    raw=$(echo "$raw" | xargs)
    
    # Match user@host:port or host:port
    if [[ "$raw" =~ ^([a-zA-Z0-9_-]+@)?([^:]+):([0-9]+)$ ]]; then
        [ -n "${BASH_REMATCH[1]}" ] && SSH_USER="${BASH_REMATCH[1]%@}"
        SSH_HOST="${BASH_REMATCH[2]}"
        SSH_PORT="${BASH_REMATCH[3]}"
    # Match user@host or host
    elif [[ "$raw" =~ ^([a-zA-Z0-9_-]+@)?([^[:space:]]+)$ ]]; then
        [ -n "${BASH_REMATCH[1]}" ] && SSH_USER="${BASH_REMATCH[1]%@}"
        SSH_HOST="${BASH_REMATCH[2]}"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -H|--host)
            SSH_HOST="$2"
            shift 2
            ;;
        -p|--port)
            SSH_PORT="$2"
            shift 2
            ;;
        -u|--user)
            SSH_USER="$2"
            shift 2
            ;;
        -d|--dest)
            REMOTE_DEST="$2"
            shift 2
            ;;
        -i|--identity)
            SSH_KEY="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Parse positional arguments
if [ ${#POSITIONAL_ARGS[@]} -ge 1 ]; then
    if [ ${#POSITIONAL_ARGS[@]} -eq 1 ]; then
        parse_connection_string "${POSITIONAL_ARGS[0]}"
    elif [ ${#POSITIONAL_ARGS[@]} -ge 2 ]; then
        FIRST="${POSITIONAL_ARGS[0]}"
        SECOND="${POSITIONAL_ARGS[1]}"
        
        if [[ "$FIRST" =~ ^[0-9]+$ ]] && [[ ! "$SECOND" =~ ^[0-9]+$ ]]; then
            SSH_PORT="$FIRST"
            parse_connection_string "$SECOND"
        elif [[ "$SECOND" =~ ^[0-9]+$ ]]; then
            SSH_PORT="$SECOND"
            parse_connection_string "$FIRST"
        else
            parse_connection_string "$FIRST"
        fi
    fi
fi

# Fallback to environment variables if host/port not provided
if [ -z "$SSH_HOST" ] && [ -n "$VAST_SSH_HOST" ]; then
    SSH_HOST="$VAST_SSH_HOST"
fi
if [ -z "$SSH_PORT" ] && [ -n "$VAST_SSH_PORT" ]; then
    SSH_PORT="$VAST_SSH_PORT"
fi

# Query Vast API if INSTANCE_ID & VAST_API_KEY exist and host/port still missing
if { [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; } && [ -n "${INSTANCE_ID:-${VAST_INSTANCE_ID}}" ] && [ -n "$VAST_API_KEY" ]; then
    TARGET_ID="${INSTANCE_ID:-${VAST_INSTANCE_ID}}"
    echo -e "${BLUE}🔍 Stored INSTANCE_ID=${TARGET_ID} detected. Fetching connection details from Vast.ai API...${NC}"
    
    VAST_INFO=$(python3 -c "
import urllib.request, json, ssl
try:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request('https://console.vast.ai/api/v0/instances/?api_key=${VAST_API_KEY}', headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        instances = data.get('instances', []) if isinstance(data, dict) else data
        for inst in instances:
            if str(inst.get('id')) == '${TARGET_ID}':
                h = inst.get('ssh_host') or inst.get('public_ipaddr') or ''
                p = inst.get('ssh_port') or ''
                print(f'{h} {p}')
                break
except Exception:
    pass
" 2>/dev/null || true)

    if [ -n "$VAST_INFO" ]; then
        VAST_H=$(echo "$VAST_INFO" | awk '{print $1}')
        VAST_P=$(echo "$VAST_INFO" | awk '{print $2}')
        [ -z "$SSH_HOST" ] && [ -n "$VAST_H" ] && SSH_HOST="$VAST_H"
        [ -z "$SSH_PORT" ] && [ -n "$VAST_P" ] && SSH_PORT="$VAST_P"
    fi
fi

# Interactive prompt if host or port are still missing
if [ -z "$SSH_HOST" ]; then
    if [ -t 0 ]; then
        read -r -p "Enter Remote SSH Host / IP (e.g., ssh4.vast.ai): " SSH_HOST
    fi
fi

if [ -z "$SSH_PORT" ]; then
    if [ -t 0 ]; then
        read -r -p "Enter Remote SSH Port (e.g., 12345): " SSH_PORT
    fi
fi

# Validation
if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    echo -e "${RED}❌ Error: Remote SSH Host and Port are required.${NC}"
    echo -e "Usage: $0 <SSH_HOST> <SSH_PORT>"
    exit 1
fi

if [ -z "$ENV_FILE" ] || [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}❌ Error: .env file not found in workspace.${NC}"
    exit 1
fi

if [ -z "$SETUP_SCRIPT" ] || [ ! -f "$SETUP_SCRIPT" ]; then
    echo -e "${RED}❌ Error: setup.sh script not found at ${SETUP_SCRIPT:-scripts/setup.sh}.${NC}"
    exit 1
fi

# --- SSH / SCP Flags ---
SSH_OPTS=(-o "StrictHostKeyChecking=no" -o "UserKnownHostsFile=/dev/null" -o "ConnectTimeout=10")
if [ -n "$SSH_KEY" ]; then
    SSH_OPTS+=(-i "$SSH_KEY")
fi

SSH_TARGET="${SSH_USER}@${SSH_HOST}"

echo -e "\n${BLUE}=================================================================${NC}"
echo -e "${BLUE}🚀 Deploying Setup Files to Remote Instance${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "  - Remote Target: ${CYAN}${SSH_TARGET}:${SSH_PORT}${NC}"
echo -e "  - Destination:   ${CYAN}${REMOTE_DEST}/${NC}"
echo -e "  - Env File:      ${GREEN}${ENV_FILE}${NC}"
echo -e "  - Setup Script:  ${GREEN}${SETUP_SCRIPT}${NC}"
echo -e "${BLUE}-----------------------------------------------------------------${NC}\n"

# Step 1: Ensure remote directory exists
echo -e "${BLUE}📁 Ensuring remote directory ${REMOTE_DEST} exists...${NC}"
if ! ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$SSH_TARGET" "mkdir -p ${REMOTE_DEST}"; then
    echo -e "${RED}❌ Failed to connect to ${SSH_TARGET}:${SSH_PORT} via SSH.${NC}"
    exit 1
fi

# Step 2: SCP .env and setup.sh
echo -e "${BLUE}📦 Copying .env and setup.sh to ${SSH_TARGET}:${REMOTE_DEST}/...${NC}"
SCP_KEY_OPTS=()
if [ -n "$SSH_KEY" ]; then
    SCP_KEY_OPTS+=(-i "$SSH_KEY")
fi

if ! scp -o "StrictHostKeyChecking=no" -o "UserKnownHostsFile=/dev/null" "${SCP_KEY_OPTS[@]}" -P "$SSH_PORT" "$ENV_FILE" "$SETUP_SCRIPT" "${SSH_TARGET}:${REMOTE_DEST}/"; then
    echo -e "${RED}❌ Failed to copy files to ${SSH_TARGET}:${REMOTE_DEST}/ via SCP.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Files copied successfully.${NC}"

# Step 3: Make setup.sh executable on the host
echo -e "${BLUE}🔑 Setting executable permissions on ${REMOTE_DEST}/setup.sh...${NC}"
if ! ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$SSH_TARGET" "chmod +x ${REMOTE_DEST}/setup.sh"; then
    echo -e "${RED}❌ Failed to set executable permissions on ${REMOTE_DEST}/setup.sh.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ ${REMOTE_DEST}/setup.sh is now executable.${NC}"

# --- Completion Summary ---
echo -e "\n${BLUE}=================================================================${NC}"
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "You can now connect to your instance and run the setup script:"
echo -e "\n  ${BOLD}ssh -p ${SSH_PORT} ${SSH_TARGET}${NC}"
echo -e "  ${BOLD}cd ${REMOTE_DEST} && ./setup.sh${NC}\n"
echo -e "Or run setup with options directly:"
echo -e "  ${BOLD}ssh -p ${SSH_PORT} ${SSH_TARGET} \"cd ${REMOTE_DEST} && ./setup.sh -y\"${NC}\n"
echo -e "${BLUE}Exiting.${NC}"
exit 0
