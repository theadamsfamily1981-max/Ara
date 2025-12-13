#!/bin/bash
# Ara Alpha - Run Script
#
# Starts the Ara Alpha server.
# Requires: Python 3.10+, FastAPI, uvicorn
#
# Usage:
#   ./ara_alpha/run.sh
#   ARA_PORT=9000 ./ara_alpha/run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check for required packages
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, yaml" 2>/dev/null || {
    echo "Missing dependencies. Install with:"
    echo "  pip install fastapi uvicorn pyyaml"
    exit 1
}

# Optional: check for OpenAI
python3 -c "import openai" 2>/dev/null || {
    echo "Note: openai package not installed. Using mock responses."
    echo "Install with: pip install openai"
}

# Create data directories
mkdir -p data/alpha_logs data/alpha_state

# Default settings
export ARA_HOST="${ARA_HOST:-0.0.0.0}"
export ARA_PORT="${ARA_PORT:-8080}"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                     ARA ALPHA SERVER                          ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Web UI:  http://${ARA_HOST}:${ARA_PORT}/                              ║"
echo "║  API:     http://${ARA_HOST}:${ARA_PORT}/api/                          ║"
echo "║  Docs:    http://${ARA_HOST}:${ARA_PORT}/docs                          ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Config:  ara_alpha/config/users.yaml                         ║"
echo "║  Logs:    data/alpha_logs/                                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Run server
python3 -m ara_alpha.server
