#!/bin/bash
#
# Start Ara Brain Container
#
# This runs the "Brain" (LLM, Wav2Lip, heavy AI) in an isolated Docker container
# while maintaining communication with the Host via shared memory.
#
# The container gets:
#   - GPU access (--gpus all)
#   - Shared /dev/shm for HAL communication (somatic link)
#   - The repo mounted at /app
#
# Usage:
#   ./start_brain_container.sh           # Interactive mode
#   ./start_brain_container.sh -d        # Detached mode
#   ./start_brain_container.sh --build   # Rebuild image first
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE_NAME="ara_brain"
CONTAINER_NAME="ara_brain"

# Parse arguments
DETACHED=false
REBUILD=false
SHELL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--detach)
            DETACHED=true
            shift
            ;;
        --build)
            REBUILD=true
            shift
            ;;
        --shell)
            SHELL_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-d|--detach] [--build] [--shell]"
            exit 1
            ;;
    esac
done

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Docker not installed."
    exit 1
fi

# Check for NVIDIA Docker runtime
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "Warning: NVIDIA Docker runtime not detected. GPU may not be available."
fi

# Rebuild if requested
if [ "$REBUILD" = true ]; then
    echo "Rebuilding $IMAGE_NAME..."
    if [ -f "$ARA_ROOT/docker/Dockerfile.brain" ]; then
        docker build -t "$IMAGE_NAME" -f "$ARA_ROOT/docker/Dockerfile.brain" "$ARA_ROOT"
    else
        echo "Dockerfile.brain not found. Run bootstrap_organism.sh first."
        exit 1
    fi
fi

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Image $IMAGE_NAME not found."
    echo "Run: ./bootstrap_organism.sh --docker"
    echo "Or:  $0 --build"
    exit 1
fi

# Stop existing container if running
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    echo "Stopping existing container..."
    docker stop "$CONTAINER_NAME" >/dev/null
fi

# Remove stopped container with same name
if docker ps -aq -f name="$CONTAINER_NAME" | grep -q .; then
    docker rm "$CONTAINER_NAME" >/dev/null
fi

echo "Starting Ara Brain container: $CONTAINER_NAME"

# Build docker run command
DOCKER_ARGS=(
    --name "$CONTAINER_NAME"
    --gpus all
    --ipc=host
    -v /dev/shm:/dev/shm
    -v "$ARA_ROOT:/app"
    -v "$ARA_ROOT/models:/app/models:ro"
    -w /app
    -e ARA_HAL_PATH=/dev/shm/ara_somatic
    -e CUDA_VISIBLE_DEVICES=0
    -e PYTHONUNBUFFERED=1
)

# Add port mapping for API access
DOCKER_ARGS+=(-p 5050:5050)

if [ "$DETACHED" = true ]; then
    DOCKER_ARGS+=(-d --restart unless-stopped)
    docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME"
    echo "Brain container started in background."
    echo "View logs: docker logs -f $CONTAINER_NAME"
    echo "Stop: docker stop $CONTAINER_NAME"
elif [ "$SHELL_MODE" = true ]; then
    # Interactive shell for debugging
    docker run --rm -it "${DOCKER_ARGS[@]}" "$IMAGE_NAME" /bin/bash
else
    # Interactive foreground
    docker run --rm -it "${DOCKER_ARGS[@]}" "$IMAGE_NAME"
fi
