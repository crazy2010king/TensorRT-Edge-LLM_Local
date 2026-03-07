#!/bin/bash
set -euo pipefail

# Environment setup and dependency check script
# TensorRT Edge-LLM Performance Test Suite

echo "Checking environment dependencies..."

# Check required commands
REQUIRED_COMMANDS=(
    "python3"
    "curl"
)

for CMD in "${REQUIRED_COMMANDS[@]}"; do
    if ! command -v "$CMD" &> /dev/null; then
        echo "ERROR: Required command not found: $CMD"
        exit 1
    fi
done

# Special handling for tegrastats (Jetson specific)
if ! command -v tegrastats &> /dev/null; then
    echo "WARNING: tegrastats command not found, attempting automatic installation..."
    # Try to install tegrastats
    if [[ $EUID -eq 0 ]]; then
        apt update -y >/dev/null 2>&1
        if apt install -y nvidia-l4t-tools >/dev/null 2>&1; then
            echo "SUCCESS: tegrastats installed successfully"
        else
            echo "WARNING: Failed to install tegrastats, will disable power and system metrics collection"
            export DISABLE_TEGRASTATS=1
        fi
    else
        echo "WARNING: Not running as root, cannot install tegrastats automatically"
        echo "WARNING: Please run 'sudo apt install nvidia-l4t-tools' to install tegrastats for full metrics collection"
        echo "WARNING: Will proceed without power and system metrics collection"
        export DISABLE_TEGRASTATS=1
    fi
else
    export DISABLE_TEGRASTATS=0
fi

# Special handling for jq (JSON parser)
if ! command -v jq &> /dev/null; then
    echo "WARNING: jq command not found, attempting automatic installation..."
    if [[ $EUID -eq 0 ]]; then
        if apt update -y >/dev/null 2>&1 && apt install -y jq >/dev/null 2>&1; then
            echo "SUCCESS: jq installed successfully"
        else
            echo "ERROR: Failed to install jq, JSON parsing will fail"
            exit 1
        fi
    else
        echo "ERROR: jq not found and cannot install without root privileges"
        echo "Please install jq manually: sudo apt install jq"
        exit 1
    fi
fi

# Special handling for yq (YAML parser)
if ! command -v yq &> /dev/null; then
    echo "WARNING: yq command not found, attempting automatic installation..."
    if [[ $EUID -eq 0 ]]; then
        if apt update -y >/dev/null 2>&1 && apt install -y yq >/dev/null 2>&1; then
            echo "SUCCESS: yq installed successfully"
        else
            # Try pip install as fallback
            if pip3 install yq --user >/dev/null 2>&1; then
                export PATH=$PATH:$HOME/.local/bin
                echo "SUCCESS: yq installed via pip successfully"
            else
                echo "ERROR: Failed to install yq, YAML parsing will fail"
                exit 1
            fi
        fi
    else
        # Try pip install for non-root users
        if pip3 install yq --user >/dev/null 2>&1; then
            export PATH=$PATH:$HOME/.local/bin
            echo "SUCCESS: yq installed via pip successfully"
        else
            echo "ERROR: yq not found and cannot install without root privileges"
            echo "Please install yq manually: sudo apt install yq or pip3 install yq"
            exit 1
        fi
    fi
fi

# Check if TensorRT Edge-LLM inference binary exists
INFERENCE_BIN="./build/bin/llm_inference"
if [[ ! -f "$INFERENCE_BIN" ]]; then
    echo "ERROR: LLM inference binary not found at $INFERENCE_BIN"
    echo "Please build the project first using build.sh"
    exit 1
fi

# Check if model engine exists
ENGINE_PATH=$(yq e '.model.tensorrt_engine_path' "$TEST_CONFIG_FILE")
if [[ ! -f "$ENGINE_PATH" ]]; then
    echo "WARNING: TensorRT engine not found at $ENGINE_PATH"
    echo "Test may fail if engine is not present"
fi

# Check Jetson device info
if [[ -f "/proc/device-tree/model" ]]; then
    DEVICE_MODEL=$(tr -d '\0' < /proc/device-tree/model)
    echo "Detected device: $DEVICE_MODEL"
    if [[ ! "$DEVICE_MODEL" == *"AGX Orin"* ]]; then
        echo "WARNING: This test suite is optimized for Jetson AGX Orin"
        echo "Performance results may vary on other devices"
    fi
else
    echo "WARNING: Could not detect Jetson device"
fi

# Verify CUDA availability
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA CUDA is not available"
    exit 1
fi

# Check Python dependencies
REQUIRED_PYTHON_PACKAGES=(
    "numpy"
    "pandas"
    "matplotlib"
    "pyyaml"
    "requests"
)

echo "Checking Python dependencies..."
for PKG in "${REQUIRED_PYTHON_PACKAGES[@]}"; do
    if ! python3 -c "import $PKG" &> /dev/null; then
        echo "ERROR: Required Python package not found: $PKG"
        echo "Install with: pip3 install $PKG"
        exit 1
    fi
done

# Create test data directory if needed
mkdir -p test_data/images

echo "Environment check completed successfully"
exit 0
