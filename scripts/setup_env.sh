#!/bin/bash
set -euo pipefail

# Environment setup and dependency check script
# TensorRT Edge-LLM Performance Test Suite

echo "Checking environment dependencies..."

# Check required commands
REQUIRED_COMMANDS=(
    "python3"
    "jq"
    "tegrastats"
    "yq"
    "curl"
)

for CMD in "${REQUIRED_COMMANDS[@]}"; do
    if ! command -v "$CMD" &> /dev/null; then
        echo "ERROR: Required command not found: $CMD"
        exit 1
    fi
done

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
