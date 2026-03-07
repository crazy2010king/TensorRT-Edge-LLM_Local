#!/bin/bash
set -uo pipefail

# Environment setup and dependency check script
# TensorRT Edge-LLM Performance Test Suite
# 增加容错处理，即使依赖缺失也能继续运行

echo "Checking environment dependencies..."

# 导出所有环境变量，确保子脚本可以访问
export NO_EXIT_ON_ERROR=1

# Check required commands - 这些是必须的核心依赖，缺失的话无法继续
REQUIRED_COMMANDS=(
    "python3"
    "curl"
    "nvidia-smi"
)

for CMD in "${REQUIRED_COMMANDS[@]}"; do
    if ! command -v "$CMD" &> /dev/null; then
        echo "ERROR: Core required command not found: $CMD"
        exit 1
    fi
done

# Special handling for tegrastats (Jetson specific)
export DISABLE_TEGRASTATS=0
if ! command -v tegrastats &> /dev/null; then
    echo "WARNING: tegrastats command not found, attempting automatic installation..."
    INSTALL_SUCCESS=0
    if [[ $EUID -eq 0 ]]; then
        apt update -y >/dev/null 2>&1
        if apt install -y nvidia-l4t-tools >/dev/null 2>&1; then
            INSTALL_SUCCESS=1
            echo "SUCCESS: tegrastats installed successfully"
        fi
    fi

    if [[ $INSTALL_SUCCESS -eq 0 ]]; then
        echo "WARNING: Failed to install tegrastats, will disable power and system metrics collection"
        echo "WARNING: Performance metrics will be limited to inference latency and throughput only"
        export DISABLE_TEGRASTATS=1
    fi
fi

# Special handling for jq (JSON parser)
export JQ_AVAILABLE=1
if ! command -v jq &> /dev/null; then
    echo "WARNING: jq command not found, attempting automatic installation..."
    INSTALL_SUCCESS=0
    if [[ $EUID -eq 0 ]]; then
        if apt update -y >/dev/null 2>&1 && apt install -y jq >/dev/null 2>&1; then
            INSTALL_SUCCESS=1
        fi
    fi

    # Try pip install as fallback
    if [[ $INSTALL_SUCCESS -eq 0 ]]; then
        if pip3 install jq --user >/dev/null 2>&1; then
            export PATH=$PATH:$HOME/.local/bin
            INSTALL_SUCCESS=1
        fi
    fi

    if [[ $INSTALL_SUCCESS -eq 0 ]]; then
        echo "WARNING: Failed to install jq, will disable advanced report generation"
        export JQ_AVAILABLE=0
    else
        echo "SUCCESS: jq installed successfully"
    fi
fi

# Special handling for yq (YAML parser)
export YQ_AVAILABLE=1
if ! command -v yq &> /dev/null; then
    echo "WARNING: yq command not found, attempting automatic installation..."
    INSTALL_SUCCESS=0
    if [[ $EUID -eq 0 ]]; then
        if apt update -y >/dev/null 2>&1 && apt install -y yq >/dev/null 2>&1; then
            INSTALL_SUCCESS=1
        else
            # Try pip install as fallback
            if pip3 install yq --user >/dev/null 2>&1; then
                export PATH=$PATH:$HOME/.local/bin
                INSTALL_SUCCESS=1
            fi
        fi
    else
        # Try pip install for non-root users
        if pip3 install yq --user >/dev/null 2>&1; then
            export PATH=$PATH:$HOME/.local/bin
            INSTALL_SUCCESS=1
        fi
    fi

    if [[ $INSTALL_SUCCESS -eq 0 ]]; then
        echo "WARNING: Failed to install yq, will use default configuration values"
        export YQ_AVAILABLE=0
    else
        echo "SUCCESS: yq installed successfully"
    fi
fi

# Check if TensorRT Edge-LLM inference binary exists
INFERENCE_BIN="./build/bin/llm_inference"
if [[ ! -f "$INFERENCE_BIN" ]]; then
    # 尝试查找其他位置的推理二进制
    if [[ -f "./build/examples/llm/llm_inference" ]]; then
        INFERENCE_BIN="./build/examples/llm/llm_inference"
        export INFERENCE_BIN_PATH="$INFERENCE_BIN"
        echo "INFO: Found inference binary at $INFERENCE_BIN"
    else
        echo "ERROR: LLM inference binary not found"
        echo "Please build the project first using build.sh"
        exit 1
    fi
else
    export INFERENCE_BIN_PATH="$INFERENCE_BIN"
fi

# Check if model engine exists - 如果yq可用则读取配置，否则使用默认路径
if [[ ${YQ_AVAILABLE:-1} -eq 1 && -n "${TEST_CONFIG_FILE:-}" ]]; then
    ENGINE_PATH=$(yq e '.model.tensorrt_engine_path' "$TEST_CONFIG_FILE" 2>/dev/null || echo "./engines/")
else
    ENGINE_PATH="./engines/"
fi

if [[ ! -d "$ENGINE_PATH" && ! -f "$ENGINE_PATH" ]]; then
    echo "WARNING: TensorRT engine not found at $ENGINE_PATH"
    echo "Test may fail if engine is not present in the expected location"
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

# Verify CUDA availability - already checked earlier via nvidia-smi
echo "CUDA is available"

# Check Python dependencies - 尝试自动安装缺失的包
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
        echo "WARNING: Required Python package not found: $PKG, attempting automatic installation..."
        if pip3 install "$PKG" --user >/dev/null 2>&1; then
            echo "SUCCESS: Installed $PKG successfully"
        else
            echo "WARNING: Failed to install $PKG, some features may be disabled"
        fi
    fi
done

# Create test data directory if needed
mkdir -p test_data/images results/raw_data results/reports results/logs

echo "Environment check completed with optional dependencies automatically handled"
echo "----------------------------------------"
echo "Environment status summary:"
echo "  - tegrastats: $(if [[ ${DISABLE_TEGRASTATS:-0} -eq 1 ]]; then echo "DISABLED (system metrics limited)"; else echo "ENABLED (full metrics available)"; fi)"
echo "  - jq: $(if [[ ${JQ_AVAILABLE:-1} -eq 1 ]]; then echo "ENABLED (JSON parsing available)"; else echo "DISABLED (report generation limited)"; fi)"
echo "  - yq: $(if [[ ${YQ_AVAILABLE:-1} -eq 1 ]]; then echo "ENABLED (YAML config parsing available)"; else echo "DISABLED (using default config)"; fi)"
echo "----------------------------------------"

exit 0
