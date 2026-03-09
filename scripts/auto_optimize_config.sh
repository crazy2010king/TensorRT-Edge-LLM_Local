#!/bin/bash
set -uo pipefail
# 配置文件自动优化脚本
# 功能：根据当前硬件平台自动生成最优的test_config.yaml配置
# 支持全系列NVIDIA硬件，自动适配最优参数

# ============== 配置参数 ==============
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${BASE_DIR}/config/test_config.yaml"
BACKUP_DIR="${BASE_DIR}/output/backups/config_$(date +%Y%m%d_%H%M%S)"

# ============== 工具函数 ==============
log_info() {
    echo -e "\033[32m[INFO] $*\033[0m"
}

log_warn() {
    echo -e "\033[33m[WARN] $*\033[0m"
}

log_error() {
    echo -e "\033[31m[ERROR] $*\033[0m"
    exit 1
}

# ============== 硬件平台自动检测 ==============
detect_hardware() {
    log_info "🔍 正在自动检测硬件平台配置..."

    # 检测CPU架构
    CPU_ARCH=$(uname -m)
    log_info "检测到CPU架构: ${CPU_ARCH}"

    # 检测CPU核心数
    CPU_CORES=$(nproc)
    log_info "检测到CPU核心数: ${CPU_CORES}"

    # 检测GPU信息
    if command -v nvidia-smi &> /dev/null; then
        # 获取GPU型号名称
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)
        log_info "检测到GPU型号: ${GPU_NAME}"

        # 获取GPU显存大小
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1 | awk '{print $1 $2}')
        log_info "检测到GPU显存: ${GPU_MEM}"

        # 获取GPU计算能力
        GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.' | awk '{print $1}')
        GPU_ARCH="sm_${GPU_COMPUTE_CAP}"
        log_info "检测到GPU计算能力: ${GPU_COMPUTE_CAP:0:1}.${GPU_COMPUTE_CAP:1} → GPU_ARCH=${GPU_ARCH}"
    else
        log_warn "未检测到nvidia-smi，使用默认配置"
        GPU_NAME="Unknown GPU"
        GPU_MEM="16GB"
        GPU_ARCH="sm_87"
        CPU_CORES=8
    fi

    # 识别平台类型
    if [[ "${CPU_ARCH}" == "aarch64" && "${GPU_ARCH}" == "sm_87" ]]; then
        PLATFORM_TYPE="NVIDIA AGX Orin"
        PLATFORM_LEVEL="embedded"
    elif [[ "${CPU_ARCH}" == "aarch64" && "${GPU_ARCH}" == "sm_86" ]]; then
        PLATFORM_TYPE="NVIDIA Jetson Orin NX/Nano"
        PLATFORM_LEVEL="embedded_low"
    elif [[ "${CPU_ARCH}" == "aarch64" && "${GPU_ARCH}" == "sm_72" ]]; then
        PLATFORM_TYPE="NVIDIA Jetson Xavier"
        PLATFORM_LEVEL="embedded_low"
    elif [[ "${CPU_ARCH}" == "x86_64" && "${GPU_ARCH}" == "sm_89" ]]; then
        PLATFORM_TYPE="NVIDIA Ada Lovelace (RTX 40系/L4/L40)"
        PLATFORM_LEVEL="consumer"
    elif [[ "${CPU_ARCH}" == "x86_64" && "${GPU_ARCH}" == "sm_86" ]]; then
        PLATFORM_TYPE="NVIDIA Ampere (RTX 30系/A10)"
        PLATFORM_LEVEL="consumer"
    elif [[ "${CPU_ARCH}" == "x86_64" && "${GPU_ARCH}" == "sm_80" ]]; then
        PLATFORM_TYPE="NVIDIA Ampere (A100/A800)"
        PLATFORM_LEVEL="datacenter"
    elif [[ "${CPU_ARCH}" == "x86_64" && "${GPU_ARCH}" == "sm_90" ]]; then
        PLATFORM_TYPE="NVIDIA Hopper (H100/H800/H200)"
        PLATFORM_LEVEL="datacenter_high"
    elif [[ "${CPU_ARCH}" == "x86_64" && "${GPU_ARCH}" == "sm_100" ]]; then
        if echo "${GPU_NAME}" | grep -qi "b200\|gb200"; then
            PLATFORM_TYPE="NVIDIA Blackwell (B200/GB200)"
        else
            PLATFORM_TYPE="NVIDIA Blackwell (B100/B200/GB200)"
        fi
        PLATFORM_LEVEL="flagship"
    elif [[ "${CPU_ARCH}" == "x86_64" && ("${GPU_ARCH}" == "sm_120" || "${GPU_ARCH}" == "sm_121") ]]; then
        PLATFORM_TYPE="NVIDIA Thor (索尔) 下一代旗舰"
        PLATFORM_LEVEL="next_gen"
    else
        PLATFORM_TYPE="通用平台"
        PLATFORM_LEVEL="general"
    fi

    log_info "✅ 识别到平台类型: ${PLATFORM_TYPE}"
    log_info "✅ 平台性能等级: ${PLATFORM_LEVEL}"
    echo ""
}

# ============== 根据平台生成最优配置 ==============
generate_optimal_config() {
    log_info "⚙️  正在生成${PLATFORM_TYPE}平台最优配置..."

    # 基础参数默认值
    MAX_BATCH_SIZE=4
    MAX_INPUT_LEN=4096
    MAX_OUTPUT_LEN=1024
    PROCESSING_THREADS=$((CPU_CORES > 8 ? 8 : CPU_CORES))
    KV_CACHE_PAGED="true"
    KV_CACHE_QUANTIZATION="fp8"
    CUDA_GRAPH="true"
    EAGLE_ENABLE="true"
    EAGLE_DRAFT_STEP=3
    EAGLE_DRAFT_TOP_K=4
    CONCURRENCY_LEVELS="[1, 2, 4, 8]"
    POWER_TEST_ENABLED="true"

    # 根据平台等级调整参数
    case ${PLATFORM_LEVEL} in
        "embedded_low") # Jetson Orin NX/Nano/Xavier < 16GB显存
            MAX_BATCH_SIZE=2
            MAX_INPUT_LEN=2048
            MAX_OUTPUT_LEN=512
            PROCESSING_THREADS=4
            CONCURRENCY_LEVELS="[1, 2, 4]"
            EAGLE_DRAFT_STEP=2
            ;;
        "embedded") # AGX Orin 32GB/64GB
            MAX_BATCH_SIZE=4
            MAX_INPUT_LEN=4096
            MAX_OUTPUT_LEN=1024
            PROCESSING_THREADS=8
            CONCURRENCY_LEVELS="[1, 2, 4, 8]"
            ;;
        "consumer") # RTX 30/40系，L4/L40
            MAX_BATCH_SIZE=8
            MAX_INPUT_LEN=8192
            MAX_OUTPUT_LEN=2048
            PROCESSING_THREADS=16
            CONCURRENCY_LEVELS="[1, 2, 4, 8, 16]"
            EAGLE_DRAFT_STEP=4
            ;;
        "datacenter") # A100/A800
            MAX_BATCH_SIZE=32
            MAX_INPUT_LEN=16384
            MAX_OUTPUT_LEN=4096
            PROCESSING_THREADS=32
            CONCURRENCY_LEVELS="[1, 2, 4, 8, 16, 32, 64]"
            EAGLE_DRAFT_STEP=5
            EAGLE_DRAFT_TOP_K=6
            ;;
        "datacenter_high") # H100/H800
            MAX_BATCH_SIZE=64
            MAX_INPUT_LEN=32768
            MAX_OUTPUT_LEN=8192
            PROCESSING_THREADS=32
            CONCURRENCY_LEVELS="[1, 2, 4, 8, 16, 32, 64, 128]"
            EAGLE_DRAFT_STEP=6
            EAGLE_DRAFT_TOP_K=8
            ;;
        "flagship") # B200/GB200 Blackwell
            MAX_BATCH_SIZE=128
            MAX_INPUT_LEN=65536
            MAX_OUTPUT_LEN=16384
            PROCESSING_THREADS=64
            CONCURRENCY_LEVELS="[1, 2, 4, 8, 16, 32, 64, 128, 256]"
            EAGLE_DRAFT_STEP=8
            EAGLE_DRAFT_TOP_K=10
            ;;
        "next_gen") # Thor索尔
            MAX_BATCH_SIZE=256
            MAX_INPUT_LEN=131072
            MAX_OUTPUT_LEN=32768
            PROCESSING_THREADS=128
            CONCURRENCY_LEVELS="[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]"
            EAGLE_DRAFT_STEP=10
            EAGLE_DRAFT_TOP_K=12
            ;;
        *)
            log_warn "未知平台等级，使用通用配置"
            ;;
    esac

    # 显存大小自适应调整（统一转换为GB单位）
    GPU_MEM_NUM=${GPU_MEM//[!0-9]/}
    if [[ "${GPU_MEM}" == *"MiB"* ]]; then
        # 单位是MiB，转换为GB
        GPU_MEM_GB=$((GPU_MEM_NUM / 1024))
    else
        # 单位已经是GB
        GPU_MEM_GB=${GPU_MEM_NUM}
    fi

    if [[ ${GPU_MEM_GB} -lt 8 ]]; then
        MAX_BATCH_SIZE=$((MAX_BATCH_SIZE / 2))
        MAX_INPUT_LEN=$((MAX_INPUT_LEN / 2))
        MAX_OUTPUT_LEN=$((MAX_OUTPUT_LEN / 2))
        log_warn "显存小于8GB，自动降低参数: batch=${MAX_BATCH_SIZE}, input_len=${MAX_INPUT_LEN}"
    elif [[ ${GPU_MEM_GB} -gt 80 ]]; then
        MAX_BATCH_SIZE=$((MAX_BATCH_SIZE * 2))
        MAX_INPUT_LEN=$((MAX_INPUT_LEN * 2))
        log_info "显存大于80GB，自动提升参数: batch=${MAX_BATCH_SIZE}, input_len=${MAX_INPUT_LEN}"
    else
        log_info "显存${GPU_MEM_GB}GB，使用标准参数配置"
    fi

    log_info "📊 最优参数配置:"
    log_info "  - 最大Batch Size: ${MAX_BATCH_SIZE}"
    log_info "  - 最大输入长度: ${MAX_INPUT_LEN}"
    log_info "  - 最大输出长度: ${MAX_OUTPUT_LEN}"
    log_info "  - 处理线程数: ${PROCESSING_THREADS}"
    log_info "  - Eagle预测步数: ${EAGLE_DRAFT_STEP}"
    log_info "  - 并发测试级别: ${CONCURRENCY_LEVELS}"
    echo ""
}

# ============== 写入优化后的配置文件 ==============
write_config() {
    # 备份原配置
    mkdir -p "${BACKUP_DIR}"
    cp -f "${CONFIG_FILE}" "${BACKUP_DIR}/"
    log_info "✅ 原配置已备份到: ${BACKUP_DIR}"

    # 生成新的配置文件
    log_info "✍️  正在写入优化后的配置文件..."
    cat > "${CONFIG_FILE}" << EOF
# Test Configuration for Qwen3-VL-2B-Instruct Performance Test
# TensorRT Edge-LLM Test Suite
# 自动优化生成，适配平台: ${PLATFORM_TYPE}
# 生成时间: $(date)

# Device configuration
device:
  name: "${PLATFORM_TYPE}"
  gpu_memory: "${GPU_MEM}"
  cpu_cores: ${CPU_CORES}
  power_mode: "MAXN"

# Model configuration
model:
  # 基础模型配置
  name: "Qwen3-VL-2B-Instruct"
  precision: "NVFP4"
  tensorrt_engine_path: "./output/engines/Qwen3-VL-2B-Instruct_nvfp4_eagle"
  max_batch_size: ${MAX_BATCH_SIZE}
  max_input_len: ${MAX_INPUT_LEN}
  max_output_len: ${MAX_OUTPUT_LEN}

  # 预处理/后处理优化配置
  async_tokenizer: true
  async_postprocessing: true
  processing_threads: ${PROCESSING_THREADS}

  # KVCache优化配置
  kv_cache_quantization: "${KV_CACHE_QUANTIZATION}"
  kv_cache_block_size: 128
  kv_cache_paged: ${KV_CACHE_PAGED}
  kv_cache_preallocate: true

  # CUDA Graph优化配置
  cuda_graph: ${CUDA_GRAPH}
  cuda_graph_capture_prefill: true
  cuda_graph_reuse_decoding: true

  # Eagle Speculative Decoding configuration
  eagle: ${EAGLE_ENABLE}
  eagle_draft_model_path: "./output/engines/Qwen3-VL-2B-Instruct_eagle3_nvfp4"
  eagle_draft_top_k: ${EAGLE_DRAFT_TOP_K}
  eagle_draft_step: ${EAGLE_DRAFT_STEP}
  eagle_verify_tree_size: $((EAGLE_DRAFT_STEP * 20))

# Test case configurations
test_cases:
  # Single request latency test
  single_request:
    enabled: true
    description: "Single request end-to-end latency test"
    input_file: "test_cases/single_request.json"
    iterations: 10
    metrics: ["latency", "gpu_memory", "cpu_memory", "utilization"]

  # Concurrent requests throughput test
  concurrent_requests:
    enabled: true
    description: "Concurrent requests throughput test"
    input_file: "test_cases/concurrent_requests.json"
    concurrency_levels: ${CONCURRENCY_LEVELS}
    duration_per_level: 30
    metrics: ["throughput", "latency_distribution", "gpu_utilization"]

  # Multi-modal input test
  multi_modal:
    enabled: true
    description: "Multi-modal (image + text) inference test"
    input_file: "test_cases/multi_modal.json"
    image_resolutions: ["640x480", "1024x768", "1920x1080"]
    metrics: ["latency", "gpu_memory", "encoding_time"]

  # Cold start test
  cold_start:
    enabled: true
    description: "Engine cold start and initialization test"
    iterations: 5
    unload_engine_between_runs: true
    metrics: ["initialization_time", "first_token_latency"]

  # Stability test
  stability:
    enabled: true
    description: "Long-running stability test"
    duration: 300
    request_interval: 1
    metrics: ["latency_degradation", "memory_leak", "error_rate"]

  # Power efficiency test
  power_efficiency:
    enabled: ${POWER_TEST_ENABLED}
    description: "Power consumption and efficiency test"
    duration: 60
    metrics: ["power_consumption", "energy_per_token", "efficiency_ratio"]

  # Input length variation test
  input_length_variation:
    enabled: true
    description: "Performance with different input token lengths"
    input_lengths: [128, 256, 512, 1024, $((MAX_INPUT_LEN / 4))]
    output_length: 128
    metrics: ["prefill_time", "latency", "throughput"]

  # Output length variation test
  output_length_variation:
    enabled: true
    description: "Performance with different output token lengths"
    input_length: 128
    output_lengths: [32, 64, 128, 512, $((MAX_OUTPUT_LEN / 2))]
    metrics: ["generation_time", "token_rate", "latency"]

  # Image resolution variation test
  image_resolution_variation:
    enabled: true
    description: "Performance with different image resolutions"
    resolutions: ["640x480", "1024x768", "1920x1080", "2560x1440"]
    metrics: ["image_encoding_time", "total_latency", "gpu_memory"]

  # Parameter sensitivity test
  parameter_sensitivity:
    enabled: true
    description: "Performance impact of different inference parameters"
    parameters:
      temperature: [0.1, 0.7, 1.0]
      top_p: [0.1, 0.5, 0.9]
      max_new_tokens: [64, 128, 256]
    metrics: ["token_rate", "latency", "generation_quality"]

# Metrics collection configuration
metrics:
  collection_interval: 100ms
  gpu_metrics:
    - utilization
    - memory_used
    - memory_free
    - power_draw
    - temperature
  cpu_metrics:
    - utilization_per_core
    - memory_used
    - memory_free
    - load_average
  inference_metrics:
    - total_latency
    - prefill_latency
    - generation_latency
    - first_token_latency
    - tokens_per_second
    - input_tokens
    - output_tokens
    - error_count

# Report configuration
report:
  formats: ["markdown", "html", "json"]
  include_charts: true
  include_statistics:
    - mean
    - median
    - p50
    - p90
    - p95
    - p99
    - min
    - max
    - std_dev
  compare_with_baseline: true
  baseline_file: "config/baseline_performance.yaml"
EOF

    log_info "✅ 配置文件已优化完成: ${CONFIG_FILE}"
    echo ""
}

# ============== 主流程 ==============
main() {
    echo "=================================================="
    echo "🚀 配置文件自动优化脚本启动"
    echo "=================================================="
    echo ""

    detect_hardware
    generate_optimal_config
    write_config

    echo "=================================================="
    echo "🎉 配置优化完成!"
    echo "=================================================="
    log_info "适配平台: ${PLATFORM_TYPE}"
    log_info "原配置备份: ${BACKUP_DIR}"
    log_info "当前配置: ${CONFIG_FILE}"
    echo ""
    log_info "💡 下一步操作:"
    log_info "  1. 确认配置参数是否符合预期"
    log_info "  2. 运行性能测试: bash performance_test.sh -t full"
    echo "=================================================="
}

main "$@"
exit 0
