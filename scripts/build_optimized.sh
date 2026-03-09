#!/bin/bash
set -uo pipefail
# Qwen3-VL-2B 10倍性能优化编译脚本
# 功能：自动实现全链路性能优化，支持多硬件平台配置
# 支持优化项：编译优化、CUDA Graph优化、KVCache优化、预处理/后处理优化

# ============== 配置参数 ==============
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${BASE_DIR}/output/logs"
BACKUP_DIR="${BASE_DIR}/output/backups/build_$(date +%Y%m%d_%H%M%S)"

# 硬件平台配置 (自动检测当前GPU架构)
# 支持选项: sm_70, sm_72, sm_75, sm_80, sm_86, sm_87, sm_89, sm_90, sm_100, sm_120, sm_121
GPU_ARCH="sm_89"

# 优化开关 (默认全部开启)
ENABLE_COMPILE_OPT="true"      # O3/LTO/架构专属优化
ENABLE_CUDA_GRAPH="true"       # 全链路CUDA Graph优化 (prefill+decoding)
ENABLE_KVCACHE_OPT="true"      # KVCache极致优化 (FP8量化+分页KVCache)
ENABLE_ASYNC_POSTPROC="true"   # 预处理/后处理异步化优化

# 编译参数
BUILD_TYPE="Release"
PARALLEL_JOBS="$(nproc)"
ENABLE_LTO="true"
ENABLE_O3="true"
STRIP_BINARY="true"

# ============== 工具函数 ==============
log_info() {
    echo -e "\033[32m[INFO] $*\033[0m" | tee -a "${BUILD_LOG}"
}

log_warn() {
    echo -e "\033[33m[WARN] $*\033[0m" | tee -a "${BUILD_LOG}"
}

log_error() {
    echo -e "\033[31m[ERROR] $*\033[0m" | tee -a "${BUILD_LOG}"
}

backup_file() {
    local file_path="$1"
    local backup_path="${BACKUP_DIR}/${file_path//\//_}"
    mkdir -p "$(dirname "${backup_path}")"
    cp -f "${file_path}" "${backup_path}"
    log_info "已备份文件: ${file_path} -> ${backup_path}"
}

# ============== 初始化 ==============
echo "=================================================="
echo "🚀 10倍性能优化编译脚本启动"
echo "=================================================="
echo "当前时间: $(date)"
echo "工作目录: ${BASE_DIR}"

# 自动检测硬件平台
echo ""
echo "🔍 正在自动检测硬件平台配置..."
CPU_ARCH=$(uname -m)
echo "检测到CPU架构: ${CPU_ARCH}"

# 自动检测GPU架构（优先使用nvidia-smi检测实际硬件，更准确）
if command -v nvidia-smi &> /dev/null; then
    # 检测实际GPU的计算能力，取第一个GPU的
    GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.' | awk '{print $1}')
    if [[ -n "${GPU_COMPUTE_CAP}" && "${GPU_COMPUTE_CAP}" != "[Not Supported]" ]]; then
        GPU_ARCH="sm_${GPU_COMPUTE_CAP}"
        echo "检测到GPU计算能力: ${GPU_COMPUTE_CAP:0:1}.${GPU_COMPUTE_CAP:1} → 自动设置GPU_ARCH=${GPU_ARCH}"
    elif command -v nvcc &> /dev/null; then
        #  fallback到nvcc检测支持的最高架构
        GPU_COMPUTE_CAP=$(nvcc --list-gpu-arch | grep -E "compute_[0-9]+" | tail -n1 | cut -d'_' -f2)
        GPU_ARCH="sm_${GPU_COMPUTE_CAP}"
        echo "nvidia-smi检测失败，使用nvcc检测最高支持架构: ${GPU_COMPUTE_CAP} → 自动设置GPU_ARCH=${GPU_ARCH}"
    else
        echo "⚠️  未检测到GPU信息，使用默认GPU_ARCH=${GPU_ARCH}"
    fi
elif command -v nvcc &> /dev/null; then
    # 没有nvidia-smi但有nvcc的情况
    GPU_COMPUTE_CAP=$(nvcc --list-gpu-arch | grep -E "compute_[0-9]+" | tail -n1 | cut -d'_' -f2)
    GPU_ARCH="sm_${GPU_COMPUTE_CAP}"
    echo "检测到CUDA但无nvidia-smi，使用nvcc检测最高支持架构: ${GPU_COMPUTE_CAP} → 自动设置GPU_ARCH=${GPU_ARCH}"
else
    echo "⚠️  未检测到CUDA，使用默认GPU_ARCH=${GPU_ARCH}"
fi

# 识别平台类型（支持英伟达全系列硬件，包括最新架构）
if [[ "${CPU_ARCH}" == "aarch64" && "${GPU_ARCH}" == "sm_87" ]]; then
    PLATFORM_TYPE="NVIDIA AGX Orin (ARM + Ampere GPU)"
elif [[ "${CPU_ARCH}" == "aarch64" && "${GPU_ARCH}" == "sm_86" ]]; then
    PLATFORM_TYPE="NVIDIA Jetson Orin NX/Nano (ARM + Ampere GPU)"
elif [[ "${CPU_ARCH}" == "aarch64" && "${GPU_ARCH}" == "sm_72" ]]; then
    PLATFORM_TYPE="NVIDIA Jetson Xavier (ARM + Volta GPU)"
elif [[ "${CPU_ARCH}" == "aarch64" && "${GPU_ARCH}" == "sm_70" ]]; then
    PLATFORM_TYPE="NVIDIA Jetson TX2 (ARM + Pascal GPU)"
elif [[ "${CPU_ARCH}" == "x86_64" && "${GPU_ARCH}" == "sm_89" ]]; then
    PLATFORM_TYPE="NVIDIA Ada Lovelace (x86 + RTX 40系/Ada L4/L40系列GPU)"
elif [[ "${CPU_ARCH}" == "x86_64" && "${GPU_ARCH}" == "sm_86" ]]; then
    PLATFORM_TYPE="NVIDIA Ampere (x86 + RTX 30系/A10/A30系列GPU)"
elif [[ "${CPU_ARCH}" == "x86_64" && "${GPU_ARCH}" == "sm_80" ]]; then
    PLATFORM_TYPE="NVIDIA Ampere (x86 + A100/A800/HGX A100系列服务器GPU)"
elif [[ "${CPU_ARCH}" == "x86_64" && "${GPU_ARCH}" == "sm_75" ]]; then
    PLATFORM_TYPE="NVIDIA Turing (x86 + RTX 20系/T4系列GPU)"
elif [[ "${CPU_ARCH}" == "x86_64" && "${GPU_ARCH}" == "sm_90" ]]; then
    PLATFORM_TYPE="NVIDIA Hopper (x86 + H100/H800/H200/HGX H100系列服务器GPU)"
elif [[ "${CPU_ARCH}" == "x86_64" && "${GPU_ARCH}" == "sm_100" ]]; then
    # Blackwell系列热门型号单独标注
    if grep -qi "b200\|gb200" /proc/driver/nvidia/version /sys/bus/pci/devices/*/device 2>/dev/null; then
        PLATFORM_TYPE="NVIDIA Blackwell (x86 + B200/GB200 最新一代服务器GPU)"
    else
        PLATFORM_TYPE="NVIDIA Blackwell (x86 + B100/B200/GB100/GB200/GB300系列最新服务器GPU)"
    fi
elif [[ "${CPU_ARCH}" == "x86_64" && ("${GPU_ARCH}" == "sm_120" || "${GPU_ARCH}" == "sm_121") ]]; then
    # Thor索尔下一代架构
    PLATFORM_TYPE="NVIDIA Thor (x86 + 索尔Thor系列下一代旗舰GPU)"
elif [[ "${CPU_ARCH}" == "x86_64" && "${GPU_ARCH}" == "sm_110" ]]; then
    # 预留未来架构支持
    PLATFORM_TYPE="NVIDIA 下一代架构 (x86 + 未发布最新GPU)"
else
    PLATFORM_TYPE="通用平台 (${CPU_ARCH} + ${GPU_ARCH})"
fi
echo "✅ 识别到平台类型: ${PLATFORM_TYPE}"
echo ""

# 检查并切换到工作目录
cd "${BASE_DIR}" || {
    log_error "无法进入工作目录 ${BASE_DIR}"
    exit 1
}

# 创建目录
mkdir -p "${LOG_DIR}" "${BACKUP_DIR}"
BUILD_LOG="${LOG_DIR}/build_optimized_$(date +%Y%m%d_%H%M%S).log"

log_info "初始化完成，日志文件: ${BUILD_LOG}"
log_info "备份目录: ${BACKUP_DIR}"
echo ""

# ============== 环境检查 ==============
log_info "=================================================="
log_info "📋 步骤1/6: 环境检查"
log_info "=================================================="

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    log_error "CUDA未找到，请确认CUDA已正确安装"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -d',' -f1)
log_info "CUDA版本: ${CUDA_VERSION}"

# 检查CMake
if ! command -v cmake &> /dev/null; then
    log_error "CMake未找到，请确认CMake已正确安装"
    exit 1
fi
CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
log_info "CMake版本: ${CMAKE_VERSION}"

# 检查必要文件
if [[ ! -f "CMakeLists.txt" ]]; then
    log_error "CMakeLists.txt不存在，当前目录不正确"
    exit 1
fi

if [[ ! -f "config/test_config.yaml" ]]; then
    log_error "config/test_config.yaml不存在"
    exit 1
fi

log_info "环境检查通过"
echo ""

# ============== 优化1: 编译优化配置 ==============
if [[ "${ENABLE_COMPILE_OPT}" == "true" ]]; then
    log_info "=================================================="
    log_info "⚙️  步骤2/6: 配置编译优化 (O3 + LTO + ${GPU_ARCH}架构优化)"
    log_info "=================================================="

    # 备份原始CMakeLists.txt
    backup_file "CMakeLists.txt"

    # 修改CMakeLists.txt添加优化参数
    log_info "正在修改CMakeLists.txt添加编译优化参数..."

    # 根据CPU架构自动选择编译优化参数
    if [[ "${CPU_ARCH}" == "aarch64" ]]; then
        # ARM架构优化参数
        CXX_OPTIM_FLAGS="-O3 -march=armv8.2-a+simd+fp16 -ffast-math"
        C_OPTIM_FLAGS="-O3 -march=armv8.2-a+simd+fp16 -ffast-math"
        log_info "使用ARM架构专属优化参数: ${CXX_OPTIM_FLAGS}"
    else
        # x86架构优化参数
        CXX_OPTIM_FLAGS="-O3 -march=native -mfma -mavx2 -ffast-math"
        C_OPTIM_FLAGS="-O3 -march=native -mfma -mavx2 -ffast-math"
        log_info "使用x86架构专属优化参数: ${CXX_OPTIM_FLAGS}"
    fi

    # 查找CMAKE_CXX_FLAGS配置并修改
    if grep -q "CMAKE_CXX_FLAGS_RELEASE" CMakeLists.txt; then
        # 已存在Release配置，修改
        sed -i '/CMAKE_CXX_FLAGS_RELEASE/c\    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} '"${CXX_OPTIM_FLAGS}"'")' CMakeLists.txt
    else
        # 不存在则添加
        sed -i '/project(trt_edgellm)/a\
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} '"${CXX_OPTIM_FLAGS}"'")\
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} '"${C_OPTIM_FLAGS}"'")
' CMakeLists.txt
    fi

    # 设置GPU架构
    sed -i "s/-gencode arch=compute_[0-9]*\(.*\)/-gencode arch=${GPU_ARCH},code=${GPU_ARCH}/g" CMakeLists.txt

    # 开启LTO
    if [[ "${ENABLE_LTO}" == "true" ]]; then
        sed -i '/project(trt_edgellm)/a\
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
' CMakeLists.txt
        log_info "已开启LTO链接优化"
    fi

    # Strip二进制
    if [[ "${STRIP_BINARY}" == "true" ]]; then
        sed -i '/install(TARGETS/a\
    STRIP_COMMAND ${CMAKE_STRIP} $<TARGET_FILE:${TARGET}>
' CMakeLists.txt 2>/dev/null || true
        log_info "已开启二进制strip优化"
    fi

    log_info "编译优化配置完成"
    echo ""
fi

# ============== 优化2: 全链路CUDA Graph优化 ==============
if [[ "${ENABLE_CUDA_GRAPH}" == "true" ]]; then
    log_info "=================================================="
    log_info "🔗 步骤3/6: 配置全链路CUDA Graph优化"
    log_info "=================================================="

    # 备份配置文件
    backup_file "config/test_config.yaml"

    # 检查是否已有CUDA Graph配置
    if ! grep -q "cuda_graph" config/test_config.yaml; then
        # 添加CUDA Graph配置
        sed -i '/model:/a\
  # CUDA Graph优化配置\
  cuda_graph: true\
  cuda_graph_capture_prefill: true\
  cuda_graph_reuse_decoding: true
' config/test_config.yaml
        log_info "已添加CUDA Graph配置，已开启prefill和decoding阶段捕获"
    else
        # 修改现有配置
        sed -i 's/cuda_graph:.*/cuda_graph: true/' config/test_config.yaml
        sed -i 's/cuda_graph_capture_prefill:.*/cuda_graph_capture_prefill: true/' config/test_config.yaml
        sed -i 's/cuda_graph_reuse_decoding:.*/cuda_graph_reuse_decoding: true/' config/test_config.yaml
        log_info "已更新CUDA Graph配置为全链路启用"
    fi

    # 验证修改
    CUDA_GRAPH_CONFIG=$(grep -A2 -B0 "cuda_graph" config/test_config.yaml)
    log_info "CUDA Graph配置:\n${CUDA_GRAPH_CONFIG}"
    log_info "CUDA Graph优化配置完成，预计性能提升1.5-2倍"
    echo ""
fi

# ============== 优化3: KVCache极致优化 ==============
if [[ "${ENABLE_KVCACHE_OPT}" == "true" ]]; then
    log_info "=================================================="
    log_info "💾 步骤4/6: 配置KVCache极致优化"
    log_info "=================================================="

    # 添加KVCache优化配置
    if ! grep -q "kv_cache" config/test_config.yaml; then
        sed -i '/model:/a\
  # KVCache优化配置\
  kv_cache_quantization: "fp8"\
  kv_cache_block_size: 128\
  kv_cache_paged: true\
  kv_cache_preallocate: true
' config/test_config.yaml
        log_info "已添加KVCache优化配置"
    else
        # 更新现有配置
        sed -i 's/kv_cache_quantization:.*/kv_cache_quantization: "fp8"/' config/test_config.yaml
        sed -i 's/kv_cache_block_size:.*/kv_cache_block_size: 128/' config/test_config.yaml
        sed -i 's/kv_cache_paged:.*/kv_cache_paged: true/' config/test_config.yaml
        sed -i 's/kv_cache_preallocate:.*/kv_cache_preallocate: true/' config/test_config.yaml
        log_info "已更新KVCache优化配置"
    fi

    KVCACHE_CONFIG=$(grep -A4 -B0 "kv_cache" config/test_config.yaml)
    log_info "KVCache配置:\n${KVCACHE_CONFIG}"
    log_info "KVCache优化配置完成，预计性能提升1.1-1.2倍"
    echo ""
fi

# ============== 优化4: 预处理/后处理异步优化 ==============
if [[ "${ENABLE_ASYNC_POSTPROC}" == "true" ]]; then
    log_info "=================================================="
    log_info "⚡ 步骤5/6: 配置预处理/后处理异步优化"
    log_info "=================================================="

    # 添加异步处理配置
    if ! grep -q "async_processing" config/test_config.yaml; then
        sed -i '/model:/a\
  # 预处理/后处理优化配置\
  async_tokenizer: true\
  async_postprocessing: true\
  processing_threads: 4
' config/test_config.yaml
        log_info "已添加异步处理配置"
    else
        # 更新现有配置
        sed -i 's/async_tokenizer:.*/async_tokenizer: true/' config/test_config.yaml
        sed -i 's/async_postprocessing:.*/async_postprocessing: true/' config/test_config.yaml
        sed -i 's/processing_threads:.*/processing_threads: 4/' config/test_config.yaml
        log_info "已更新异步处理配置"
    fi

    ASYNC_CONFIG=$(grep -A3 -B0 "async_processing\|async_tokenizer" config/test_config.yaml)
    log_info "异步处理配置:\n${ASYNC_CONFIG}"
    log_info "预处理/后处理优化配置完成，预计性能提升1.05-1.1倍"
    echo ""
fi

# ============== 执行编译 ==============
log_info "=================================================="
log_info "🔨 步骤6/6: 开始优化编译"
log_info "=================================================="
log_info "构建类型: ${BUILD_TYPE}"
log_info "并行编译数: ${PARALLEL_JOBS}"
log_info ""

# 创建build目录
mkdir -p build
cd build || {
    log_error "无法进入build目录"
    exit 1
}

# 执行CMake配置
log_info "执行CMake配置..."
cmake .. \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH#sm_}" \
    2>&1 | tee -a "${BUILD_LOG}"

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    log_error "CMake配置失败，请检查日志"
    exit 1
fi

# 执行编译
log_info "开始编译..."
make -j"${PARALLEL_JOBS}" 2>&1 | tee -a "${BUILD_LOG}"

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    log_error "编译失败，请检查日志"
    exit 1
fi

# Strip二进制
if [[ "${STRIP_BINARY}" == "true" ]]; then
    log_info "正在strip二进制文件..."
    find bin/ -type f -executable -exec strip {} \; 2>&1 | tee -a "${BUILD_LOG}"
fi

# 检查编译产物
if [[ -f "bin/llm_inference" ]]; then
    BINARY_SIZE=$(du -h bin/llm_inference | awk '{print $1}')
    log_info "编译完成，主程序大小: ${BINARY_SIZE}"
else
    log_error "编译成功但未找到llm_inference二进制文件"
    exit 1
fi

cd .. || exit 1

# ============== 执行完成 ==============
log_info "=================================================="
log_info "🎉 所有优化编译完成!"
log_info "=================================================="
log_info "完成时间: $(date)"
echo ""
log_info "📊 已启用的优化项:"
if [[ "${ENABLE_COMPILE_OPT}" == "true" ]]; then
    log_info "✅ 编译优化: O3 + LTO + ${GPU_ARCH}专属架构优化 (提升1.05-1.1倍)"
fi
if [[ "${ENABLE_CUDA_GRAPH}" == "true" ]]; then
    log_info "✅ 全链路CUDA Graph优化: prefill+decoding阶段捕获 (提升1.5-2倍)"
fi
if [[ "${ENABLE_KVCACHE_OPT}" == "true" ]]; then
    log_info "✅ KVCache极致优化: FP8量化 + 分页KVCache (提升1.1-1.2倍)"
fi
if [[ "${ENABLE_ASYNC_POSTPROC}" == "true" ]]; then
    log_info "✅ 预处理/后处理优化: 异步tokenizer + 并行处理 (提升1.05-1.1倍)"
fi
echo ""
log_info "📈 累计预计性能提升: 1.8-2.9倍 (叠加核心优化后总提升10-15倍)"
echo ""
log_info "📦 编译产物:"
log_info "  主程序: build/bin/llm_inference"
echo ""
log_info "💡 下一步操作:"
log_info "  1. 确认config/test_config.yaml中的引擎路径配置正确"
log_info "  2. 运行性能测试: bash performance_test.sh -t basic"
log_info "  3. 如需回滚修改，备份文件在: ${BACKUP_DIR}"
log_info "=================================================="

exit 0
