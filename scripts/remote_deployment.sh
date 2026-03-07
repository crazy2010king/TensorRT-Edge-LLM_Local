#!/bin/bash
# 远端设备部署脚本
# 用于在 Jetson AGX Orin 64GB 上部署和测试 Qwen3-VL-2B-Instruct 模型

# 配色方案
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 命令检查函数
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 目录检查函数
dir_exists() {
    if [ -d "$1" ]; then
        return 0
    else
        return 1
    fi
}

# 文件检查函数
file_exists() {
    if [ -f "$1" ]; then
        return 0
    else
        return 1
    fi
}

# 清理函数
cleanup() {
    log_info "正在清理临时资源..."
    # 删除临时输入配置文件
    if [ -n "$INPUT_CONFIG" ] && file_exists "$INPUT_CONFIG"; then
        rm -f "$INPUT_CONFIG"
    fi
}

# 信号处理
trap 'log_info "接收到中断信号，正在终止..."; cleanup; exit 1' INT TERM

# 默认配置
PROJECT_DIR="/home/nvidia/work_dev/wqq/nfs/test_cc_dev/TensorRT-Edge-LLM"
MODEL_DIR="/home/nvidia/work_dev/wqq/nfs/test_cc_dev/Qwen3-VL-2B-Instruct"
OUTPUT_DIR="$PROJECT_DIR/output"
LLM_OUTPUT_DIR="$OUTPUT_DIR/llm"
VISUAL_OUTPUT_DIR="$OUTPUT_DIR/visual"
ENGINES_DIR="$PROJECT_DIR/engines"
LLM_ENGINE_DIR="$ENGINES_DIR/llm"
VISUAL_ENGINE_DIR="$ENGINES_DIR/visual"
EXAMPLES_DIR="$PROJECT_DIR/examples"
INPUT_PATH="$EXAMPLES_DIR/multimodal/pics/giant_panda.jpeg"
OUTPUT_PATH="$PROJECT_DIR/test_output.json"
NUM_THREADS=$(nproc)  # 检测CPU核心数
RUN_PERFORMANCE_TEST=false  # 是否运行性能测试

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-dir)
            PROJECT_DIR="$2"
            OUTPUT_DIR="$PROJECT_DIR/output"
            LLM_OUTPUT_DIR="$OUTPUT_DIR/llm"
            VISUAL_OUTPUT_DIR="$OUTPUT_DIR/visual"
            ENGINES_DIR="$PROJECT_DIR/engines"
            LLM_ENGINE_DIR="$ENGINES_DIR/llm"
            VISUAL_ENGINE_DIR="$ENGINES_DIR/visual"
            EXAMPLES_DIR="$PROJECT_DIR/examples"
            INPUT_PATH="$EXAMPLES_DIR/multimodal/pics/giant_panda.jpeg"
            OUTPUT_PATH="$PROJECT_DIR/test_output.json"
            shift; shift
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift; shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            LLM_OUTPUT_DIR="$OUTPUT_DIR/llm"
            VISUAL_OUTPUT_DIR="$OUTPUT_DIR/visual"
            shift; shift
            ;;
        --engine-dir)
            ENGINES_DIR="$2"
            LLM_ENGINE_DIR="$ENGINES_DIR/llm"
            VISUAL_ENGINE_DIR="$ENGINES_DIR/visual"
            shift; shift
            ;;
        --threads)
            NUM_THREADS="$2"
            shift; shift
            ;;
        --run-performance-test)
            RUN_PERFORMANCE_TEST=true
            shift
            ;;
        --help)
            echo "使用说明: $0 [选项]"
            echo "  --project-dir <路径>    指定项目目录 (默认: $PROJECT_DIR)"
            echo "  --model-dir <路径>      指定模型目录 (默认: $MODEL_DIR)"
            echo "  --output-dir <路径>     指定模型转换输出目录 (默认: $OUTPUT_DIR)"
            echo "  --engine-dir <路径>     指定引擎输出目录 (默认: $ENGINES_DIR)"
            echo "  --threads <数量>        指定构建线程数 (默认: $NUM_THREADS)"
            echo "  --run-performance-test  运行完整性能测试套件"
            echo "  --help                  显示帮助信息"
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查系统依赖
check_system_dependencies() {
    log_info "检查系统依赖..."

    # 检查 CUDA（在 Jetson 设备上可能没有 nvcc，但 CUDA 可能仍然可用）
    if ! command_exists "nvcc"; then
        log_warning "CUDA 编译器 nvcc 未找到，这在 Jetson 设备上是常见的"
        # 尝试通过其他方式检查 CUDA 是否可用
        if command_exists "nvidia-smi"; then
            log_info "nvidia-smi 命令可用，CUDA 可能已安装"
        else
            log_warning "无法确定 CUDA 状态"
        fi
    else
        log_info "CUDA 编译器 nvcc 已找到"
    fi

    # 检查 CMake
    if ! command_exists "cmake"; then
        log_error "CMake 未找到"
        return 1
    fi

    # 检查 make
    if ! command_exists "make"; then
        log_error "make 未找到"
        return 1
    fi

    # 检查 TensorRT（在 Jetson 设备上可能没有 trtexec）
    if ! command_exists "trtexec"; then
        log_warning "TensorRT 执行器 trtexec 未找到，这在 Jetson 设备上是常见的"
    fi

    log_success "系统依赖检查完成"
    return 0
}

# 检查项目环境
check_project_environment() {
    log_info "检查项目环境..."

    # 检查项目目录
    if ! dir_exists "$PROJECT_DIR"; then
        log_error "项目目录不存在: $PROJECT_DIR"
        return 1
    fi

    # 检查模型目录
    if ! dir_exists "$MODEL_DIR"; then
        log_error "模型目录不存在: $MODEL_DIR"
        return 1
    fi

    # 检查转换后的模型是否存在
    if ! dir_exists "$LLM_OUTPUT_DIR" || ! dir_exists "$VISUAL_OUTPUT_DIR"; then
        log_error "转换后的模型目录不存在"
        return 1
    fi

    if ! file_exists "$LLM_OUTPUT_DIR/model.onnx" || ! file_exists "$VISUAL_OUTPUT_DIR/model.onnx"; then
        log_error "转换后的模型文件不存在"
        return 1
    fi

    log_success "项目环境检查完成"
    return 0
}

# 创建引擎目录
create_engine_directories() {
    log_info "创建引擎目录..."

    if mkdir -p "$LLM_ENGINE_DIR" "$VISUAL_ENGINE_DIR"; then
        log_success "引擎目录创建成功"
    else
        log_error "引擎目录创建失败"
        return 1
    fi

    return 0
}

# 构建 llm_build 可执行文件
build_llm_builder() {
    log_info "检查并构建 llm_build 可执行文件..."

    if file_exists "$PROJECT_DIR/build/examples/llm/llm_build"; then
        log_info "llm_build 已存在，跳过构建"
        return 0
    fi

    log_info "构建 llm_build 可执行文件..."

    # 确保 build 目录存在
    mkdir -p "$PROJECT_DIR/build"

    # 进入 build 目录
    cd "$PROJECT_DIR/build" || return 1

    # 运行 CMake 配置，明确指定 CUDA 编译器路径和 TRT 包目录
    log_info "运行 CMake 配置..."
    if ! cmake .. -DTRT_EDGELLM_BUILD_EXAMPLES=ON -DTRT_EDGELLM_BUILD_TESTS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -DTRT_PACKAGE_DIR=/usr -DCUDA_VERSION=12.6 -DCUDA_DIR=/usr/local/cuda-12.6; then
        log_error "CMake 配置失败"
        return 1
    fi

    # 构建 llm_build
    log_info "编译 llm_build (使用 $NUM_THREADS 个线程)..."
    if ! make -j"$NUM_THREADS" llm_build; then
        log_error "llm_build 编译失败"
        return 1
    fi

    log_success "llm_build 构建成功"
    return 0
}

# 构建视觉模型构建器
build_visual_builder() {
    log_info "检查并构建 visual_build 可执行文件..."

    if file_exists "$PROJECT_DIR/build/examples/multimodal/visual_build"; then
        log_info "visual_build 已存在，跳过构建"
        return 0
    fi

    log_info "构建 visual_build 可执行文件..."

    cd "$PROJECT_DIR/build" || return 1

    log_info "编译 visual_build (使用 $NUM_THREADS 个线程)..."
    if ! make -j"$NUM_THREADS" visual_build; then
        log_error "visual_build 编译失败"
        return 1
    fi

    log_success "visual_build 构建成功"
    return 0
}

# 构建 llm_inference 可执行文件
build_llm_inference() {
    log_info "检查并构建 llm_inference 可执行文件..."

    if file_exists "$PROJECT_DIR/build/examples/llm/llm_inference"; then
        log_info "llm_inference 已存在，跳过构建"
        return 0
    fi

    log_info "构建 llm_inference 可执行文件..."

    cd "$PROJECT_DIR/build" || return 1

    log_info "编译 llm_inference (使用 $NUM_THREADS 个线程)..."
    if ! make -j"$NUM_THREADS" llm_inference; then
        log_error "llm_inference 编译失败"
        return 1
    fi

    log_success "llm_inference 构建成功"
    return 0
}

# 构建 LLM 引擎
build_llm_engine() {
    log_info "构建 LLM 引擎..."

    cd "$PROJECT_DIR" || return 1

    ./build/examples/llm/llm_build \
        --onnxDir "$LLM_OUTPUT_DIR" \
        --engineDir "$LLM_ENGINE_DIR" \
        --maxInputLen 1024 \
        --maxKVCacheCapacity 4096 \
        --maxBatchSize 1 \
        --debug

    if [ $? -ne 0 ]; then
        log_error "LLM 引擎构建失败"
        return 1
    fi

    if file_exists "$LLM_ENGINE_DIR/llm.engine"; then
        log_success "LLM 引擎构建成功"
    else
        log_error "LLM 引擎文件不存在"
        return 1
    fi

    return 0
}

# 构建视觉引擎
build_visual_engine() {
    log_info "构建视觉引擎..."

    cd "$PROJECT_DIR" || return 1

    ./build/examples/multimodal/visual_build \
        --onnxDir "$VISUAL_OUTPUT_DIR" \
        --engineDir "$VISUAL_ENGINE_DIR" \
        --minImageTokens 4 \
        --maxImageTokens 1024 \
        --maxImageTokensPerImage 512 \
        --debug

    if [ $? -ne 0 ]; then
        log_error "视觉引擎构建失败"
        return 1
    fi

    if file_exists "$VISUAL_ENGINE_DIR/visual.engine"; then
        log_success "视觉引擎构建成功"
    else
        log_error "视觉引擎文件不存在"
        return 1
    fi

    return 0
}

# 运行推理测试
run_inference_test() {
    log_info "运行推理测试..."

    cd "$PROJECT_DIR" || return 1

    # 创建输入配置文件
    INPUT_CONFIG=$(mktemp /tmp/test_input.XXXXXX.json)
    cat > "$INPUT_CONFIG" <<EOF
{
    "requests": [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请描述这张图片"
                        },
                        {
                            "type": "image",
                            "image": "$INPUT_PATH"
                        }
                    ]
                }
            ]
        }
    ],
    "parameters": {
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": false
    }
}
EOF

    # 运行推理
    ./build/examples/llm/llm_inference \
        --inputFile "$INPUT_CONFIG" \
        --engineDir "$LLM_ENGINE_DIR" \
        --multimodalEngineDir "$VISUAL_ENGINE_DIR" \
        --outputFile "$OUTPUT_PATH" \
        --debug

    if [ $? -ne 0 ]; then
        log_error "推理测试失败"
        return 1
    fi

    # 验证输出
    if ! file_exists "$OUTPUT_PATH"; then
        log_error "推理输出文件不存在"
        return 1
    fi

    log_success "推理测试成功"
    log_info "输出结果:"
    cat "$OUTPUT_PATH"

    return 0
}

# 运行单元测试
run_unit_tests() {
    log_info "运行单元测试..."

    if file_exists "$PROJECT_DIR/build/unittests/unit_tests"; then
        log_info "执行单元测试..."
        cd "$PROJECT_DIR" || return 1
        ./build/unittests/unit_tests
        if [ $? -ne 0 ]; then
            log_warning "单元测试失败"
        else
            log_success "单元测试通过"
        fi
    else
        log_info "单元测试可执行文件未找到，跳过"
    fi

    return 0
}

# 主函数
main() {
    log_info "=============================================="
    log_info "Qwen3-VL-2B-Instruct 部署运行测试自动化流程"
    log_info "目标设备: Jetson AGX Orin 64GB"
    log_info "项目路径: $PROJECT_DIR"
    log_info "模型路径: $MODEL_DIR"
    log_info "线程数: $NUM_THREADS"
    log_info "=============================================="

    # 检查系统依赖
    if ! check_system_dependencies; then
        cleanup
        exit 1
    fi

    # 检查项目环境
    if ! check_project_environment; then
        cleanup
        exit 1
    fi

    # 创建引擎目录
    if ! create_engine_directories; then
        cleanup
        exit 1
    fi

    # 构建可执行文件
    if ! build_llm_builder; then
        cleanup
        exit 1
    fi

    if ! build_visual_builder; then
        cleanup
        exit 1
    fi

    if ! build_llm_inference; then
        cleanup
        exit 1
    fi

    # 构建引擎
    if ! build_llm_engine; then
        cleanup
        exit 1
    fi

    if ! build_visual_engine; then
        cleanup
        exit 1
    fi

    # 运行推理测试
    if ! run_inference_test; then
        cleanup
        exit 1
    fi

    # 运行单元测试
    run_unit_tests

    # 运行性能测试（如果启用）
    if [ "$RUN_PERFORMANCE_TEST" = true ]; then
        log_info "=============================================="
        log_info "开始运行性能测试套件..."
        cd "$PROJECT_DIR" || return 1

        # 运行性能测试脚本
        if ./performance_test.sh --test-suite basic; then
            log_success "性能测试完成"
            # 查找最新的测试报告
            LATEST_REPORT=$(ls -td "$PROJECT_DIR/results/reports/"* | head -1)
            if [ -n "$LATEST_REPORT" ]; then
                log_info "性能测试报告已生成: $LATEST_REPORT"
            fi
        else
            log_error "性能测试执行失败"
        fi
    fi

    log_info "=============================================="
    log_success "部署和测试完成"
    log_info "输出文件: $OUTPUT_PATH"
    log_info "引擎目录: $ENGINES_DIR"
    if [ "$RUN_PERFORMANCE_TEST" = true ] && [ -n "$LATEST_REPORT" ]; then
        log_info "性能报告: $LATEST_REPORT"
    fi
    log_info "=============================================="

    cleanup
    return 0
}

# 执行主函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
