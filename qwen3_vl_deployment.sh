#!/bin/bash
# Qwen3-VL-2B-Instruct 模型在Jetson AGX Orin上的完整部署运行测试自动化流程

# 配色方案
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目配置
MODEL_NAME="Qwen3-VL-2B-Instruct"
MODEL_DIR="/home/nvidia/work_dev/wqq/nfs/test_cc_dev/Qwen3-VL-2B-Instruct"
PROJECT_DIR="/home/nvidia/work_dev/wqq/nfs/test_cc_dev/TensorRT-Edge-LLM"
OUTPUT_DIR="$PROJECT_DIR/output"
LLM_OUTPUT_DIR="$OUTPUT_DIR/llm"
VISUAL_OUTPUT_DIR="$OUTPUT_DIR/visual"
ENGINE_DIR="$PROJECT_DIR/engines"
LLM_ENGINE_DIR="$ENGINE_DIR/llm"
VISUAL_ENGINE_DIR="$ENGINE_DIR/visual"
TEST_DIR="$PROJECT_DIR/tests"
LOG_DIR="$PROJECT_DIR/logs"

# 创建必要的目录
mkdir -p "$OUTPUT_DIR" "$LLM_OUTPUT_DIR" "$VISUAL_OUTPUT_DIR" "$ENGINE_DIR" "$LLM_ENGINE_DIR" "$VISUAL_ENGINE_DIR" "$LOG_DIR"

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

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查目录是否存在
dir_exists() {
    if [ -d "$1" ]; then
        return 0
    else
        return 1
    fi
}

# 阶段1: 环境检查
stage1_env_check() {
    log_info "阶段1: 检查环境"

    # 检查项目目录是否存在
    if ! dir_exists "$PROJECT_DIR"; then
        log_error "项目目录不存在: $PROJECT_DIR"
        return 1
    fi

    # 检查模型目录是否存在
    if ! dir_exists "$MODEL_DIR"; then
        log_error "模型目录不存在: $MODEL_DIR"
        return 1
    fi

    # 检查Python
    if ! command_exists "python3"; then
        log_error "Python3未找到"
        return 1
    fi

    # 检查CUDA
    if ! command_exists "nvcc"; then
        log_warning "nvcc未找到，可能CUDA未正确安装"
    fi

    # 检查CMake
    if ! command_exists "cmake"; then
        log_error "CMake未找到"
        return 1
    fi

    log_success "环境检查通过"
    return 0
}

# 阶段2: 安装依赖
stage2_install_dependencies() {
    log_info "阶段2: 安装依赖"

    # 进入项目目录
    cd "$PROJECT_DIR" || return 1

    # 安装Python依赖
    if [ -f "requirements.txt" ]; then
        log_info "安装Python依赖..."
        pip3 install -r requirements.txt 2>&1
        if [ $? -ne 0 ]; then
            log_error "Python依赖安装失败"
            return 1
        fi
    fi

    # 检查是否需要更新子模块
    git submodule status 2>&1
    if [ $? -ne 0 ]; then
        log_info "更新git子模块..."
        git submodule update --init --recursive 2>&1
        if [ $? -ne 0 ]; then
            log_error "子模块更新失败"
            return 1
        fi
    fi

    # 设置Python路径
    if [ -z "$PYTHONPATH" ]; then
        export PYTHONPATH="$PROJECT_DIR"
    else
        export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    fi

    log_success "依赖安装完成"
    return 0
}

# 阶段3: 导出模型为ONNX格式
stage3_export_model() {
    log_info "阶段3: 导出模型为ONNX格式"

    # 导出LLM模型
    log_info "导出LLM模型..."
    python3 "$PROJECT_DIR/tensorrt_edgellm/scripts/export_llm.py" \
        --model_dir "$MODEL_DIR" \
        --output_dir "$LLM_OUTPUT_DIR" \
        --device cuda 2>&1

    if [ $? -ne 0 ]; then
        log_error "LLM模型导出失败"
        return 1
    fi

    # 导出视觉模型
    log_info "导出视觉模型..."
    python3 "$PROJECT_DIR/tensorrt_edgellm/scripts/export_visual.py" \
        --model_dir "$MODEL_DIR" \
        --output_dir "$VISUAL_OUTPUT_DIR" \
        --dtype fp16 \
        --device cuda 2>&1

    if [ $? -ne 0 ]; then
        log_error "视觉模型导出失败"
        return 1
    fi

    # 检查导出是否成功
    if ! dir_exists "$LLM_OUTPUT_DIR" || ! dir_exists "$VISUAL_OUTPUT_DIR"; then
        log_error "导出的目录不存在"
        return 1
    fi

    log_success "模型导出成功"
    log_info "LLM导出目录: $LLM_OUTPUT_DIR"
    log_info "视觉模型导出目录: $VISUAL_OUTPUT_DIR"
    return 0
}

# 阶段4: 构建TensorRT引擎
stage4_build_tensorrt_engine() {
    log_info "阶段4: 构建TensorRT引擎"

    # 构建项目
    log_info "构建项目..."
    cd "$PROJECT_DIR" || return 1

    if [ ! -d "build" ]; then
        mkdir -p build && cd build
        cmake .. -DTRT_EDGELLM_BUILD_EXAMPLES=ON -DTRT_EDGELLM_BUILD_TESTS=ON
        if [ $? -ne 0 ]; then
            log_error "CMake配置失败"
            return 1
        fi
    fi

    cd "$PROJECT_DIR/build" || return 1
    make -j$(nproc)
    if [ $? -ne 0 ]; then
        log_error "编译失败"
        return 1
    fi

    log_success "项目构建成功"

    # 构建LLM引擎
    log_info "构建LLM引擎..."
    cd "$PROJECT_DIR/build" || return 1
    ./examples/llm/llm_build \
        --onnxDir "$LLM_OUTPUT_DIR" \
        --engineDir "$LLM_ENGINE_DIR" \
        --maxInputLen 1024 \
        --maxKVCacheCapacity 4096 \
        --maxBatchSize 1 \
        --debug 2>&1

    if [ $? -ne 0 ]; then
        log_error "LLM引擎构建失败"
        return 1
    fi

    # 构建视觉模型引擎
    log_info "构建视觉模型引擎..."
    ./examples/multimodal/visual_build \
        --onnxDir "$VISUAL_OUTPUT_DIR" \
        --engineDir "$VISUAL_ENGINE_DIR" \
        --minImageTokens 4 \
        --maxImageTokens 1024 \
        --maxImageTokensPerImage 512 \
        --debug 2>&1

    if [ $? -ne 0 ]; then
        log_error "视觉模型引擎构建失败"
        return 1
    fi

    log_success "TensorRT引擎构建成功"
    log_info "LLM引擎目录: $LLM_ENGINE_DIR"
    log_info "视觉引擎目录: $VISUAL_ENGINE_DIR"
    return 0
}

# 阶段5: 运行推理测试
stage5_run_inference_test() {
    log_info "阶段5: 运行推理测试"

    # 检查引擎是否存在
    if ! dir_exists "$LLM_ENGINE_DIR" || ! dir_exists "$VISUAL_ENGINE_DIR"; then
        log_error "引擎目录不存在"
        return 1
    fi

    # 准备输入文件
    INPUT_FILE="$LOG_DIR/test_input.json"
    cat > "$INPUT_FILE" << 'EOF'
{
    "input": [
        {
            "text": "这张图片中有什么?",
            "image": ["$PROJECT_DIR/examples/multimodal/pics/giant_panda.jpeg"]
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

    # 替换项目路径
    sed -i "s|\$PROJECT_DIR|$PROJECT_DIR|g" "$INPUT_FILE"

    # 运行推理
    log_info "运行视觉语言推理测试..."
    cd "$PROJECT_DIR/build" || return 1

    ./examples/llm/llm_inference \
        --inputFile "$INPUT_FILE" \
        --engineDir "$LLM_ENGINE_DIR" \
        --visualEngineDir "$VISUAL_ENGINE_DIR" \
        --outputFile "$LOG_DIR/test_output.json" \
        --debug 2>&1

    if [ $? -ne 0 ]; then
        log_error "推理失败"
        return 1
    fi

    log_success "推理成功"

    # 显示输出
    if [ -f "$LOG_DIR/test_output.json" ]; then
        log_info "推理输出:"
        cat "$LOG_DIR/test_output.json"
    fi

    return 0
}

# 阶段6: 运行测试用例
stage6_run_tests() {
    log_info "阶段6: 运行项目测试用例"

    cd "$PROJECT_DIR/build" || return 1

    # 运行单元测试
    if [ -d "unittests" ] && [ -f "unittests/unit_tests" ]; then
        log_info "运行单元测试..."
        ./unittests/unit_tests 2>&1
        if [ $? -ne 0 ]; then
            log_warning "单元测试失败"
        else
            log_success "单元测试通过"
        fi
    fi

    # 运行集成测试
    if [ -d "tests" ] && [ -f "tests/integration_tests" ]; then
        log_info "运行集成测试..."
        ./tests/integration_tests 2>&1
        if [ $? -ne 0 ]; then
            log_warning "集成测试失败"
        else
            log_success "集成测试通过"
        fi
    fi

    return 0
}

# 阶段7: 清理临时文件
stage7_cleanup() {
    log_info "阶段7: 清理临时文件"

    # 可选清理
    log_info "是否需要清理临时文件? (y/N)"
    read -t 5 -p "" answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        log_info "清理临时文件..."

        # 删除导出的ONNX文件
        if [ -d "$OUTPUT_DIR" ]; then
            rm -rf "$OUTPUT_DIR"
        fi

        # 删除构建的引擎
        if [ -d "$ENGINE_DIR" ]; then
            rm -rf "$ENGINE_DIR"
        fi

        # 删除测试输出
        if [ -f "$LOG_DIR/test_input.json" ]; then
            rm "$LOG_DIR/test_input.json"
        fi

        if [ -f "$LOG_DIR/test_output.json" ]; then
            rm "$LOG_DIR/test_output.json"
        fi

        log_success "临时文件清理完成"
    else
        log_info "跳过清理临时文件"
    fi

    return 0
}

# 主函数
main() {
    log_info "=============================================="
    log_info "Qwen3-VL-2B-Instruct 部署运行测试自动化流程"
    log_info "目标设备: Jetson AGX Orin 64GB"
    log_info "模型: $MODEL_NAME"
    log_info "项目路径: $PROJECT_DIR"
    log_info "模型路径: $MODEL_DIR"
    log_info "=============================================="

    # 运行各个阶段
    if ! stage1_env_check; then
        exit 1
    fi

    if ! stage2_install_dependencies; then
        exit 1
    fi

    if ! stage3_export_model; then
        exit 1
    fi

    if ! stage4_build_tensorrt_engine; then
        exit 1
    fi

    if ! stage5_run_inference_test; then
        exit 1
    fi

    if ! stage6_run_tests; then
        log_warning "测试阶段完成，但有测试未通过"
    fi

    stage7_cleanup

    log_info "=============================================="
    log_success "Qwen3-VL-2B-Instruct 部署运行测试完成！"
    log_info "=============================================="

    return 0
}

# 处理信号
trap 'log_info "接收到中断信号，正在终止..."; exit 1' INT TERM

# 执行主函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
