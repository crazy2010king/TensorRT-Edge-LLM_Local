#!/bin/bash
# 本地模型转换脚本
# 用于在本地计算环境中转换 Qwen3-VL-2B-Instruct 模型为 ONNX 格式

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
    # 这里可以添加需要清理的临时文件或目录
}

# 信号处理
trap 'log_info "接收到中断信号，正在终止..."; cleanup; exit 1' INT TERM

# 默认配置
PROJECT_DIR="/mnt/test_cc_dev/TensorRT-Edge-LLM"
MODEL_DIR="/mnt/test_cc_dev/Qwen3-VL-2B-Instruct"
OUTPUT_DIR="$PROJECT_DIR/output"
LLM_OUTPUT_DIR="$OUTPUT_DIR/llm"
VISUAL_OUTPUT_DIR="$OUTPUT_DIR/visual"
DEVICE="cuda"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-dir)
            PROJECT_DIR="$2"
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
        --device)
            DEVICE="$2"
            shift; shift
            ;;
        --help)
            echo "使用说明: $0 [选项]"
            echo "  --project-dir <路径>    指定项目目录 (默认: $PROJECT_DIR)"
            echo "  --model-dir <路径>      指定模型目录 (默认: $MODEL_DIR)"
            echo "  --output-dir <路径>     指定输出目录 (默认: $OUTPUT_DIR)"
            echo "  --device <设备>         指定计算设备 (默认: $DEVICE)"
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

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."

    # 检查 Python
    if ! command_exists "python3"; then
        log_error "Python3 未找到"
        return 1
    fi

    # 检查 pip
    if ! command_exists "pip3"; then
        log_error "pip3 未找到"
        return 1
    fi

    # 检查必要的 Python 库
    log_info "检查 Python 依赖..."
    required_packages=("torch" "transformers" "onnx")

    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            log_warning "$package 未找到，正在安装依赖..."

            if ! pip3 install -r "$PROJECT_DIR/requirements.txt"; then
                log_error "依赖安装失败"
                return 1
            fi

            # 只需要安装一次
            break
        fi
    done

    log_success "依赖检查完成"
    return 0
}

# 检查环境
check_environment() {
    log_info "检查运行环境..."

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

    # 检查必要的模型文件
    required_model_files=("config.json" "tokenizer.json" "model.safetensors" "vocab.json")
    for file in "${required_model_files[@]}"; do
        if ! file_exists "$MODEL_DIR/$file"; then
            log_warning "模型文件缺失: $file"
        fi
    done

    # 检查设备可用性
    if [ "$DEVICE" = "cuda" ] || [[ "$DEVICE" =~ cuda:[0-9]+ ]]; then
        log_info "检查 CUDA 设备..."
        if ! command_exists "nvidia-smi"; then
            log_warning "nvidia-smi 未找到，CUDA 设备可能不可用"
        fi
    fi

    log_success "环境检查完成"
    return 0
}

# 创建输出目录
create_output_dirs() {
    log_info "创建输出目录..."

    if mkdir -p "$LLM_OUTPUT_DIR" "$VISUAL_OUTPUT_DIR"; then
        log_success "输出目录创建成功"
    else
        log_error "输出目录创建失败"
        return 1
    fi

    return 0
}

# 转换语言模型
convert_llm_model() {
    log_info "开始转换语言模型..."

    python3 "$PROJECT_DIR/tensorrt_edgellm/scripts/export_llm.py" \
        --model_dir "$MODEL_DIR" \
        --output_dir "$LLM_OUTPUT_DIR" \
        --device "$DEVICE"

    if [ $? -ne 0 ]; then
        log_error "语言模型转换失败"
        return 1
    fi

    if file_exists "$LLM_OUTPUT_DIR/model.onnx"; then
        log_success "语言模型转换成功"
    else
        log_error "语言模型输出文件不存在"
        return 1
    fi

    return 0
}

# 转换视觉模型
convert_visual_model() {
    log_info "开始转换视觉模型..."

    python3 "$PROJECT_DIR/tensorrt_edgellm/scripts/export_visual.py" \
        --model_dir "$MODEL_DIR" \
        --output_dir "$VISUAL_OUTPUT_DIR" \
        --dtype fp16 \
        --device "$DEVICE"

    if [ $? -ne 0 ]; then
        log_error "视觉模型转换失败"
        return 1
    fi

    if file_exists "$VISUAL_OUTPUT_DIR/model.onnx"; then
        log_success "视觉模型转换成功"
    else
        log_error "视觉模型输出文件不存在"
        return 1
    fi

    return 0
}

# 主函数
main() {
    log_info "=============================================="
    log_info "Qwen3-VL-2B-Instruct 本地模型转换脚本"
    log_info "项目目录: $PROJECT_DIR"
    log_info "模型目录: $MODEL_DIR"
    log_info "输出目录: $OUTPUT_DIR"
    log_info "计算设备: $DEVICE"
    log_info "=============================================="

    # 检查依赖
    if ! check_dependencies; then
        cleanup
        exit 1
    fi

    # 检查环境
    if ! check_environment; then
        cleanup
        exit 1
    fi

    # 创建输出目录
    if ! create_output_dirs; then
        cleanup
        exit 1
    fi

    # 设置 PYTHONPATH
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

    # 转换语言模型
    if ! convert_llm_model; then
        cleanup
        exit 1
    fi

    # 转换视觉模型
    if ! convert_visual_model; then
        cleanup
        exit 1
    fi

    log_info "=============================================="
    log_success "模型转换完成"
    log_info "语言模型输出目录: $LLM_OUTPUT_DIR"
    log_info "视觉模型输出目录: $VISUAL_OUTPUT_DIR"
    log_info "=============================================="

    cleanup
    return 0
}

# 执行主函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
