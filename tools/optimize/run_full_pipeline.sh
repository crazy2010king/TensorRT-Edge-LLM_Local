#!/bin/bash
# 全流程开发测试闭环入口脚本
set -e

# 配置默认参数
MODEL=""
QUANTIZATION=""
ENABLE_EAGLE3=0
DEVICE="agx_orin_64g"
OUTPUT_DIR="./pipeline_results"
SKIP_UNIT_TESTS=0
SKIP_PERFORMANCE_TEST=0
BATCH_SIZE=1
MAX_SEQ_LEN=2048

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --enable-eagle3)
            ENABLE_EAGLE3=1
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-unit-tests)
            SKIP_UNIT_TESTS=1
            shift
            ;;
        --skip-performance-test)
            SKIP_PERFORMANCE_TEST=1
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-seq-len)
            MAX_SEQ_LEN="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "使用方法:"
            echo "  $0 --model <模型ID/路径> --quantization <量化方案> [选项]"
            echo "选项:"
            echo "  --enable-eagle3                启用Eagle3草稿模型优化"
            echo "  --device <设备类型>            目标设备 (默认: agx_orin_64g)"
            echo "  --output-dir <目录>            输出目录 (默认: ./pipeline_results)"
            echo "  --skip-unit-tests              跳过单元测试"
            echo "  --skip-performance-test        跳过性能测试"
            echo "  --batch-size <N>               推理批次大小 (默认: 1)"
            echo "  --max-seq-len <N>              最大序列长度 (默认: 2048)"
            exit 1
            ;;
    esac
done

# 检查必填参数
if [[ -z "$MODEL" || -z "$QUANTIZATION" ]]; then
    echo "错误: --model 和 --quantization 是必填参数"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
X86_OUTPUT_DIR="$OUTPUT_DIR/x86_optimize"
EDGE_OUTPUT_DIR="$OUTPUT_DIR/edge_deploy"
FINAL_REPORT="$OUTPUT_DIR/final_report.json"

echo "=================================================="
echo "TensorRT-Edge-LLM 全流程开发测试闭环"
echo "=================================================="
echo "模型: $MODEL"
echo "量化方案: $QUANTIZATION"
echo "启用Eagle3: $([ $ENABLE_EAGLE3 == 1 ] && echo "是" || echo "否")"
echo "目标设备: $DEVICE"
echo "输出目录: $OUTPUT_DIR"
echo "=================================================="

# 步骤1: 执行x86侧优化流水线
echo -e "\n[1/4] 执行x86侧优化流水线 (量化、导出、单元测试)"
X86_CMD="python3 tools/optimize/x86_optimize_pipeline.py \
    --model $MODEL \
    --quantization $QUANTIZATION \
    --output-dir $X86_OUTPUT_DIR"

if [ $ENABLE_EAGLE3 == 1 ]; then
    X86_CMD+=" --enable-eagle3"
fi

if [ $SKIP_UNIT_TESTS == 1 ]; then
    X86_CMD+=" --skip-unit-tests"
fi

eval $X86_CMD

# 提取ONNX路径
ONNX_PATH=$(jq -r '.onnx_path' "$X86_OUTPUT_DIR/optimize_result.json")
EAGLE3_ONNX_PATH=$(jq -r '.eagle3_onnx_path' "$X86_OUTPUT_DIR/optimize_result.json")

echo "x86优化完成，ONNX模型路径: $ONNX_PATH"
if [ $ENABLE_EAGLE3 == 1 ]; then
    echo "Eagle3模型路径: $EAGLE3_ONNX_PATH"
fi

# 步骤2: 执行端侧部署测试流水线
if [ $SKIP_PERFORMANCE_TEST == 0 ]; then
    echo -e "\n[2/4] 执行端侧部署测试流水线"
    EDGE_CMD="python3 tools/optimize/edge_deploy_pipeline.py \
        --onnx-path $ONNX_PATH \
        --device $DEVICE \
        --engine-output-path $EDGE_OUTPUT_DIR \
        --batch-size $BATCH_SIZE \
        --max-seq-len $MAX_SEQ_LEN"

    if [ $ENABLE_EAGLE3 == 1 ] && [ "$EAGLE3_ONNX_PATH" != "null" ]; then
        EDGE_CMD+=" --eagle3-onnx-path $EAGLE3_ONNX_PATH"
    fi

    eval $EDGE_CMD
fi

# 步骤3: 性能基线对比和质量门禁
echo -e "\n[3/4] 执行性能基线对比和质量门禁检查"
BASELINE_CHECK_CMD="python3 tools/optimize/check_performance_baseline.py \
    --model $MODEL \
    --quantization $QUANTIZATION \
    --device $DEVICE \
    --report $EDGE_OUTPUT_DIR/performance_report.json"

if [ $SKIP_PERFORMANCE_TEST == 0 ]; then
    eval $BASELINE_CHECK_CMD
fi

# 步骤4: 生成最终报告
echo -e "\n[4/4] 生成最终测试报告"
python3 tools/optimize/generate_final_report.py \
    --x86-result "$X86_OUTPUT_DIR/optimize_result.json" \
    --deploy-result "$EDGE_OUTPUT_DIR/deploy_result.json" \
    --output "$FINAL_REPORT"

echo "=================================================="
echo "全流程执行完成!"
echo "最终报告: $FINAL_REPORT"
echo "=================================================="

# 打印结果摘要
if [ $SKIP_PERFORMANCE_TEST == 0 ]; then
    echo -e "\n性能测试结果摘要:"
    jq '.performance_summary' "$FINAL_REPORT"
fi
