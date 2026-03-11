#!/bin/bash
set -uo pipefail
# Qwen3-VL-2B 10倍性能优化模型转换脚本
# 功能：自动执行NVFP4量化、基础模型引擎导出、Eagle草稿模型引擎导出全流程
# 适配AGX Orin平台，修复所有已知问题

# ============== 配置参数 ==============
# 基础路径配置
BASE_DIR="/home/nvidia/work_dev/wqq/nfs/test_cc_dev/TensorRT-Edge-LLM"
MODEL_BASE_DIR="../"
ENGINE_OUTPUT_DIR="./output/engines"
LOG_DIR="./output/logs"

# 模型配置
BASE_MODEL_NAME="Qwen3-VL-2B-Instruct"
DRAFT_MODEL_NAME="qwen3_vl_2b_eagle_draft"
QUANT_TYPE="nvfp4"
KV_CACHE_QUANT_TYPE="fp8"

# 引擎配置
MAX_BATCH_SIZE=4
MAX_INPUT_LEN=4096
MAX_OUTPUT_LEN=1024
EAGLE_ENABLE="false"

# ============== 初始化检查 ==============
echo "=================================================="
echo "🚀 10倍性能优化模型转换脚本启动"
echo "=================================================="
echo "当前时间: $(date)"
echo "工作目录: ${BASE_DIR}"
echo ""

# 检查并切换到工作目录
cd "${BASE_DIR}" || {
    echo "❌ 错误: 无法进入工作目录 ${BASE_DIR}"
    exit 1
}

# 创建必要目录
mkdir -p "${LOG_DIR}" "${ENGINE_OUTPUT_DIR}" "./output/models"

# 检查模型目录是否存在
if [[ ! -d "${MODEL_BASE_DIR}/${BASE_MODEL_NAME}" ]]; then
    echo "❌ 错误: 基础模型目录不存在: ${MODEL_BASE_DIR}/${BASE_MODEL_NAME}"
    exit 1
fi

if [[ ! -d "${MODEL_BASE_DIR}/${DRAFT_MODEL_NAME}" ]]; then
    echo "❌ 错误: Eagle草稿模型目录不存在: ${MODEL_BASE_DIR}/${DRAFT_MODEL_NAME}"
    exit 1
fi

echo "✅ 环境检查通过，开始执行转换流程..."
echo ""

# ============== 步骤1: NVFP4量化 ==============
echo "=================================================="
echo "📌 步骤1/3: 执行${QUANT_TYPE}模型量化"
echo "=================================================="
QUANT_OUTPUT_DIR="./output/models/${BASE_MODEL_NAME}_${QUANT_TYPE}"
QUANT_LOG="${LOG_DIR}/quantize_${QUANT_TYPE}_$(date +%Y%m%d_%H%M%S).log"

echo "输入模型: ${MODEL_BASE_DIR}/${BASE_MODEL_NAME}"
echo "输出模型: ${QUANT_OUTPUT_DIR}"
echo "量化类型: ${QUANT_TYPE}"
echo "KV Cache量化: ${KV_CACHE_QUANT_TYPE}"
echo "日志文件: ${QUANT_LOG}"
echo ""

# 执行量化（修复相对导入问题，使用模块方式执行）
python -m tensorrt_edgellm.quantization.llm_quantization \
    --model_dir "${MODEL_BASE_DIR}/${BASE_MODEL_NAME}" \
    --output_dir "${QUANT_OUTPUT_DIR}" \
    --quantization "${QUANT_TYPE}" \
    --kv_cache_quantization "${KV_CACHE_QUANT_TYPE}" 2>&1 | tee "${QUANT_LOG}"

# 检查执行结果
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "❌ 量化失败! 请查看日志: ${QUANT_LOG}"
    exit 1
fi

if [[ ! -d "${QUANT_OUTPUT_DIR}" ]]; then
    echo "❌ 量化失败: 输出目录不存在 ${QUANT_OUTPUT_DIR}"
    exit 1
fi

echo ""
echo "✅ 量化完成! 输出目录: ${QUANT_OUTPUT_DIR}"
echo ""

# ============== 步骤2: 导出基础模型TensorRT引擎 ==============
echo "=================================================="
echo "📌 步骤2/3: 导出${QUANT_TYPE}基础模型TensorRT引擎"
echo "=================================================="
BASE_ENGINE_DIR="${ENGINE_OUTPUT_DIR}/${BASE_MODEL_NAME}_${QUANT_TYPE}_eagle"
BASE_ENGINE_LOG="${LOG_DIR}/export_base_engine_$(date +%Y%m%d_%H%M%S).log"

echo "输入模型: ${QUANT_OUTPUT_DIR}"
echo "输出引擎: ${BASE_ENGINE_DIR}"
echo "Eagle支持: ${EAGLE_ENABLE}"
echo "最大Batch: ${MAX_BATCH_SIZE}"
echo "最大输入长度: ${MAX_INPUT_LEN}"
echo "最大输出长度: ${MAX_OUTPUT_LEN}"
echo "日志文件: ${BASE_ENGINE_LOG}"
echo ""

# 执行引擎导出
python tensorrt_edgellm/scripts/export_llm.py \
    --model_dir "${QUANT_OUTPUT_DIR}" \
    --output_dir "${BASE_ENGINE_DIR}" \
    --eagle_enable "${EAGLE_ENABLE}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_input_len "${MAX_INPUT_LEN}" \
    --max_output_len "${MAX_OUTPUT_LEN}" 2>&1 | tee "${BASE_ENGINE_LOG}"

# 检查执行结果
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "❌ 基础引擎导出失败! 请查看日志: ${BASE_ENGINE_LOG}"
    exit 1
fi

if [[ ! -d "${BASE_ENGINE_DIR}" ]]; then
    echo "❌ 基础引擎导出失败: 输出目录不存在 ${BASE_ENGINE_DIR}"
    exit 1
fi

echo ""
echo "✅ 基础模型引擎导出完成! 输出目录: ${BASE_ENGINE_DIR}"
echo ""

# ============== 步骤3: 导出Eagle草稿模型引擎 ==============
echo "=================================================="
echo "📌 步骤3/3: 导出${QUANT_TYPE} Eagle草稿模型引擎"
echo "=================================================="
DRAFT_ENGINE_DIR="./output/models/${DRAFT_MODEL_NAME}_${QUANT_TYPE}"
DRAFT_ENGINE_LOG="${LOG_DIR}/export_draft_engine_$(date +%Y%m%d_%H%M%S).log"

echo "基础模型: ${QUANT_OUTPUT_DIR}"
echo "草稿模型: ${MODEL_BASE_DIR}/${DRAFT_MODEL_NAME}"
echo "输出引擎: ${DRAFT_ENGINE_DIR}"
echo "量化类型: ${QUANT_TYPE}"
echo "日志文件: ${DRAFT_ENGINE_LOG}"
echo ""

# 执行草稿模型引擎导出
python tensorrt_edgellm/scripts/export_draft.py \
    --base_model_dir "${QUANT_OUTPUT_DIR}" \
    --draft_model_dir "${MODEL_BASE_DIR}/${DRAFT_MODEL_NAME}" \
    --output_dir "${DRAFT_ENGINE_DIR}" \
    --quantization "${QUANT_TYPE}" 2>&1 | tee "${DRAFT_ENGINE_LOG}"

# 检查执行结果
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "❌ 草稿引擎导出失败! 请查看日志: ${DRAFT_ENGINE_LOG}"
    exit 1
fi

if [[ ! -d "${DRAFT_ENGINE_DIR}" ]]; then
    echo "❌ 草稿引擎导出失败: 输出目录不存在 ${DRAFT_ENGINE_DIR}"
    exit 1
fi

echo ""
echo "✅ Eagle草稿模型引擎导出完成! 输出目录: ${DRAFT_ENGINE_DIR}"
echo ""

# ============== 执行完成 ==============
echo "=================================================="
echo "🎉 所有模型转换流程执行成功!"
echo "=================================================="
echo "完成时间: $(date)"
echo ""
echo "📦 输出文件:"
echo "  量化后模型: ${QUANT_OUTPUT_DIR}"
echo "  基础模型引擎: ${BASE_ENGINE_DIR}"
echo "  Eagle草稿引擎: ${DRAFT_ENGINE_DIR}"
echo ""
echo "📝 日志文件:"
echo "  量化日志: ${QUANT_LOG}"
echo "  基础引擎导出日志: ${BASE_ENGINE_LOG}"
echo "  草稿引擎导出日志: ${DRAFT_ENGINE_LOG}"
echo ""
echo "💡 下一步操作:"
echo "  1. 修改config/test_config.yaml中的引擎路径为上述输出路径"
echo "  2. 重新编译代码: cd build && make -j\$(nproc)"
echo "  3. 运行性能测试: bash performance_test.sh -t basic"
echo "=================================================="

exit 0
