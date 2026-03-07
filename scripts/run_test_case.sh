#!/bin/bash
set -uo pipefail

# Test case execution script
# TensorRT Edge-LLM Performance Test Suite
# 增加容错处理，适应不同环境

TEST_CASE="$1"
RUN_NUMBER="${2:-1}"

# 使用环境变量中的推理二进制路径，或者查找默认位置
INFERENCE_BIN="${INFERENCE_BIN_PATH:-./build/bin/llm_inference}"
if [[ ! -f "$INFERENCE_BIN" ]]; then
    if [[ -f "./build/examples/llm/llm_inference" ]]; then
        INFERENCE_BIN="./build/examples/llm/llm_inference"
    else
        echo "ERROR: Inference binary not found"
        exit 1
    fi
fi

# 获取配置，如果yq不可用则使用默认值
if [[ ${YQ_AVAILABLE:-1} -eq 1 && -n "${TEST_CONFIG_FILE:-}" ]]; then
    ENGINE_PATH=$(yq e '.model.tensorrt_engine_path' "$TEST_CONFIG_FILE" 2>/dev/null || echo "./engines/")
    PRECISION=$(yq e '.model.precision' "$TEST_CONFIG_FILE" 2>/dev/null || echo "FP16")
    MAX_BATCH_SIZE=$(yq e '.model.max_batch_size' "$TEST_CONFIG_FILE" 2>/dev/null || echo "1")
else
    ENGINE_PATH="./engines/"
    PRECISION="FP16"
    MAX_BATCH_SIZE="1"
fi

OUTPUT_RAW_DIR="$TEST_OUTPUT_DIR/raw_data/$TEST_ID"
mkdir -p "$OUTPUT_RAW_DIR" "$TEST_OUTPUT_DIR/logs"

RESULT_FILE="$OUTPUT_RAW_DIR/${TEST_CASE}_run_${RUN_NUMBER}.json"
LOG_FILE="$TEST_OUTPUT_DIR/logs/${TEST_ID}_${TEST_CASE}_run_${RUN_NUMBER}.log"

echo "Executing test case: $TEST_CASE (run $RUN_NUMBER)"

# Common inference parameters
INFERENCE_PARAMS=(
    --engineDir "$ENGINE_PATH"
    --precision "$PRECISION"
    --max_batch_size "$MAX_BATCH_SIZE"
)

# Function to run inference and capture metrics
run_inference() {
    local input_file="$1"
    local output_file="$2"

    # Run inference with timing
    local start_time=$(date +%s.%N)

    # 运行推理，检查是否有--multimodalEngineDir参数
    if [[ -d "${ENGINE_PATH%/}/../visual" || -d "./engines/visual" ]]; then
        # 多模态模型
        "$INFERENCE_BIN" "${INFERENCE_PARAMS[@]}" \
            --multimodalEngineDir "./engines/visual" \
            --inputFile "$input_file" \
            --outputFile "$output_file" \
            --dumpOutput \
            2>> "$LOG_FILE" >/dev/null
    else
        # 纯LLM模型
        "$INFERENCE_BIN" "${INFERENCE_PARAMS[@]}" \
            --inputFile "$input_file" \
            --outputFile "$output_file" \
            --dumpOutput \
            2>> "$LOG_FILE" >/dev/null
    fi

    local exit_code=$?
    local end_time=$(date +%s.%N)

    # Calculate latency
    local total_latency="0"
    if command -v bc &> /dev/null; then
        total_latency=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
    fi

    # Calculate token count if jq available
    local output_tokens="0"
    local first_token_latency="0"
    local tokens_per_second="0"

    if [[ ${JQ_AVAILABLE:-1} -eq 1 && -f "$output_file" ]]; then
        output_text=$(jq -r '.responses[0].output_text' "$output_file" 2>/dev/null || echo "")
        if [[ -n "$output_text" ]]; then
            # 粗略估算token数，按每个token对应4个字符
            output_tokens=$(echo -n "$output_text" | wc -c | awk '{print int($1/4)}')
        fi
        if [[ $output_tokens -gt 0 && $total_latency > 0 ]]; then
            tokens_per_second=$(echo "scale=2; $output_tokens / $total_latency" | bc -l 2>/dev/null || echo "0")
        fi
    fi

    # 返回结果
    jq -n \
        --argjson total_latency "$total_latency" \
        --argjson output_tokens "$output_tokens" \
        --argjson first_token_latency "$first_token_latency" \
        --argjson tokens_per_second "$tokens_per_second" \
        --argjson exit_code "$exit_code" \
        '{
            "metrics": {
                "total_latency": $total_latency,
                "output_tokens": $output_tokens,
                "first_token_latency": $first_token_latency,
                "tokens_per_second": $tokens_per_second
            },
            "exit_code": $exit_code
        }' 2>/dev/null || echo '{"metrics": {}, "exit_code": 0}'
}

# 根据测试用例类型选择输入文件
INPUT_FILE=""
case $TEST_CASE in
    "single_request"|"warmup")
        INPUT_FILE="test_cases/single_request.json"
        ;;
    "multi_modal")
        INPUT_FILE="test_cases/multi_modal.json"
        ;;
    "concurrent_requests")
        INPUT_FILE="test_cases/concurrent_requests.json"
        ;;
    *)
        # 默认使用单请求测试用例
        INPUT_FILE="test_cases/single_request.json"
        ;;
esac

# 确保输入文件存在
if [[ ! -f "$INPUT_FILE" ]]; then
    # 查找其他位置的测试用例
    if [[ -f "./tests/test_cases/$(basename $INPUT_FILE)" ]]; then
        INPUT_FILE="./tests/test_cases/$(basename $INPUT_FILE)"
    else
        echo "ERROR: Test case file $INPUT_FILE not found"
        exit 1
    fi
fi

# 创建临时输出文件
TEMP_OUTPUT=$(mktemp)

# 运行推理
echo "Running inference with input file: $INPUT_FILE"
result=$(run_inference "$INPUT_FILE" "$TEMP_OUTPUT")

# 保存结果
echo "$result" > "$RESULT_FILE"

# 清理临时文件
rm -f "$TEMP_OUTPUT"

echo "Test case $TEST_CASE run $RUN_NUMBER completed"
echo "Results saved to: $RESULT_FILE"

exit 0
