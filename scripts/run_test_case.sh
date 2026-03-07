#!/bin/bash
set -uo pipefail

# Test case execution script
# TensorRT Edge-LLM Performance Test Suite
# 修复推理参数和测试用例路径问题

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
    ENGINE_PATH=$(yq e '.model.tensorrt_engine_path' "$TEST_CONFIG_FILE" 2>/dev/null || echo "./engines/llm")
else
    ENGINE_PATH="./engines/llm"
fi

OUTPUT_RAW_DIR="$TEST_OUTPUT_DIR/raw_data/$TEST_ID"
mkdir -p "$OUTPUT_RAW_DIR" "$TEST_OUTPUT_DIR/logs"

RESULT_FILE="$OUTPUT_RAW_DIR/${TEST_CASE}_run_${RUN_NUMBER}.json"
LOG_FILE="$TEST_OUTPUT_DIR/logs/${TEST_ID}_${TEST_CASE}_run_${RUN_NUMBER}.log"

echo "Executing test case: $TEST_CASE (run $RUN_NUMBER)"

# Common inference parameters - 移除不存在的参数
INFERENCE_PARAMS=(
    --engineDir "$ENGINE_PATH"
)

# Function to run inference and capture metrics
run_inference() {
    local input_file="$1"
    local output_file="$2"

    # Run inference with timing
    local start_time=$(date +%s.%N)

    # 运行推理，检查是否有多模态引擎
    local exit_code=0
    if [[ -d "./engines/visual" ]]; then
        # 多模态模型，使用正确的参数
        "$INFERENCE_BIN" "${INFERENCE_PARAMS[@]}" \
            --multimodalEngineDir "./engines/visual" \
            --inputFile "$input_file" \
            --outputFile "$output_file" \
            --dumpOutput \
            2>> "$LOG_FILE" >/dev/null || exit_code=$?
    else
        # 纯LLM模型
        "$INFERENCE_BIN" "${INFERENCE_PARAMS[@]}" \
            --inputFile "$input_file" \
            --outputFile "$output_file" \
            --dumpOutput \
            2>> "$LOG_FILE" >/dev/null || exit_code=$?
    fi

    local end_time=$(date +%s.%N)

    # Calculate latency - 使用python计算，避免依赖bc
    local total_latency=$(python3 -c "print($end_time - $start_time)" 2>/dev/null || echo "0")

    # Calculate token count if jq available and output file exists
    local output_tokens="0"
    local first_token_latency="0"
    local tokens_per_second="0"

    if [[ ${JQ_AVAILABLE:-1} -eq 1 && -f "$output_file" ]]; then
        output_text=$(jq -r '.responses[0].output_text' "$output_file" 2>/dev/null || echo "")
        if [[ -n "$output_text" && "$output_text" != "null" ]]; then
            # 粗略估算token数，按每个token对应4个字符
            output_tokens=$(echo -n "$output_text" | wc -c | awk '{print int($1/4)}')
        fi
        if [[ $output_tokens -gt 0 && $(python3 -c "print(1 if $total_latency > 0 else 0)") -eq 1 ]]; then
            tokens_per_second=$(python3 -c "print(round($output_tokens / $total_latency, 2))" 2>/dev/null || echo "0")
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
        }' 2>/dev/null || echo '{"metrics": {"total_latency": 0, "output_tokens": 0, "first_token_latency": 0, "tokens_per_second": 0}, "exit_code": 1}'
}

# 根据测试用例类型选择输入文件，优先使用tests/test_cases下的文件
INPUT_FILE=""
case $TEST_CASE in
    "single_request"|"warmup")
        if [[ -f "./tests/test_cases/vlm_basic.json" ]]; then
            INPUT_FILE="./tests/test_cases/vlm_basic.json"
        elif [[ -f "./test_cases/single_request.json" ]]; then
            INPUT_FILE="./test_cases/single_request.json"
        else
            # 创建一个简单的测试用例
            cat > /tmp/test_input.json << EOF
[
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片"},
                    {"type": "image", "image": "examples/multimodal/pics/giant_panda.jpeg"}
                ]
            }
        ],
        "max_new_tokens": 100
    }
]
EOF
            INPUT_FILE="/tmp/test_input.json"
        fi
        ;;
    "multi_modal")
        if [[ -f "./tests/test_cases/vlm_basic.json" ]]; then
            INPUT_FILE="./tests/test_cases/vlm_basic.json"
        elif [[ -f "./test_cases/multi_modal.json" ]]; then
            INPUT_FILE="./test_cases/multi_modal.json"
        else
            INPUT_FILE="/tmp/test_input.json"
        fi
        ;;
    "concurrent_requests")
        if [[ -f "./tests/test_cases/vlm_basic.json" ]]; then
            INPUT_FILE="./tests/test_cases/vlm_basic.json"
        elif [[ -f "./test_cases/concurrent_requests.json" ]]; then
            INPUT_FILE="./test_cases/concurrent_requests.json"
        else
            INPUT_FILE="/tmp/test_input.json"
        fi
        ;;
    *)
        # 默认使用多模态测试用例
        if [[ -f "./tests/test_cases/vlm_basic.json" ]]; then
            INPUT_FILE="./tests/test_cases/vlm_basic.json"
        else
            INPUT_FILE="/tmp/test_input.json"
        fi
        ;;
esac

# 确保输入文件存在
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Test case file $INPUT_FILE not found"
    exit 1
fi

# 创建临时输出文件
TEMP_OUTPUT=$(mktemp)

# 运行推理
echo "Running inference with input file: $INPUT_FILE"
result=$(run_inference "$INPUT_FILE" "$TEMP_OUTPUT")

# 保存结果
echo "$result" > "$RESULT_FILE"

# 清理临时文件
rm -f "$TEMP_OUTPUT" "/tmp/test_input.json"

echo "Test case $TEST_CASE run $RUN_NUMBER completed"
echo "Results saved to: $RESULT_FILE"

# 输出日志最后几行便于调试
if [[ -f "$LOG_FILE" ]]; then
    echo "Last 10 lines of log:"
    tail -10 "$LOG_FILE"
fi

exit 0
