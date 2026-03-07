#!/bin/bash
set -euo pipefail

# Test case execution script
# TensorRT Edge-LLM Performance Test Suite

TEST_CASE="$1"
RUN_NUMBER="${2:-1}"

INFERENCE_BIN="./build/bin/llm_inference"
ENGINE_PATH=$(yq e '.model.tensorrt_engine_path' "$TEST_CONFIG_FILE")
OUTPUT_RAW_DIR="$TEST_OUTPUT_DIR/raw_data/$TEST_ID"
mkdir -p "$OUTPUT_RAW_DIR"

RESULT_FILE="$OUTPUT_RAW_DIR/${TEST_CASE}_run_${RUN_NUMBER}.json"
LOG_FILE="$TEST_OUTPUT_DIR/logs/$TEST_ID_${TEST_CASE}_run_${RUN_NUMBER}.log"

echo "Executing test case: $TEST_CASE (run $RUN_NUMBER)"

# Common inference parameters
INFERENCE_PARAMS=(
    --engine "$ENGINE_PATH"
    --precision "$(yq e '.model.precision' "$TEST_CONFIG_FILE")"
    --max_batch_size "$(yq e '.model.max_batch_size' "$TEST_CONFIG_FILE")"
)

# Function to run inference and capture metrics
run_inference() {
    local prompt="$1"
    local params="$2"
    local output_file="$3"

    # Extract inference parameters
    local temp=$(echo "$params" | jq -r '.temperature')
    local top_p=$(echo "$params" | jq -r '.top_p')
    local max_new_tokens=$(echo "$params" | jq -r '.max_new_tokens')

    # Run inference with timing
    local start_time=$(date +%s.%N)
    local response=$("$INFERENCE_BIN" "${INFERENCE_PARAMS[@]}" \
        --prompt "$prompt" \
        --temperature "$temp" \
        --top_p "$top_p" \
        --max_new_tokens "$max_new_tokens" \
        --json_output 2>> "$LOG_FILE")
    local end_time=$(date +%s.%N)

    # Calculate latency
    local total_latency=$(echo "$end_time - $start_time" | bc -l)

    # Extract metrics from response
    local prefill_latency=$(echo "$response" | jq -r '.prefill_latency // 0')
    local generation_latency=$(echo "$response" | jq -r '.generation_latency // 0')
    local first_token_latency=$(echo "$response" | jq -r '.first_token_latency // 0')
    local input_tokens=$(echo "$response" | jq -r '.input_tokens // 0')
    local output_tokens=$(echo "$response" | jq -r '.output_tokens // 0')
    local tokens_per_second=$(echo "$response" | jq -r '.tokens_per_second // 0')

    # Generate result JSON
    jq -n \
        --arg test_case "$TEST_CASE" \
        --arg run_number "$RUN_NUMBER" \
        --arg timestamp "$(date -Iseconds)" \
        --argjson total_latency "$total_latency" \
        --argjson prefill_latency "$prefill_latency" \
        --argjson generation_latency "$generation_latency" \
        --argjson first_token_latency "$first_token_latency" \
        --argjson input_tokens "$input_tokens" \
        --argjson output_tokens "$output_tokens" \
        --argjson tokens_per_second "$tokens_per_second" \
        --arg prompt "$prompt" \
        '{
            test_case: $test_case,
            run_number: $run_number,
            timestamp: $timestamp,
            metrics: {
                total_latency: $total_latency,
                prefill_latency: $prefill_latency,
                generation_latency: $generation_latency,
                first_token_latency: $first_token_latency,
                input_tokens: $input_tokens,
                output_tokens: $output_tokens,
                tokens_per_second: $tokens_per_second
            },
            request: {
                prompt: $prompt,
                temperature: $temp,
                top_p: $top_p,
                max_new_tokens: $max_new_tokens
            }
        }' > "$output_file"
}

# Execute test case based on type
case $TEST_CASE in
    warmup)
        # Simple warmup inference
        warmup_prompt="Hello, this is a warmup request."
        run_inference "$warmup_prompt" '{"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 32}' /dev/null
        ;;

    single_request)
        # Read test case input
        request=$(jq -r '.requests[0]' test_cases/single_request.json)
        prompt=$(echo "$request" | jq -r '.text')
        params=$(echo "$request" | jq -r '.parameters')
        run_inference "$prompt" "$params" "$RESULT_FILE"
        ;;

    concurrent_requests)
        # Concurrent request test handled by orchestration
        echo "Concurrent test executed by main controller" > "$RESULT_FILE"
        ;;

    multi_modal)
        # Multi-modal test case
        test_case=$(jq -r '.test_cases[0]' test_cases/multi_modal.json)
        prompt=$(echo "$test_case" | jq -r '.text')
        image=$(echo "$test_case" | jq -r '.image')
        params=$(echo "$test_case" | jq -r '.parameters')

        # Add image parameter to inference
        INFERENCE_PARAMS+=(--image "$image")
        run_inference "$prompt" "$params" "$RESULT_FILE"
        ;;

    cold_start)
        # Unload engine before run
        if [[ "$(yq e '.test_cases.cold_start.unload_engine_between_runs' "$TEST_CONFIG_FILE")" == "true" ]]; then
            # Drop caches to simulate cold start
            sudo sync && sudo echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
        fi

        request=$(jq -r '.requests[0]' test_cases/single_request.json)
        prompt=$(echo "$request" | jq -r '.text')
        params=$(echo "$request" | jq -r '.parameters')
        run_inference "$prompt" "$params" "$RESULT_FILE"
        ;;

    *)
        # Generic test case execution
        echo "Generic test case execution for $TEST_CASE" > "$RESULT_FILE"
        ;;
esac

echo "Test case $TEST_CASE run $RUN_NUMBER completed successfully"
exit 0
