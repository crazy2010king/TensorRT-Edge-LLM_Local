#!/bin/bash
set -uo pipefail

# Performance report generation script
# TensorRT Edge-LLM Performance Test Suite
# 增加容错处理，即使jq/yq缺失也能生成基础报告

TEST_ID="$1"

RAW_DATA_DIR="$TEST_OUTPUT_DIR/raw_data/$TEST_ID"
REPORT_DIR="$TEST_OUTPUT_DIR/reports/$TEST_ID"
mkdir -p "$REPORT_DIR"

REPORT_MD="$REPORT_DIR/performance_report.md"
REPORT_JSON="$REPORT_DIR/performance_results.json"

echo "Generating performance report for test run: $TEST_ID"

# 检查jq是否可用
if [[ ${JQ_AVAILABLE:-1} -eq 0 || ! -x "$(command -v jq)" ]]; then
    echo "WARNING: jq not available, generating simplified report"

    # 生成简化版Markdown报告
    cat > "$REPORT_MD" << EOF
# TensorRT Edge-LLM Performance Test Report
## Test ID: $TEST_ID
## Date: $(date)
## Device: $(if [[ ${YQ_AVAILABLE:-1} -eq 1 && -n "${TEST_CONFIG_FILE:-}" ]]; then yq e '.device.name' "$TEST_CONFIG_FILE" 2>/dev/null; else echo "Jetson AGX Orin (default)"; fi)

## Summary
Test completed successfully. Full detailed report is not available due to missing jq dependency.

## Raw Test Results
Raw data files are available in: $RAW_DATA_DIR
EOF

    echo "Simplified report generated at: $REPORT_MD"
    exit 0
fi

# Function to calculate statistics for a metric
calculate_stats() {
    local data_file="$1"
    local metric="$2"

    # Extract all values
    values=$(jq -r ".metrics.$metric" "$data_file"/*_run_*.json 2>/dev/null | grep -v null | sort -g)

    if [[ -z "$values" ]]; then
        echo "{}"
        return
    fi

    # Calculate statistics
    count=$(echo "$values" | wc -l)
    min=$(echo "$values" | head -1)
    max=$(echo "$values" | tail -1)
    mean=$(echo "$values" | awk '{ sum += $1 } END { if (NR > 0) print sum / NR }')
    median=$(echo "$values" | awk '{ a[i++]=$1 } END { print a[int(i/2)] }')
    p90=$(echo "$values" | awk -v p=90 '{ a[i++]=$1 } END { print a[int(i*p/100)] }')
    p95=$(echo "$values" | awk -v p=95 '{ a[i++]=$1 } END { print a[int(i*p/100)] }')
    p99=$(echo "$values" | awk -v p=99 '{ a[i++]=$1 } END { print a[int(i*p/100)] }')
    std_dev=$(echo "$values" | awk -v mean="$mean" '{ sum += ($1 - mean)^2 } END { if (NR > 1) print sqrt(sum / (NR - 1)) }')

    jq -n \
        --argjson count "$count" \
        --argjson min "$min" \
        --argjson max "$max" \
        --argjson mean "$mean" \
        --argjson median "$median" \
        --argjson p50 "$median" \
        --argjson p90 "$p90" \
        --argjson p95 "$p95" \
        --argjson p99 "$p99" \
        --argjson std_dev "$std_dev" \
        '{
            count: $count,
            min: $min,
            max: $max,
            mean: $mean,
            median: $median,
            p50: $p50,
            p90: $p90,
            p95: $p95,
            p99: $p99,
            std_dev: $std_dev
        }' 2>/dev/null || echo "{}"
}

# Process all test cases
ALL_RESULTS="{}"
TEST_CASES=$(ls "$RAW_DATA_DIR"/*_run_*.json 2>/dev/null | xargs basename 2>/dev/null | cut -d'_' -f1 | sort | uniq || true)

for TEST_CASE in $TEST_CASES; do
    echo "Processing test case: $TEST_CASE"

    # Calculate metrics for this test case
    latency_stats=$(calculate_stats "$RAW_DATA_DIR" "total_latency")
    tps_stats=$(calculate_stats "$RAW_DATA_DIR" "tokens_per_second")
    first_token_stats=$(calculate_stats "$RAW_DATA_DIR" "first_token_latency")

    # Add to results
    ALL_RESULTS=$(echo "$ALL_RESULTS" | jq \
        --arg test_case "$TEST_CASE" \
        --argjson latency "$latency_stats" \
        --argjson tps "$tps_stats" \
        --argjson first_token "$first_token_stats" \
        '. + {($test_case): {
            latency: $latency,
            tokens_per_second: $tps,
            first_token_latency: $first_token
        }}' 2>/dev/null || echo "$ALL_RESULTS")
done

# Save JSON results
echo "$ALL_RESULTS" | jq '.' > "$REPORT_JSON" 2>/dev/null || echo "{}" > "$REPORT_JSON"

# Generate Markdown report
DEVICE_NAME="Jetson AGX Orin (default)"
if [[ ${YQ_AVAILABLE:-1} -eq 1 && -n "${TEST_CONFIG_FILE:-}" ]]; then
    DEVICE_NAME=$(yq e '.device.name' "$TEST_CONFIG_FILE" 2>/dev/null || echo "Jetson AGX Orin")
fi

cat > "$REPORT_MD" << EOF
# TensorRT Edge-LLM Performance Test Report
## Test ID: $TEST_ID
## Date: $(date)
## Device: $DEVICE_NAME

## Summary
| Test Case | Avg Latency (s) | P95 Latency (s) | Avg Token Rate (tokens/s) | P95 First Token (s) |
|-----------|----------------|-----------------|---------------------------|---------------------|
EOF

# Add test case results to table
for TEST_CASE in $TEST_CASES; do
    avg_latency=$(echo "$ALL_RESULTS" | jq -r ".$TEST_CASE.latency.mean // \"N/A\"" 2>/dev/null || echo "N/A")
    p95_latency=$(echo "$ALL_RESULTS" | jq -r ".$TEST_CASE.latency.p95 // \"N/A\"" 2>/dev/null || echo "N/A")
    avg_tps=$(echo "$ALL_RESULTS" | jq -r ".$TEST_CASE.tokens_per_second.mean // \"N/A\"" 2>/dev/null || echo "N/A")
    p95_ft=$(echo "$ALL_RESULTS" | jq -r ".$TEST_CASE.first_token_latency.p95 // \"N/A\"" 2>/dev/null || echo "N/A")

    echo "| $TEST_CASE | $avg_latency | $p95_latency | $avg_tps | $p95_ft |" >> "$REPORT_MD"
done

# Add detailed metrics sections
cat >> "$REPORT_MD" << EOF

## Detailed Metrics

EOF

for TEST_CASE in $TEST_CASES; do
    echo -e "\n### $TEST_CASE" >> "$REPORT_MD"
    echo -e "\n#### Latency Statistics" >> "$REPORT_MD"
    echo "| Metric | Value |" >> "$REPORT_MD"
    echo "|--------|-------|" >> "$REPORT_MD"

    for metric in count min max mean median p50 p90 p95 p99 std_dev; do
        value=$(echo "$ALL_RESULTS" | jq -r ".$TEST_CASE.latency.$metric // \"N/A\"" 2>/dev/null || echo "N/A")
        echo "| $metric | $value |" >> "$REPORT_MD"
    done

    echo -e "\n#### Token Generation Rate Statistics" >> "$REPORT_MD"
    echo "| Metric | Value |" >> "$REPORT_MD"
    echo "|--------|-------|" >> "$REPORT_MD"

    for metric in count min max mean median p50 p90 p95 p99 std_dev; do
        value=$(echo "$ALL_RESULTS" | jq -r ".$TEST_CASE.tokens_per_second.$metric // \"N/A\"" 2>/dev/null || echo "N/A")
        echo "| $metric | $value |" >> "$REPORT_MD"
    done
done

# Add system metrics section
cat >> "$REPORT_MD" << EOF

## System Metrics
System metrics are available in CSV format in the raw data directory: $RAW_DATA_DIR

*GPU utilization, memory usage, power consumption, and temperature metrics are collected during test execution.*
EOF

echo "Performance report generated successfully!"
echo "Report location: $REPORT_MD"
echo "Raw data location: $RAW_DATA_DIR"

exit 0
