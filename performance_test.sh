#!/bin/bash
set -uo pipefail
# 移除set -e，让脚本遇到错误时继续执行而不是直接退出

# Performance Test Script for Qwen3-VL-2B-Instruct on Jetson AGX Orin
# TensorRT Edge-LLM Performance Test Suite

# Default configuration
CONFIG_FILE="config/test_config.yaml"
WARMUP_RUNS=5
TEST_RUNS=10
OUTPUT_DIR="results"
TEST_SUITE="all"
VERBOSE=false

# Print help message
print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Performance test suite for TensorRT Edge-LLM"
    echo
    echo "Options:"
    echo "  -c, --config FILE       Path to test configuration file (default: config/test_config.yaml)"
    echo "  -w, --warmup NUM        Number of warmup runs (default: 5)"
    echo "  -r, --runs NUM          Number of test runs per test case (default: 10)"
    echo "  -o, --output-dir DIR    Output directory for results (default: results)"
    echo "  -t, --test-suite NAME   Test suite to run: all, basic, edge, model, engineering (default: all)"
    echo "  -v, --verbose           Enable verbose output"
    echo "  -h, --help              Print this help message"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        -r|--runs)
            TEST_RUNS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--test-suite)
            TEST_SUITE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Export configuration variables
export TEST_CONFIG_FILE="$CONFIG_FILE"
export TEST_WARMUP_RUNS="$WARMUP_RUNS"
export TEST_RUNS="$TEST_RUNS"
export TEST_OUTPUT_DIR="$OUTPUT_DIR"
export TEST_SUITE="$TEST_SUITE"
export TEST_VERBOSE="$VERBOSE"

# Create output directories
mkdir -p "$OUTPUT_DIR/raw_data" "$OUTPUT_DIR/reports" "$OUTPUT_DIR/logs"

# Generate unique test ID
TEST_ID=$(date +%Y%m%d_%H%M%S)
export TEST_ID
echo "Starting performance test run: $TEST_ID"
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Warmup runs: $WARMUP_RUNS"
echo "  Test runs: $TEST_RUNS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Test suite: $TEST_SUITE"
echo

# Step 1: Setup environment and check dependencies
echo "=== Step 1: Environment Setup ==="
bash scripts/setup_env.sh
ENV_EXIT_CODE=$?
if [[ $ENV_EXIT_CODE -ne 0 ]]; then
    echo "WARNING: Environment setup encountered some issues, but will attempt to continue"
    echo "Some features may be disabled or limited"
else
    echo "Environment setup completed successfully"
fi
echo

# Step 2: Run warmup if enabled
if [[ $WARMUP_RUNS -gt 0 ]]; then
    echo "=== Step 2: Warmup Runs ==="
    export TEST_MODE="warmup"
    for ((i=1; i<=WARMUP_RUNS; i++)); do
        echo "Warmup run $i/$WARMUP_RUNS..."
        if ! bash scripts/run_test_case.sh warmup; then
            echo "WARNING: Warmup run $i failed, continuing..."
        fi
    done
    unset TEST_MODE
    echo "Warmup completed"
    echo
fi

# Step 3: Execute test cases
echo "=== Step 3: Test Execution ==="
TEST_CASES=()

# Select test cases based on test suite
case $TEST_SUITE in
    all)
        TEST_CASES=(
            "single_request"
            "concurrent_requests"
            "multi_modal"
            "cold_start"
            "stability"
            "power_efficiency"
            "input_length_variation"
            "output_length_variation"
            "image_resolution_variation"
            "parameter_sensitivity"
        )
        ;;
    basic)
        TEST_CASES=(
            "single_request"
            "concurrent_requests"
            "multi_modal"
        )
        ;;
    edge)
        TEST_CASES=(
            "cold_start"
            "stability"
            "power_efficiency"
        )
        ;;
    model)
        TEST_CASES=(
            "input_length_variation"
            "output_length_variation"
            "image_resolution_variation"
        )
        ;;
    engineering)
        TEST_CASES=(
            "parameter_sensitivity"
        )
        ;;
    *)
        echo "ERROR: Unknown test suite: $TEST_SUITE"
        exit 1
        ;;
esac

# Execute each test case
for TEST_CASE in "${TEST_CASES[@]}"; do
    echo "Running test case: $TEST_CASE"
    TEST_CASE_LOG="$OUTPUT_DIR/logs/$TEST_ID_$TEST_CASE.log"

    # Start metrics collection in background
    METRICS_PID=""
    if bash scripts/collect_metrics.sh start "$TEST_CASE" "$TEST_ID"; then
        METRICS_PID=$!
    fi

    # Run test case multiple times
    for ((run=1; run<=TEST_RUNS; run++)); do
        echo "  Run $run/$TEST_RUNS..."
        if ! bash scripts/run_test_case.sh "$TEST_CASE" "$run" >> "$TEST_CASE_LOG" 2>&1; then
            echo "  WARNING: Run $run failed for test case $TEST_CASE"
        fi
    done

    # Stop metrics collection
    if [[ -n $METRICS_PID ]]; then
        bash scripts/collect_metrics.sh stop "$TEST_CASE" "$TEST_ID"
        wait $METRICS_PID 2>/dev/null || true
    fi

    echo "Test case $TEST_CASE completed"
    echo
done

# Step 4: Generate performance report
echo "=== Step 4: Report Generation ==="
bash scripts/generate_report.sh "$TEST_ID"
REPORT_EXIT_CODE=$?
if [[ $REPORT_EXIT_CODE -ne 0 ]]; then
    echo "WARNING: Report generation encountered some issues"
    echo "Raw test data is still available in: $OUTPUT_DIR/raw_data/$TEST_ID/"
else
    echo "Report generated successfully"
fi

echo
echo "======================================================================"
echo "✅ Performance test execution completed!"
echo "Test ID: $TEST_ID"
echo "Raw test data: $OUTPUT_DIR/raw_data/$TEST_ID/"
if [[ -d "$OUTPUT_DIR/reports/$TEST_ID/" ]]; then
    echo "Report available at: $OUTPUT_DIR/reports/$TEST_ID/"
fi
echo "======================================================================"
echo
exit 0
