#!/bin/bash
set -euo pipefail

# System metrics collection script
# TensorRT Edge-LLM Performance Test Suite

ACTION="$1"
TEST_CASE="$2"
TEST_ID="$3"

OUTPUT_RAW_DIR="$TEST_OUTPUT_DIR/raw_data/$TEST_ID"
METRICS_FILE="$OUTPUT_RAW_DIR/${TEST_CASE}_system_metrics.csv"
PID_FILE="/tmp/metrics_collector_${TEST_CASE}_${TEST_ID}.pid"
COLLECTION_INTERVAL_MS=$(yq e '.metrics.collection_interval' "$TEST_CONFIG_FILE" | sed 's/ms//')
COLLECTION_INTERVAL=$(echo "scale=3; $COLLECTION_INTERVAL_MS / 1000" | bc)

mkdir -p "$OUTPUT_RAW_DIR"

# Start metrics collection
start_collection() {
    # Check if metrics collection is disabled
    if [[ ${DISABLE_TEGRASTATS:-0} -eq 1 && ! -x "$(command -v nvidia-smi)" ]]; then
        echo "WARNING: Both tegrastats and nvidia-smi are not available, skipping metrics collection"
        return 1
    fi

    echo "Starting metrics collection for $TEST_CASE"
    echo "timestamp,gpu_utilization,gpu_memory_used_mb,gpu_power_w,gpu_temp_c,cpu_utilization,cpu_memory_used_mb,cpu_load_1min" > "$METRICS_FILE"

    # Run collection in background
    (
        while true; do
            timestamp=$(date +%s.%N)
            gpu_util=""
            gpu_mem=""
            gpu_power=""
            gpu_temp=""

            # Get GPU metrics
            if [[ ${DISABLE_TEGRASTATS:-0} -eq 0 && -x "$(command -v tegrastats)" ]]; then
                tegrastats_output=$(tegrastats --interval 100 --stop 1 2>/dev/null || true)
                gpu_util=$(echo "$tegrastats_output" | grep -oP 'GR3D_FREQ \K[0-9]+%' | head -1 | sed 's/%//' || echo "")
                gpu_mem=$(echo "$tegrastats_output" | grep -oP 'RAM \K[0-9]+/[0-9]+' | cut -d'/' -f1 || echo "")
                gpu_power=$(echo "$tegrastats_output" | grep -oP 'POM_5V_GPU \K[0-9]+/[0-9]+' | cut -d'/' -f1 || echo "")
                gpu_temp=$(echo "$tegrastats_output" | grep -oP 'GPU@\K[0-9.]+C' | sed 's/C//' || echo "")
            elif [[ -x "$(command -v nvidia-smi)" ]]; then
                # Fallback to nvidia-smi if tegrastats not available or disabled
                gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "")
                gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "")
                gpu_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/ .*//' || echo "")
                gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "")
            fi

            # Get CPU metrics
            cpu_util=$(top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\([0-9.]*\)%* id.*/\1/' | awk '{print 100 - $1}' 2>/dev/null || echo "")
            cpu_mem=$(free -m | grep Mem | awk '{print $3}' 2>/dev/null || echo "")
            cpu_load=$(uptime | awk -F'load average:' '{print $2}' | cut -d',' -f1 | sed 's/ //g' 2>/dev/null || echo "")

            # Write to CSV, handle empty values
            echo "$timestamp,${gpu_util:-},${gpu_mem:-},${gpu_power:-},${gpu_temp:-},${cpu_util:-},${cpu_mem:-},${cpu_load:-}" >> "$METRICS_FILE"

            sleep "$COLLECTION_INTERVAL"
        done
    ) &

    COLLECTOR_PID=$!
    echo $COLLECTOR_PID > "$PID_FILE"
    echo "Metrics collector started with PID $COLLECTOR_PID"
    return $COLLECTOR_PID
}

# Stop metrics collection
stop_collection() {
    if [[ -f "$PID_FILE" ]]; then
        COLLECTOR_PID=$(cat "$PID_FILE")
        if kill -0 "$COLLECTOR_PID" 2>/dev/null; then
            kill "$COLLECTOR_PID"
            wait "$COLLECTOR_PID" 2>/dev/null || true
            echo "Metrics collector stopped (PID $COLLECTOR_PID)"
        fi
        rm -f "$PID_FILE"
    else
        echo "No metrics collector running for $TEST_CASE"
    fi
}

# Main execution
case $ACTION in
    start)
        start_collection
        exit 0
        ;;
    stop)
        stop_collection
        exit 0
        ;;
    *)
        echo "Usage: $0 {start|stop} <test_case> <test_id>"
        exit 1
        ;;
esac
