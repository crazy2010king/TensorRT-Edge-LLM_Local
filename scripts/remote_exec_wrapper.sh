#!/bin/bash
# /env_exec远程执行封装脚本，支持自动重试和异常处理
set -e

MAX_RETRIES=3
RETRY_DELAY=5
COMMAND=""
DEVICE="agx_orin"

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --retry-delay)
            RETRY_DELAY="$2"
            shift 2
            ;;
        *)
            COMMAND="$*"
            break
            ;;
    esac
done

if [[ -z "$COMMAND" ]]; then
    echo "错误: 请指定要执行的命令"
    exit 1
fi

# 选择远程执行命令
case "$DEVICE" in
    agx_orin*)
        REMOTE_CMD="env_exec"
        ;;
    orin_nx*)
        REMOTE_CMD="env_exec5"
        ;;
    drive_thor)
        REMOTE_CMD="env_exec_drive"
        ;;
    *)
        echo "错误: 不支持的设备类型: $DEVICE"
        exit 1
        ;;
esac

# 执行命令，支持重试
retry_count=0
while [[ $retry_count -lt $MAX_RETRIES ]]; do
    echo "执行远程命令 (尝试 $((retry_count+1))/$MAX_RETRIES): $COMMAND"
    if $REMOTE_CMD "$COMMAND"; then
        echo "命令执行成功"
        exit 0
    fi

    retry_count=$((retry_count+1))
    if [[ $retry_count -lt $MAX_RETRIES ]]; then
        echo "命令执行失败，$RETRY_DELAY 秒后重试..."
        sleep $RETRY_DELAY
    fi
done

echo "错误: 命令执行失败，已达到最大重试次数 $MAX_RETRIES"
exit 1
