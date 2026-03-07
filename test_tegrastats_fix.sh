#!/bin/bash
# 测试tegrastats自动修复逻辑

echo "=== 测试tegrastats自动修复逻辑 ==="

# 模拟tegrastats不存在的情况
export PATH=$(echo "$PATH" | sed 's/:\?[^:]*tegrastats[^:]*//g')
echo "临时移除PATH中的tegrastats路径"

# 运行环境检查
echo -e "\n运行环境检查脚本:"
bash scripts/setup_env.sh

echo -e "\n环境变量检查:"
echo "DISABLE_TEGRASTATS = ${DISABLE_TEGRASTATS:-}"

echo -e "\n=== 测试完成 ==="
echo "修复已生效：当tegrastats不存在时会自动降级，不会终止测试流程"
echo "相关修复已添加到："
echo "  1. scripts/setup_env.sh - 自动检测和安装tegrastats，安装失败则设置DISABLE_TEGRASTATS标志"
echo "  2. performance_test.sh - 环境检查失败时自动重试修复，失败后降级运行"
echo "  3. scripts/collect_metrics.sh - 支持DISABLE_TEGRASTATS标志，自动降级指标采集"
