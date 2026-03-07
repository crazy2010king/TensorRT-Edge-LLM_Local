#!/bin/bash
set -e

# 远程设备验证脚本
# 自动在远端AGX设备上运行性能测试并生成运行记录

REMOTE_HOST="172.16.119.188"
REMOTE_USER="nvidia"
REMOTE_PASS="nvidia"
PROJECT_PATH="/home/nvidia/work_dev/wqq/nfs/test_cc_dev/TensorRT-Edge-LLM"
RUN_RECORD="性能测试运行记录_$(date +%Y%m%d_%H%M%S).md"

echo "=== 远端AGX设备性能测试验证 ===" | tee "$RUN_RECORD"
echo "测试时间: $(date)" | tee -a "$RUN_RECORD"
echo "远端设备: $REMOTE_HOST" | tee -a "$RUN_RECORD"
echo "项目路径: $PROJECT_PATH" | tee -a "$RUN_RECORD"
echo "" | tee -a "$RUN_RECORD"

# 1. 环境检查
echo "=== 步骤1: 远程环境检查 ===" | tee -a "$RUN_RECORD"
sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" << EOF | tee -a "$RUN_RECORD"
echo "设备信息: \$(cat /proc/device-tree/model | tr -d '\0')"
echo "CUDA版本: \$(nvcc --version | grep release | awk '{print \$6}' | cut -d',' -f1)"
echo "Docker容器状态: \$(docker inspect -f '{{.State.Status}}' ros2_nitr)"
echo "项目目录存在: \$(if [ -d "$PROJECT_PATH" ]; then echo "是"; else echo "否"; fi)"
EOF
echo "" | tee -a "$RUN_RECORD"

# 2. 性能测试脚本检查
echo "=== 步骤2: 测试脚本检查 ===" | tee -a "$RUN_RECORD"
sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "docker exec ros2_nitr bash -c 'cd $PROJECT_PATH && ls -la performance_test.sh config/ scripts/ test_cases/'" | tee -a "$RUN_RECORD"
echo "" | tee -a "$RUN_RECORD"

# 3. 环境依赖检查
echo "=== 步骤3: 依赖检查 ===" | tee -a "$RUN_RECORD"
sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "docker exec ros2_nitr bash -c 'cd $PROJECT_PATH && bash scripts/setup_env.sh'" | tee -a "$RUN_RECORD"
echo "" | tee -a "$RUN_RECORD"

# 4. 执行简短性能测试
echo "=== 步骤4: 执行性能测试（快速模式） ===" | tee -a "$RUN_RECORD"
echo "运行测试套件: basic，预热2次，运行3次" | tee -a "$RUN_RECORD"
TEST_START=$(date +%s)
sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "docker exec ros2_nitr bash -c 'cd $PROJECT_PATH && ./performance_test.sh --warmup 2 --runs 3 --test-suite basic'" | tee -a "$RUN_RECORD"
TEST_END=$(date +%s)
TEST_DURATION=$((TEST_END - TEST_START))
echo "测试总耗时: $TEST_DURATION 秒" | tee -a "$RUN_RECORD"
echo "" | tee -a "$RUN_RECORD"

# 5. 检查结果
echo "=== 步骤5: 测试结果验证 ===" | tee -a "$RUN_RECORD"
sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "docker exec ros2_nitr bash -c 'cd $PROJECT_PATH && ls -la results/reports/'" | tee -a "$RUN_RECORD"

# 获取最新报告
LATEST_REPORT=$(sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "docker exec ros2_nitr bash -c 'cd $PROJECT_PATH && ls -td results/reports/* | head -1'")
if [ -n "$LATEST_REPORT" ]; then
    echo "" | tee -a "$RUN_RECORD"
    echo "=== 测试报告摘要 ===" | tee -a "$RUN_RECORD"
    sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "docker exec ros2_nitr bash -c 'cd $PROJECT_PATH && head -50 $LATEST_REPORT/performance_report.md'" | tee -a "$RUN_RECORD"
fi

echo "" | tee -a "$RUN_RECORD"
echo "=== 测试完成 ===" | tee -a "$RUN_RECORD"
echo "运行记录已保存到: $RUN_RECORD"
echo "完整报告路径: $REMOTE_HOST:$PROJECT_PATH/$LATEST_REPORT/"
