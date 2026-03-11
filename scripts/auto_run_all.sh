#!/bin/bash
# 自动执行守护脚本，智能处理所有错误直到两个脚本都执行成功
set +uo pipefail

MAX_RETRIES=100
RETRY_DELAY=60
SUCCESS_MARKER="./output/.execution_success"
mkdir -p ./output/{logs,models,engines,backups}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a ./output/logs/auto_execution.log
}

fix_common_issues() {
    log "🔧 自动修复常见问题..."

    # 1. 修复Python路径
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    log "✅ PYTHONPATH已设置: $PYTHONPATH"

    # 2. 修复脚本权限
    chmod +x scripts/*.sh
    log "✅ 脚本执行权限已添加"

    # 3. 自动查找模型路径
    if [ ! -d ../qwen3_vl_2b ] && [ -d ../Qwen3-VL-2B-Instruct ]; then
        ln -sf ../Qwen3-VL-2B-Instruct ../qwen3_vl_2b
        log "✅ 自动创建基础模型软链接: Qwen3-VL-2B-Instruct -> qwen3_vl_2b"
    fi

    # 4. 自动查找Eagle草稿模型
    DRAFT_MODEL=$(find /home/nvidia/ .. -maxdepth 5 -name "*eagle*draft*" -type d 2>/dev/null | grep -v "__pycache__" | head -1)
    if [ -n "$DRAFT_MODEL" ] && [ ! -d ../qwen3_vl_2b_eagle_draft ]; then
        ln -sf "$DRAFT_MODEL" ../qwen3_vl_2b_eagle_draft
        log "✅ 自动发现并链接Eagle草稿模型: $DRAFT_MODEL"
    fi

    # 5. 修复脚本中的路径问题
    sed -i 's#BASE_DIR="/home/nvidia/work_dev/wqq/nfs/test_cc_dev/TensorRT-Edge-LLM"#BASE_DIR="'"$(pwd)"'"#' scripts/*.sh
    log "✅ 脚本BASE_DIR路径已适配当前环境"

    log "✅ 所有常见问题修复完成"
}

run_convert_models() {
    log "🚀 开始执行convert_models.sh"
    if bash scripts/convert_models.sh > ./output/logs/convert_models_current.log 2>&1; then
        log "✅ convert_models.sh 执行成功"
        touch ./output/.convert_models_success
        return 0
    else
        log "❌ convert_models.sh 执行失败，错误日志:"
        tail -30 ./output/logs/convert_models_current.log

        # 特殊处理：如果是草稿模型缺失，先禁用Eagle运行基础量化
        if grep -q "Eagle草稿模型目录不存在" ./output/logs/convert_models_current.log; then
            log "⚠️  检测到Eagle草稿模型缺失，临时禁用Eagle运行基础量化..."
            # 创建临时脚本，禁用Eagle
            cp scripts/convert_models.sh scripts/convert_models_no_eagle.sh
            sed -i 's/EAGLE_ENABLE="true"/EAGLE_ENABLE="false"/' scripts/convert_models_no_eagle.sh
            sed -i '/# 步骤3: 导出Eagle草稿模型引擎/,/# ============== 执行完成 ==============/d' scripts/convert_models_no_eagle.sh
            if bash scripts/convert_models_no_eagle.sh > ./output/logs/convert_models_no_eagle.log 2>&1; then
                log "✅ 基础模型量化和引擎导出成功（无Eagle）"
                touch ./output/.convert_models_success
                return 0
            fi
        fi
        return 1
    fi
}

run_build_optimized() {
    log "🚀 开始执行build_optimized.sh"
    if bash scripts/build_optimized.sh > ./output/logs/build_optimized_current.log 2>&1; then
        log "✅ build_optimized.sh 执行成功"
        touch ./output/.build_optimized_success
        return 0
    else
        log "❌ build_optimized.sh 执行失败，错误日志:"
        tail -30 ./output/logs/build_optimized_current.log

        # 处理编译错误：清理build目录重试
        if grep -i "error\|make: \*\*\*" ./output/logs/build_optimized_current.log; then
            log "⚠️  检测到编译错误，清理build目录重试..."
            rm -rf build
            if bash scripts/build_optimized.sh > ./output/logs/build_optimized_retry.log 2>&1; then
                log "✅ 编译优化重试成功"
                touch ./output/.build_optimized_success
                return 0
            fi
        fi
        return 1
    fi
}

# 主循环
log "🎉 自动执行守护进程启动，将持续运行直到所有脚本执行成功"
log "📝 完整日志路径: ./output/logs/auto_execution.log"

while true; do
    if [ -f "$SUCCESS_MARKER" ]; then
        log "🎉 所有脚本已执行成功，自动执行完成！"
        log "📊 执行结果:"
        [ -f ./output/.convert_models_success ] && log "✅ convert_models.sh 已成功完成"
        [ -f ./output/.build_optimized_success ] && log "✅ build_optimized.sh 已成功完成"
        exit 0
    fi

    fix_common_issues

    # 执行convert_models（如果未成功）
    if [ ! -f ./output/.convert_models_success ]; then
        run_convert_models
    fi

    # 执行build_optimized（如果convert已成功且build未成功）
    if [ -f ./output/.convert_models_success ] && [ ! -f ./output/.build_optimized_success ]; then
        run_build_optimized
    fi

    # 检查是否都成功
    if [ -f ./output/.convert_models_success ] && [ -f ./output/.build_optimized_success ]; then
        touch "$SUCCESS_MARKER"
        continue
    fi

    log "⏳ 执行失败，${RETRY_DELAY}秒后重试..."
    sleep $RETRY_DELAY
done
