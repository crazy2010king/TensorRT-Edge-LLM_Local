#!/usr/bin/env python3
"""
生成最终全流程测试报告
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="生成最终全流程测试报告")
    parser.add_argument("--x86-result", required=True, help="x86优化结果JSON路径")
    parser.add_argument("--deploy-result", help="部署测试结果JSON路径")
    parser.add_argument("--output", required=True, help="最终报告输出路径")

    args = parser.parse_args()

    # 加载x86优化结果
    with open(args.x86_result, "r") as f:
        x86_data = json.load(f)

    # 加载部署测试结果
    deploy_data = {}
    if args.deploy_result and Path(args.deploy_result).exists():
        with open(args.deploy_result, "r") as f:
            deploy_data = json.load(f)

    # 构建最终报告
    final_report = {
        "report_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "generated_at": datetime.now().isoformat(),
        "x86_optimization": x86_data,
        "edge_deployment": deploy_data,
        "status": "success" if x86_data.get("status") == "success" and (not deploy_data or deploy_data.get("status") == "success") else "failed",
        "performance_summary": None
    }

    # 提取性能摘要
    if deploy_data and "performance_report" in deploy_data and deploy_data["performance_report"]:
        perf = deploy_data["performance_report"]
        final_report["performance_summary"] = {
            "avg_latency_ms": perf.get("avg_latency_ms"),
            "throughput_tps": perf.get("throughput_tps"),
            "memory_usage_mb": perf.get("memory_usage_mb"),
            "accuracy_loss": perf.get("accuracy_loss"),
            "power_consumption_w": perf.get("power_consumption_w")
        }

    # 保存报告
    with open(args.output, "w") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"最终报告已生成: {args.output}")

if __name__ == "__main__":
    main()
