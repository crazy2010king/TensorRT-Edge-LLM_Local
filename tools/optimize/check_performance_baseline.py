#!/usr/bin/env python3
"""
性能基线检查工具：对比测试结果与基线，识别性能回归
"""
import argparse
import json
import os
import sys
from pathlib import Path
import yaml

BASELINE_DIR = Path(__file__).parent.parent.parent / "config" / "baseline_performance"

def load_baseline_config(device, model=None, quantization=None):
    """加载指定设备的基线配置"""
    baseline_file = BASELINE_DIR / f"{device}.yaml"
    if not baseline_file.exists():
        print(f"[警告] 未找到设备 {device} 的基线配置文件: {baseline_file}")
        return None

    with open(baseline_file, "r") as f:
        baseline = yaml.safe_load(f)

    # 如果指定了模型和量化方案，返回对应配置
    if model and quantization:
        model_key = model.split("/")[-1].lower()
        if model_key in baseline.get("models", {}) and quantization in baseline["models"][model_key]:
            return baseline["models"][model_key][quantization]
        else:
            print(f"[警告] 未找到模型 {model} 量化方案 {quantization} 的基线配置")
            return None

    return baseline

def compare_performance(report_data, baseline_config, fail_on_violation=False):
    """对比测试结果与基线"""
    if not baseline_config:
        print("[信息] 无基线配置，跳过对比")
        return True

    passed = True
    results = {
        "passed": True,
        "metrics": {}
    }

    # 检查吞吐量
    if "throughput_tps" in report_data and "min_throughput_tps" in baseline_config:
        actual = report_data["throughput_tps"]
        expected = baseline_config["min_throughput_tps"]
        status = "✅" if actual >= expected else "❌"
        passed = passed and (actual >= expected)
        results["metrics"]["throughput_tps"] = {
            "actual": actual,
            "expected": expected,
            "status": "pass" if actual >= expected else "fail"
        }
        print(f"{status} 吞吐量: {actual:.2f} token/s (基线: ≥ {expected} token/s)")

    # 检查延迟
    if "avg_latency_ms" in report_data and "max_latency_ms" in baseline_config:
        actual = report_data["avg_latency_ms"]
        expected = baseline_config["max_latency_ms"]
        status = "✅" if actual <= expected else "❌"
        passed = passed and (actual <= expected)
        results["metrics"]["avg_latency_ms"] = {
            "actual": actual,
            "expected": expected,
            "status": "pass" if actual <= expected else "fail"
        }
        print(f"{status} 平均延迟: {actual:.2f} ms (基线: ≤ {expected} ms)")

    # 检查内存占用
    if "memory_usage_mb" in report_data and "max_memory_mb" in baseline_config:
        actual = report_data["memory_usage_mb"]
        expected = baseline_config["max_memory_mb"]
        status = "✅" if actual <= expected else "❌"
        passed = passed and (actual <= expected)
        results["metrics"]["memory_usage_mb"] = {
            "actual": actual,
            "expected": expected,
            "status": "pass" if actual <= expected else "fail"
        }
        print(f"{status} 内存占用: {actual:.2f} MB (基线: ≤ {expected} MB)")

    # 检查精度损失
    if "accuracy_loss" in report_data and "max_accuracy_loss" in baseline_config:
        actual = report_data["accuracy_loss"]
        expected = baseline_config["max_accuracy_loss"]
        status = "✅" if actual <= expected else "❌"
        passed = passed and (actual <= expected)
        results["metrics"]["accuracy_loss"] = {
            "actual": actual,
            "expected": expected,
            "status": "pass" if actual <= expected else "fail"
        }
        print(f"{status} 精度损失: {actual:.2f}% (基线: ≤ {expected}%)")

    results["passed"] = passed

    if passed:
        print("\n✅ 所有性能指标符合基线要求")
    else:
        print("\n❌ 部分性能指标未达到基线要求")
        if fail_on_violation:
            sys.exit(1)

    return results

def main():
    parser = argparse.ArgumentParser(description="性能基线检查工具")
    parser.add_argument("--report", required=True, help="性能报告JSON路径")
    parser.add_argument("--device", required=True, help="设备类型")
    parser.add_argument("--model", help="模型ID/路径")
    parser.add_argument("--quantization", help="量化方案")
    parser.add_argument("--fail-on-violation", action="store_true", help="未达基线时返回错误")
    parser.add_argument("--output", help="对比结果输出路径")

    args = parser.parse_args()

    # 加载测试报告
    with open(args.report, "r") as f:
        report_data = json.load(f)

    # 加载基线配置
    baseline_config = load_baseline_config(args.device, args.model, args.quantization)

    # 执行对比
    results = compare_performance(report_data, baseline_config, args.fail_on_violation)

    # 输出结果
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"对比结果已保存到: {args.output}")

if __name__ == "__main__":
    main()
