#!/usr/bin/env python3
"""
x86侧一站式优化工具：单命令完成量化、导出、单元测试全流程
"""
import argparse
import os
import sys
import subprocess
import json
from pathlib import Path

SUPPORTED_QUANTIZATION_SCHEMES = [
    "fp16", "fp8", "fp8_lepto", "smoothquant",
    "int4_awq", "int4_gptq", "2bit", "1.25bit"
]

def run_command(cmd, cwd=None, env=None):
    """执行命令并返回结果"""
    print(f"[执行命令] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"[错误] 命令执行失败: {cmd}")
        print(f"错误输出: {result.stderr}")
        sys.exit(1)
    print(f"[输出] {result.stdout}")
    return result.stdout

def main():
    parser = argparse.ArgumentParser(description="x86侧一站式量化导出工具")
    parser.add_argument("--model", required=True, help="模型路径或HuggingFace模型ID")
    parser.add_argument("--quantization", required=True, choices=SUPPORTED_QUANTIZATION_SCHEMES,
                        help=f"量化方案，支持: {', '.join(SUPPORTED_QUANTIZATION_SCHEMES)}")
    parser.add_argument("--enable-eagle3", action="store_true", help="是否启用Eagle3草稿模型优化")
    parser.add_argument("--output-dir", default="./optimize_output", help="输出目录")
    parser.add_argument("--skip-unit-tests", action="store_true", help="跳过单元测试")
    parser.add_argument("--eagle3-train-epochs", type=int, default=10, help="Eagle3训练轮数")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_output_path = output_dir / "model.onnx"
    eagle3_output_path = output_dir / "eagle3_model.onnx" if args.enable_eagle3 else None

    print(f"=== 开始x86优化流水线 ===")
    print(f"模型: {args.model}")
    print(f"量化方案: {args.quantization}")
    print(f"启用Eagle3: {args.enable_eagle3}")
    print(f"输出目录: {output_dir}")

    # 步骤1: 执行量化
    print("\n=== 步骤1: 执行量化 ===")
    quant_cmd = f"python3 quantize.py --model {args.model} --quant {args.quantization} --output {output_dir / 'quantized_model'}"
    run_command(quant_cmd, cwd="/mnt/test_cc_dev/git_source/TensorRT-Edge-LLM_Local")

    # 步骤2: 导出ONNX模型
    print("\n=== 步骤2: 导出ONNX模型 ===")
    export_cmd = f"python3 export_onnx.py --model {output_dir / 'quantized_model'} --output {onnx_output_path}"
    if args.quantization in ["int4_awq", "int4_gptq", "2bit", "1.25bit"]:
        export_cmd += " --enable-weight-only-quant"
    run_command(export_cmd, cwd="/mnt/test_cc_dev/git_source/TensorRT-Edge-LLM_Local")

    # 步骤3: Eagle3训练和导出（可选）
    if args.enable_eagle3:
        print("\n=== 步骤3: Eagle3草稿模型训练和导出 ===")
        eagle3_train_cmd = f"python3 train_eagle3.py --base-model {output_dir / 'quantized_model'} --epochs {args.eagle3_train_epochs} --output {output_dir / 'eagle3_checkpoint'}"
        run_command(eagle3_train_cmd, cwd="/mnt/test_cc_dev/git_source/TensorRT-Edge-LLM_Local")

        eagle3_export_cmd = f"python3 export_onnx.py --model {output_dir / 'eagle3_checkpoint'} --output {eagle3_output_path} --eagle3"
        run_command(eagle3_export_cmd, cwd="/mnt/test_cc_dev/git_source/TensorRT-Edge-LLM_Local")

    # 步骤4: 单元测试验证
    if not args.skip_unit_tests:
        print("\n=== 步骤4: 单元测试验证 ===")
        test_cmd = f"pytest tests/quantization/test_{args.quantization}.py -v"
        run_command(test_cmd, cwd="/mnt/test_cc_dev/git_source/TensorRT-Edge-LLM_Local")

        # 验证ONNX模型正确性
        onnx_test_cmd = f"python3 tests/onnx/test_onnx_export.py --model {onnx_output_path}"
        run_command(onnx_test_cmd, cwd="/mnt/test_cc_dev/git_source/TensorRT-Edge-LLM_Local")

        if args.enable_eagle3:
            eagle3_test_cmd = f"python3 tests/eagle3/test_eagle3_export.py --model {eagle3_output_path}"
            run_command(eagle3_test_cmd, cwd="/mnt/test_cc_dev/git_source/TensorRT-Edge-LLM_Local")

    # 生成结果报告
    result = {
        "model": args.model,
        "quantization": args.quantization,
        "enable_eagle3": args.enable_eagle3,
        "output_dir": str(output_dir),
        "onnx_path": str(onnx_output_path),
        "eagle3_onnx_path": str(eagle3_output_path) if args.enable_eagle3 else None,
        "status": "success"
    }

    with open(output_dir / "optimize_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== 优化流水线执行完成 ===")
    print(f"结果报告: {output_dir / 'optimize_result.json'}")
    print(f"ONNX模型路径: {onnx_output_path}")
    if args.enable_eagle3:
        print(f"Eagle3模型路径: {eagle3_output_path}")

if __name__ == "__main__":
    main()
