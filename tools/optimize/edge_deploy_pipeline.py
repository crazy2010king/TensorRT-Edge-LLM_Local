#!/usr/bin/env python3
"""
端侧一站式部署工具：单命令完成远程设备的引擎编译、部署、性能测试
"""
import argparse
import os
import sys
import subprocess
import json
import shutil
from pathlib import Path

SUPPORTED_DEVICES = ["agx_orin_32g", "agx_orin_64g", "orin_nx_8g", "orin_nx_16g", "drive_thor"]
DEVICE_ENV_MAP = {
    "agx_orin_32g": "env_exec",
    "agx_orin_64g": "env_exec",
    "orin_nx_8g": "env_exec5",
    "orin_nx_16g": "env_exec5",
    "drive_thor": "env_exec_drive"
}

def run_command(cmd, cwd=None, env=None, capture_output=True):
    """执行命令并返回结果"""
    print(f"[执行命令] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"[错误] 命令执行失败: {cmd}")
        print(f"错误输出: {result.stderr}")
        sys.exit(1)
    if capture_output:
        print(f"[输出] {result.stdout}")
    return result.stdout

def run_remote_command(device, cmd, capture_output=True):
    """在远程设备执行命令"""
    env_cmd = DEVICE_ENV_MAP[device]
    full_cmd = f"{env_cmd} '{cmd}'"
    return run_command(full_cmd, capture_output=capture_output)

def transfer_file_to_remote(device, local_path, remote_path):
    """传输文件到远程设备"""
    print(f"[传输文件] {local_path} -> 远程设备[{device}]:{remote_path}")
    if device.startswith("agx_orin"):
        # AGX设备通过scp传输
        scp_cmd = f"scp {local_path} nvidia@172.16.119.188:/home/nvidia/workspace/TensorRT-Edge-LLM/{remote_path}"
    elif device.startswith("orin_nx"):
        # RDK X5设备
        scp_cmd = f"scp -P 2222 {local_path} sunrise@172.16.119.199:/home/sunrise/TensorRT-Edge-LLM/{remote_path}"
    else:
        print(f"[错误] 不支持的设备类型: {device}")
        sys.exit(1)

    run_command(scp_cmd)

def transfer_file_from_remote(device, remote_path, local_path):
    """从远程设备拉取文件"""
    print(f"[拉取文件] 远程设备[{device}]:{remote_path} -> {local_path}")
    if device.startswith("agx_orin"):
        scp_cmd = f"scp nvidia@172.16.119.188:/home/nvidia/workspace/TensorRT-Edge-LLM/{remote_path} {local_path}"
    elif device.startswith("orin_nx"):
        scp_cmd = f"scp -P 2222 sunrise@172.16.119.199:/home/sunrise/TensorRT-Edge-LLM/{remote_path} {local_path}"
    else:
        print(f"[错误] 不支持的设备类型: {device}")
        sys.exit(1)

    run_command(scp_cmd)

def main():
    parser = argparse.ArgumentParser(description="端侧一站式部署测试工具")
    parser.add_argument("--onnx-path", required=True, help="ONNX模型本地路径")
    parser.add_argument("--device", required=True, choices=SUPPORTED_DEVICES,
                        help=f"目标设备类型，支持: {', '.join(SUPPORTED_DEVICES)}")
    parser.add_argument("--engine-output-path", default="./engine_output", help="引擎输出目录")
    parser.add_argument("--batch-size", type=int, default=1, help="推理批次大小")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--skip-performance-test", action="store_true", help="跳过性能测试")
    parser.add_argument("--eagle3-onnx-path", help="Eagle3草稿模型ONNX路径（可选）")

    args = parser.parse_args()

    # 检查输入文件
    onnx_path = Path(args.onnx_path).resolve()
    if not onnx_path.exists():
        print(f"[错误] ONNX文件不存在: {onnx_path}")
        sys.exit(1)

    if args.eagle3_onnx_path:
        eagle3_onnx_path = Path(args.eagle3_onnx_path).resolve()
        if not eagle3_onnx_path.exists():
            print(f"[错误] Eagle3 ONNX文件不存在: {eagle3_onnx_path}")
            sys.exit(1)
    else:
        eagle3_onnx_path = None

    # 创建输出目录
    output_dir = Path(args.engine_output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 开始端侧部署流水线 ===")
    print(f"目标设备: {args.device}")
    print(f"ONNX模型: {onnx_path}")
    print(f"Eagle3模型: {eagle3_onnx_path if eagle3_onnx_path else '未启用'}")
    print(f"批次大小: {args.batch_size}")
    print(f"最大序列长度: {args.max_seq_len}")

    # 步骤1: 同步代码到远程设备
    print("\n=== 步骤1: 同步代码到远程设备 ===")
    run_remote_command(args.device, "git pull")

    # 步骤2: 传输ONNX模型到远程设备
    print("\n=== 步骤2: 传输ONNX模型到远程设备 ===")
    remote_onnx_path = f"tmp/{onnx_path.name}"
    transfer_file_to_remote(args.device, str(onnx_path), remote_onnx_path)

    if eagle3_onnx_path:
        remote_eagle3_path = f"tmp/{eagle3_onnx_path.name}"
        transfer_file_to_remote(args.device, str(eagle3_onnx_path), remote_eagle3_path)

    # 步骤3: 编译TensorRT引擎
    print("\n=== 步骤3: 编译TensorRT引擎 ===")
    build_cmd = f"python3 build_engine.py --onnx {remote_onnx_path} --output tmp/model.engine --batch-size {args.batch_size} --max-seq-len {args.max_seq_len}"
    if eagle3_onnx_path:
        build_cmd += f" --eagle3-onnx {remote_eagle3_path}"
    run_remote_command(args.device, build_cmd)

    # 步骤4: 执行性能测试
    performance_report = None
    if not args.skip_performance_test:
        print("\n=== 步骤4: 执行性能测试 ===")
        test_cmd = f"bash performance_test.sh --engine tmp/model.engine --device {args.device} --output tmp/performance_report.json"
        run_remote_command(args.device, test_cmd)

        # 拉取性能报告
        local_report_path = output_dir / "performance_report.json"
        transfer_file_from_remote(args.device, "tmp/performance_report.json", str(local_report_path))

        with open(local_report_path, "r") as f:
            performance_report = json.load(f)

        # 拉取引擎文件
        local_engine_path = output_dir / "model.engine"
        transfer_file_from_remote(args.device, "tmp/model.engine", str(local_engine_path))

    # 步骤5: 清理远程临时文件
    print("\n=== 步骤5: 清理远程临时文件 ===")
    run_remote_command(args.device, "rm -rf tmp/*")

    # 生成部署结果
    result = {
        "device": args.device,
        "onnx_path": str(onnx_path),
        "eagle3_onnx_path": str(eagle3_onnx_path) if eagle3_onnx_path else None,
        "engine_path": str(output_dir / "model.engine") if not args.skip_performance_test else None,
        "performance_report": performance_report,
        "status": "success"
    }

    with open(output_dir / "deploy_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== 部署流水线执行完成 ===")
    print(f"结果报告: {output_dir / 'deploy_result.json'}")
    if performance_report:
        print(f"性能测试结果:")
        print(f"  平均延迟: {performance_report.get('avg_latency_ms', 'N/A')} ms")
        print(f"  吞吐量: {performance_report.get('throughput_tps', 'N/A')} token/s")
        print(f"  内存占用: {performance_report.get('memory_usage_mb', 'N/A')} MB")
        print(f"  精度损失: {performance_report.get('accuracy_loss', 'N/A')} %")

if __name__ == "__main__":
    main()
