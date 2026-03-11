# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
One-stop model optimization script for TensorRT Edge-LLM.

This script provides a single command to quantize, optimize, and export models
for end-to-end deployment on edge devices.
"""

import argparse
import os
import sys
import traceback
from typing import Optional

from tensorrt_edgellm.quantization.llm_quantization import quantize_and_save_llm
from tensorrt_edgellm.onnx_export.llm_export import export_llm
from tensorrt_edgellm.onnx_export.visual_export import export_visual

def main() -> None:
    """
    Main function for one-stop model optimization.
    """
    parser = argparse.ArgumentParser(
        description="One-stop model optimization for TensorRT Edge-LLM"
    )

    # Common parameters
    parser.add_argument("--model_dir",
                        type=str,
                        required=True,
                        help="Path to the input HuggingFace model directory")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Path to save the optimized model")
    parser.add_argument("--model_type",
                        type=str,
                        required=True,
                        choices=["llm", "vlm", "audio"],
                        help="Type of model to optimize")
    parser.add_argument("--target_hardware",
                        type=str,
                        required=True,
                        choices=["agx_orin", "orin_nx_16gb", "orin_nx_8gb", "drive_thor"],
                        help="Target edge hardware for deployment")
    parser.add_argument("--device",
                        type=str,
                        required=False,
                        default="cuda",
                        help="Device to use for optimization (cuda/cpu)")

    # Quantization parameters
    parser.add_argument("--quantization",
                        type=str,
                        required=False,
                        choices=["auto", "fp8", "lepto_fp8", "int4_awq", "w4a8", "int8_sq", "none"],
                        default="auto",
                        help="Quantization method, 'auto' selects best for target hardware")
    parser.add_argument("--smoothquant_alpha",
                        type=float,
                        required=False,
                        default=0.5,
                        help="SmoothQuant alpha parameter (0.0-1.0)")
    parser.add_argument("--quantize_visual_module",
                        action="store_true",
                        default=False,
                        help="Quantize visual module (VLM only, default: disabled to preserve visual precision)")

    # Eagle3 parameters
    parser.add_argument("--enable_eagle3",
                        action="store_true",
                        default=False,
                        help="Enable Eagle3 speculative decoding optimization")
    parser.add_argument("--eagle3_draft_model_dir",
                        type=str,
                        required=False,
                        help="Path to pre-trained Eagle3 draft model (required if enable_eagle3 is True)")
    parser.add_argument("--max_draft_length",
                        type=int,
                        default=6,
                        help="Maximum draft tokens to generate for Eagle3")

    # Export parameters
    parser.add_argument("--export_onnx",
                        action="store_true",
                        default=True,
                        help="Export optimized model to ONNX format")
    parser.add_argument("--opset_version",
                        type=int,
                        default=18,
                        help="ONNX opset version")

    args = parser.parse_args()

    try:
        print(f"Starting one-stop model optimization for {args.model_type} model")
        print(f"Target hardware: {args.target_hardware}")

        # Auto-select quantization based on target hardware
        if args.quantization == "auto":
            if args.target_hardware == "agx_orin":
                args.quantization = "lepto_fp8"
            elif args.target_hardware == "orin_nx_16gb":
                args.quantization = "int4_awq"
            elif args.target_hardware == "orin_nx_8gb":
                args.quantization = "w4a8"
            elif args.target_hardware == "drive_thor":
                args.quantization = "nvfp4"
            print(f"Auto-selected quantization: {args.quantization}")

        # Create output directories
        quantized_dir = os.path.join(args.output_dir, "quantized_model")
        onnx_dir = os.path.join(args.output_dir, "onnx")
        os.makedirs(quantized_dir, exist_ok=True)
        os.makedirs(onnx_dir, exist_ok=True)

        # Step 1: Quantize LLM model
        print(f"\nStep 1/3: Quantizing model with {args.quantization} quantization")
        if args.quantization != "none":
            quantize_and_save_llm(
                model_dir=args.model_dir,
                output_dir=quantized_dir,
                quantization=args.quantization,
                dtype="fp16",
                dataset_dir="cnn_dailymail",
                lm_head_quantization=args.quantization if args.quantization in ["fp8", "lepto_fp8", "nvfp4", "mxfp8"] else None,
                kv_cache_quantization="fp8" if args.quantization in ["fp8", "lepto_fp8"] else None,
                smoothquant_alpha=args.smoothquant_alpha,
                device=args.device
            )
        else:
            print("Quantization disabled, skipping step 1")
            quantized_dir = args.model_dir

        # Step 2: Export to ONNX
        if args.export_onnx:
            print(f"\nStep 2/3: Exporting model to ONNX format")
            export_llm(
                model_dir=quantized_dir,
                output_dir=os.path.join(onnx_dir, "llm"),
                opset_version=args.opset_version,
                device=args.device
            )

            # Export visual encoder for VLM
            if args.model_type == "vlm":
                print("Exporting visual encoder for VLM model")
                export_visual(
                    model_dir=args.model_dir,
                    output_dir=os.path.join(onnx_dir, "visual"),
                    opset_version=args.opset_version,
                    device=args.device,
                    quantize=args.quantize_visual_module,
                    precision=args.quantization if args.quantize_visual_module else "none"
                )

        # Step 3: Process Eagle3 draft model
        if args.enable_eagle3:
            print(f"\nStep 3/3: Processing Eagle3 draft model")
            from tensorrt_edgellm.quantization.llm_quantization import quantize_and_save_draft
            from tensorrt_edgellm.onnx_export.llm_export import export_draft_model

            assert args.eagle3_draft_model_dir is not None, "eagle3_draft_model_dir is required when enable_eagle3 is True"

            eagle3_quantized_dir = os.path.join(args.output_dir, "eagle3_quantized")
            eagle3_onnx_dir = os.path.join(onnx_dir, "eagle3")
            os.makedirs(eagle3_quantized_dir, exist_ok=True)
            os.makedirs(eagle3_onnx_dir, exist_ok=True)

            # Quantize Eagle3 draft model
            print("Quantizing Eagle3 draft model")
            quantize_and_save_draft(
                base_model_dir=quantized_dir,
                draft_model_dir=args.eagle3_draft_model_dir,
                output_dir=eagle3_quantized_dir,
                quantization=args.quantization,
                device=args.device,
                dtype="fp16",
                dataset_dir="cnn_dailymail",
                lm_head_quantization=args.quantization if args.quantization in ["fp8", "lepto_fp8", "nvfp4", "mxfp8"] else None,
                kv_cache_quantization="fp8" if args.quantization in ["fp8", "lepto_fp8"] else None,
                smoothquant_alpha=args.smoothquant_alpha
            )

            # Export Eagle3 ONNX
            if args.export_onnx:
                print("Exporting Eagle3 draft model to ONNX")
                export_draft_model(
                    draft_model_dir=eagle3_quantized_dir,
                    output_dir=eagle3_onnx_dir,
                    base_model_dir=quantized_dir,
                    device=args.device
                )

        print(f"\n✅ Optimization completed successfully!")
        print(f"Output saved to: {args.output_dir}")
        print(f"\nNext steps:")
        print(f"1. Copy the {args.output_dir} directory to your target edge device")
        print(f"2. On the edge device, run the TensorRT engine compilation command:")
        print(f"   ./build/examples/llm/llm_build --onnxDir={os.path.basename(args.output_dir)}/onnx --engineDir=./engine --maxBatchSize=1")
        if args.model_type == "vlm":
            print(f"   Add --vlm flag for VLM models")
        if args.enable_eagle3:
            print(f"   Add --eagle flag to enable speculative decoding")

    except Exception as e:
        print(f"❌ Error during model optimization: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
