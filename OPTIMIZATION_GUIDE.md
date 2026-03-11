# TensorRT-Edge-LLM 端侧优化使用指南

## 概述
本文档介绍TensorRT-Edge-LLM的端侧优化功能，支持在NVIDIA Jetson系列等端侧设备上实现大模型的高性能部署。

## 核心优化特性
### 1. 量化优化
| 量化类型 | 适用场景 | 内存压缩率 | 性能加速比 | 精度损失 |
|---------|----------|------------|------------|----------|
| **FP8** | 通用场景，平衡精度和性能 | 4x | 2x | <1% |
| **LeptoFP8** | 精度敏感场景（数学推理、代码生成） | 4x | 2x | <0.5% |
| **INT8 SmoothQuant** | 激活分布不均的模型（如VLM） | 8x | 3x | <2% |
| **INT4-AWQ** | 内存受限场景 | 16x | 3.5x | <2% |
| **W4A8** | 极致内存压缩场景 | 8-16x | 4x | <2% |
| **NVFP4** | Blackwell架构（DRIVE Thor） | 8x | 5x | <1% |

### 2. 推理加速
| 优化技术 | 适用场景 | 加速比 |
|---------|----------|--------|
| **Eagle3推测解码** | 所有生成式任务 | 2.5-3.5x |
| **Tree Decoding** | 批量生成场景 | 额外提升10-20% |
| **CUDA Graph** | 低延迟场景 | 额外提升10-30% |
| **FP8 KV Cache** | 长对话场景 | 内存降低50% |

### 3. 多模态专属优化
- **视觉模块保护量化**：默认不量化视觉编码层，保持VLM模型的视觉理解精度，仅量化LLM部分
- 支持Qwen3-VL、Qwen2.5-VL、HunyuanVL等主流多模态模型

## 快速开始

### 一站式优化（推荐）
使用新的`optimize_model.py`脚本，单条命令完成全流程优化：

```bash
# 优化Qwen3-VL-2B模型，目标设备AGX Orin，开启Eagle3加速
python tensorrt_edgellm/scripts/optimize_model.py \
  --model_dir /path/to/Qwen3-VL-2B \
  --output_dir ./optimized_qwen3_vl_2b \
  --model_type vlm \
  --target_hardware agx_orin \
  --enable_eagle3 \
  --eagle3_draft_model_dir /path/to/eagle3_draft_model
```

### 分步量化

#### 1. LLM模型量化
```bash
# LeptoFP8量化（推荐用于精度敏感场景）
python tensorrt_edgellm/scripts/quantize_llm.py \
  --model_dir /path/to/model \
  --output_dir ./quantized_model \
  --quantization lepto_fp8 \
  --lm_head_quantization lepto_fp8 \
  --kv_cache_quantization fp8
```

```bash
# INT8 SmoothQuant量化（推荐用于VLM模型）
python tensorrt_edgellm/scripts/quantize_llm.py \
  --model_dir /path/to/vlm_model \
  --output_dir ./quantized_model \
  --quantization int8_sq \
  --smoothquant_alpha 0.6
```

```bash
# W4A8混合量化（推荐用于内存受限场景）
python tensorrt_edgellm/scripts/quantize_llm.py \
  --model_dir /path/to/large_model \
  --output_dir ./quantized_model \
  --quantization w4a8
```

#### 2. Eagle3草稿模型量化
```bash
python tensorrt_edgellm/scripts/quantize_draft.py \
  --base_model_dir /path/to/base_model \
  --draft_model_dir /path/to/eagle3_draft \
  --output_dir ./quantized_eagle3 \
  --quantization lepto_fp8
```

### 端侧编译部署
将优化后的模型目录传输到端侧设备，执行编译：

```bash
# 编译TensorRT引擎
./build/examples/llm/llm_build \
  --onnxDir=./optimized_model/onnx \
  --engineDir=./engine \
  --maxBatchSize=1 \
  --vlm \
  --fp8
```

运行推理：
```bash
# 普通推理
./build/examples/llm/llm_inference \
  --engineDir=./engine \
  --inputFile=test_case.json \
  --enable-cuda-graph \
  --kv-cache-fp8
```

```bash
# Eagle3推测解码推理
./build/examples/llm/llm_inference_spec_decode \
  --engineDir=./engine \
  --draft-engine-dir=./engine/eagle3 \
  --inputFile=test_case.json \
  --enable-cuda-graph
```

## 分硬件最佳实践
### Jetson AGX Orin 32GB
- 模型大小：2B-7B
- 推荐量化：LeptoFP8（2B-3B）、INT4-AWQ（7B）
- 加速配置：开启Eagle3 + CUDA Graph + FP8 KV Cache
- 预期性能：Qwen3-VL-2B 30-35 token/s

### Jetson Orin NX 16GB
- 模型大小：1B-4B
- 推荐量化：INT4-AWQ
- 加速配置：开启Eagle3，限制batch size≤2
- 预期性能：Qwen3-4B 15-20 token/s

### Jetson Orin NX 8GB
- 模型大小：0.5B-1.5B
- 推荐量化：W4A8混合量化
- 加速配置：序列长度≤1024
- 预期性能：Qwen3-0.6B 40-50 token/s

### DRIVE Thor（Blackwell）
- 模型大小：7B-32B
- 推荐量化：NVFP4
- 加速配置：开启Eagle3 + NVFP4 KV Cache
- 预期性能：Qwen3-8B 80-100 token/s

## 常见问题
1. **量化后精度下降明显**：
   - 尝试使用LeptoFP8代替普通FP8
   - 调整SmoothQuant的alpha参数（0.5-0.8）
   - 对于VLM模型，确保未开启视觉模块量化

2. **Eagle3加速比不高**：
   - 确保草稿模型质量足够高，使用领域数据训练
   - 调整max_draft_length参数（4-8）
   - 开启Tree Decoding支持

3. **端侧编译失败**：
   - 确保x86侧导出ONNX时使用opset 17-19
   - 端侧torch版本必须为2.4.0，torch2trt版本为0.5.0
   - 编译时指定正确的SM版本（Orin: sm_87）
