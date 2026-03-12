# TensorRT-Edge-LLM 最佳端侧设备优化方案 实施规划

## 一、项目概述
### 1.1 项目背景
本项目将AngelSlim大模型压缩工具包与TensorRT-Edge-LLM端侧部署框架深度集成，形成端到端的端侧大模型优化流水线，实现以下核心目标：
- 内存占用降低50%-75%
- 推理速度提升2-4倍
- 精度损失控制在2%以内

### 1.2 适用范围
- **支持平台**：NVIDIA Jetson系列（Orin NX/AGX Orin）、DRIVE Thor
- **支持模型**：
  - LLM：Qwen3、Qwen2.5、Hunyuan、DeepSeek等
  - VLM：Qwen3-VL、HunyuanVL等
  - Audio模型、Diffusion生成模型、MoE模型

---

## 二、已完成工作
✅ **阶段一（基础功能集成）已全部完成**：
1. 集成AngelSlim FP8/INT4量化算法到TensorRT-Edge-LLM导出流水线
2. 支持FP8/INT4量化模型的TensorRT引擎编译
3. 验证Qwen3/Llama系列模型量化精度和性能达标

✅ **阶段二（高级特性集成）已全部完成**：
1. **高级量化算法集成**：
   - LeptoQuant增强FP8量化（精度提升1-2%）
   - SmoothQuant平滑量化
   - VLM视觉模块保护量化（视觉精度损失<0.5%）
2. **Eagle3推测解码全流程**：
   - Eagle3草稿模型训练工具集成（x86）
   - Eagle3 ONNX导出支持
   - Eagle3 SpecDecode运行时集成（端侧C++）
   - Tree decoding优化
   - 推理速度在FP8基础上再提升1.4-1.9倍
3. **VLM端到端优化**：
   - Qwen3-VL-2B全流程支持
   - 多模态精度/性能达标：AGX Orin上生成速度≥25 token/s，精度损失<2%

✅ **阶段三（极致优化开发）核心功能完成**：
1. **极低比特量化支持**：
   - NVFP4量化支持（Blackwell架构/DRIVE Thor）
   - Tequila 2bit三值量化算子开发（3B模型可在4GB设备运行，精度损失<3%）
   - Sherry 1.25bit量化算子开发（极致内存压缩）
2. **推理性能优化**：
   - CUDA Graph支持Eagle3场景
   - 分层KV Cache优化（FP8/INT4存储）
   - 推理延迟降低10-30%，KV Cache内存占用降低30%

---

## 三、待实施计划
### 3.1 阶段三剩余工作（2026.4.4-2026.5.2）
| 任务ID | 任务名称 | 子任务 | 验收标准 |
|--------|----------|--------|----------|
| T3.3 | 自动化工具链开发 | 1. x86一站式量化导出工具<br>2. 端侧一站式编译部署工具 | 单命令即可完成全流程优化，无需手动分步操作，易用性提升80% |

### 3.2 阶段四：场景落地迭代（持续进行）
| 任务ID | 任务名称 | 交付物 | 完成时间 |
|--------|----------|--------|----------|
| T4.1 | 车载场景定制优化方案 | 场景化优化指南、预编译引擎、集成示例 | 2026.5.15 |
| T4.2 | 机器人场景定制优化方案 | 场景化优化指南、预编译引擎、集成示例 | 2026.5.30 |
| T4.3 | 工业场景定制优化方案 | 场景化优化指南、预编译引擎、集成示例 | 2026.6.15 |
| T4.4 | 场景化性能基准库 | 公开性能/精度对比报告 | 持续更新 |

---

## 四、端到端执行流程（标准化）
### 4.1 x86侧执行步骤（量化+导出）
```bash
# 1. 环境准备
pip install angelslim[all] tensorrt_edgellm

# 2. 量化压缩（支持fp8/fp8_lepto/int4_awq/int8_sq/2bit_tequila/1.25bit_sherry）
python /path/to/AngelSlim/tools/run.py --config my_config.yaml

# 3. 可选：训练Eagle3草稿模型
python /path/to/AngelSlim/tools/train_eagle3_offline.py --model ./quantized_model --output ./eagle3_draft

# 4. ONNX导出
tensorrt-edgellm-export-llm --model_dir ./quantized_model --output_dir ./onnx_output
tensorrt-edgellm-export-visual --model_dir Qwen/Qwen3-VL-2B --output_dir ./onnx_output/visual
```

### 4.2 端侧执行步骤（编译+部署）
```bash
# 1. 编译TensorRT引擎
./build/examples/llm/llm_build --onnxDir=./onnx_output --engineDir=./engine_output --maxBatchSize=1 --vlm --fp8 --sm 87

# 2. 运行推理
./build/examples/llm/llm_inference_spec_decode --engineDir=./engine_output \
  --draft-engine-dir=./engine_output/eagle3 \
  --inputFile=test_case.json \
  --enable-cuda-graph \
  --kv-cache-fp8
```

---

## 五、分硬件最佳实践
| 硬件平台 | 推荐模型大小 | 最佳量化方案 | 预期性能 |
|----------|--------------|--------------|----------|
| Jetson AGX Orin 32GB | 2B-7B | FP8 + Eagle3 / INT4-AWQ + Eagle3 | Qwen3-VL-2B: 25-35 token/s |
| Jetson Orin NX 16GB | 1B-4B | INT4-AWQ + Eagle3 | Qwen3-1.7B: 30-40 token/s |
| Jetson Orin NX 8GB | 0.5B-1.5B | INT4-AWQ + Eagle3 / 2bit Tequila | Qwen3-0.6B: 40-50 token/s |
| DRIVE Thor | 7B-32B | NVFP4 + Eagle3 | Qwen3-8B: 80-100 token/s |

---

## 六、质量保障体系
### 6.1 精度验证（x86侧执行）
- 量化后先在x86环境用LM-Evaluation-Harness验证CEVAL、MMLU等指标
- 精度损失超过2%自动降级到更高精度或调整量化参数
- 关键任务定制验证数据集，确保业务精度达标

### 6.2 端侧验证（必须在真实硬件执行）
- 端侧运行时和x86侧量化结果精度对齐验证
- 性能基准测试：延迟、吞吐量、显存占用、功耗
- 7*24小时压力测试，确保运行稳定性

---

## 七、关键交付物清单
| 交付物 | 输出路径 | 完成节点 |
|--------|----------|----------|
| 1. 高级量化算法代码 | `tensorrt_edgellm/quantization/` | 已完成 |
| 2. Eagle3运行时代码 | `cpp/runtime/spec_decode/` | 已完成 |
| 3. 低比特量化算子库 | `cpp/kernels/low_bit/` | 已完成 |
| 4. 自动化工具链 | `tools/optimize/` | 阶段三结束 |
| 5. 各场景优化指南 | `docs/scenario_guide/` | 阶段四结束 |
| 6. 完整性能/精度测试报告 | `docs/reports/` | 各阶段结束 |

---

## 八、实施进度跟踪
| 阶段 | 完成度 | 状态 |
|------|--------|------|
| 阶段一：基础功能集成 | 100% | ✅ 已完成 |
| 阶段二：高级特性集成 | 100% | ✅ 已完成 |
| 阶段三：极致优化开发 | 70% | ⚙️ 进行中 |
| 阶段四：场景落地迭代 | 0% | ⏳ 待启动 |

---

**文档版本**：v1.0
**最后更新**：2026-03-12
**维护人**：TensorRT-Edge-LLM团队
