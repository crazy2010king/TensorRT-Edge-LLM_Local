# TensorRT Edge-LLM 性能测试套件使用说明

## 概述
本性能测试套件专为Jetson AGX Orin设备上的Qwen3-VL-2B-Instruct模型设计，提供全面的性能测试能力，覆盖端侧AI模型的所有标准测试场景。

## 目录结构
```
├── performance_test.sh              # 主测试入口脚本
├── config/
│   └── test_config.yaml             # 测试配置文件（可自定义测试参数）
├── scripts/
│   ├── setup_env.sh                 # 环境初始化和依赖检查
│   ├── run_test_case.sh             # 单个测试用例执行
│   ├── collect_metrics.sh           # 系统性能指标收集
│   └── generate_report.sh           # 性能报告生成
├── test_cases/
│   ├── single_request.json          # 单请求测试用例
│   ├── concurrent_requests.json     # 并发测试用例
│   └── multi_modal.json             # 多模态测试用例
└── results/
    ├── raw_data/                    # 原始测试数据（JSON/CSV格式）
    ├── reports/                     # 生成的测试报告（Markdown/HTML/JSON）
    └── logs/                        # 测试运行日志
```

## 快速开始

### 1. 基础运行
```bash
# 运行完整测试套件
./performance_test.sh

# 仅运行基础性能测试
./performance_test.sh --test-suite basic

# 自定义参数运行
./performance_test.sh \
    --warmup 3 \
    --runs 5 \
    --output-dir ./my_results \
    --test-suite all
```

### 2. 与部署流程集成
在远端部署时自动运行性能测试：
```bash
./scripts/remote_deployment.sh --run-performance-test
```

### 3. 命令行参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-c, --config FILE` | 测试配置文件路径 | `config/test_config.yaml` |
| `-w, --warmup NUM` | 预热运行次数 | 5 |
| `-r, --runs NUM` | 每个测试用例运行次数 | 10 |
| `-o, --output-dir DIR` | 结果输出目录 | `results` |
| `-t, --test-suite NAME` | 测试套件：all/basic/edge/model/engineering | `all` |
| `-v, --verbose` | 启用详细输出 | 关闭 |
| `-h, --help` | 显示帮助信息 | - |

## 测试场景覆盖

### 基础性能测试
- 单请求延迟测试（总延迟、分阶段延迟统计）
- 并发吞吐量测试（1/2/4/8并发等级）
- 内存使用测试（GPU/CPU内存峰值/平均值）
- 资源利用率测试（GPU/CPU利用率监控）

### 端侧特定测试
- 冷启动性能测试（引擎首次加载和初始化时间）
- 稳定性测试（长时间运行稳定性和资源泄漏检测）
- 能效测试（功率消耗和能效比计算）

### 模型特定测试
- 不同输入长度测试（128/256/512/1024 token）
- 不同输出长度测试（32/64/128/512 token）
- 多模态输入测试（不同图像+文本组合）
- 不同图像分辨率测试（640x480到2560x1440）

### 工程化测试
- 预热优化效果测试
- 参数调整测试（温度、top_p、max_new_tokens影响）
- 多次运行稳定性测试

## 核心指标覆盖
| 指标类别 | 具体指标 |
|----------|----------|
| 延迟指标 | 总延迟、P50/P90/P95/P99延迟、分阶段延迟（初始化/编码/生成）、首Token延迟 |
| 吞吐量指标 | 单请求吞吐量、并发吞吐量、Token生成速率（tokens/sec） |
| 资源占用 | GPU显存峰值/平均值、CPU内存峰值/平均值、GPU/CPU利用率、功耗 |
| 算法指标 | Prefill Token重用率、Eagle speculative decoding接受率 |

## 测试报告
测试完成后会自动生成以下报告：
1. **Markdown报告**：适合阅读和归档，包含所有指标统计结果
2. **JSON数据**：结构化原始数据，适合自动化分析和历史对比
3. **系统指标CSV**：GPU/CPU/功耗等实时监控数据

报告默认位置：`results/reports/<测试ID>/`

## 依赖要求
- Python 3.8+
- jq, yq 命令行工具
- 必要Python包：numpy, pandas, matplotlib, pyyaml
- Jetson设备要求：tegrastats工具（系统自带）

## 最佳实践
1. 测试前关闭不必要的后台进程，确保测试结果准确
2. 建议默认使用5次预热和10次测试运行，获得稳定的统计结果
3. 对于基线测试，建议固定设备功率模式为MAXN
4. 性能测试建议在空闲设备上运行，避免其他进程干扰
