# TensorRT Edge-LLM 项目开发指南

## 项目概述
TensorRT Edge-LLM 是 NVIDIA 针对边缘平台的高性能大语言模型(LLM)和视觉语言模型(VLM)推理框架，支持在资源受限设备（如Jetson、DRIVE平台）上高效部署SOTA语言模型。

## 技术栈
- **核心语言**: C++17 (运行时) / Python 3.8+ (导出工具链)
- **核心依赖**: TensorRT, CUDA, cuDNN
- **构建系统**: CMake
- **代码规范**: 遵循 `CODING_GUIDELINES.md`，使用 clang-format 格式化C++代码

## 目录结构
```
.
├── cpp/                    # C++ 运行时核心代码
│   ├── builder/            # TensorRT引擎构建模块
│   ├── kernels/            # CUDA内核实现
│   └── common/             # 通用工具库
├── cmake/                  # CMake配置文件
├── 3rdParty/               # 第三方依赖库
├── tests/                  # 测试用例
├── docs/                   # 文档
├── .github/                # GitHub配置
├── LICENSE                 # 许可证
├── README.md               # 项目说明
├── CODING_GUIDELINES.md    # 代码规范
├── CONTRIBUTING.md         # 贡献指南
└── CHANGELOG.md            # 版本变更记录
```

## 常用开发命令
### 构建项目
```bash
# 本地构建
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 交叉编译aarch64
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64_linux_toolchain.cmake
```

### 代码检查
```bash
# 格式化代码
clang-format -i $(find cpp -name "*.cpp" -o -name "*.h")

# 运行pre-commit检查
pre-commit run --all-files
```

## 可用自动化技能
### 远程开发
- `/env_exec`: 自动连接172.16.119.188 AGX服务器，进入ros2_nitr Docker容器并导航到项目目录
- `/env_exec5`: 免密登录RDK X5设备(172.16.119.199:2222)并进入项目目录
- `/agx_188`: 仅连接到172.16.119.188服务器

### 代码优化
- `/simplify`: 自动审查代码变更，优化代码质量和效率

## 开发规范
1. C++代码遵循 `CODING_GUIDELINES.md` 规范
2. 提交代码前必须通过pre-commit检查
3. 新功能需要添加对应的测试用例
4. 提交信息遵循约定式提交规范

## 参考文档
- 官方文档: https://nvidia.github.io/TensorRT-Edge-LLM/
- 快速开始: https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/quick-start-guide.html
- 支持模型列表: https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/supported-models.html
