# J-lens 与大模型可解释性资源

## Knowledge

### 当前仓库与直接上游

- [仓库 README](./README.md)
  项目目标、快速启动、核心公式、适用模型和限制。用于建立全局地图。
- [Lens 文档](./docs/lenses.md)
  说明预拟合 lens、Neuronpedia lens、自行拟合、logit lens 回退方案及模型兼容范围。用于理解“lens 从哪里来”。
- [干预文档](./docs/interventions.md)
  说明 Replace、Add、Remove、Erase 的数学含义、位置范围、验证规则和实现路径。用于区分读出证据与因果证据。
- [性能优化历史](./docs/perf/README.md)
  记录拟合、解码、界面和搜索的瓶颈与正确性门槛。用于理解研究代码为何出现自定义内核和分析式 Jacobian。
- [`jlens_qwen/model.py`](./jlens_qwen/model.py)
  模型加载、逐层残差捕获、缓存解码和干预注入的核心实现。用于追踪一次推理。
- [`jlens_qwen/lens.py`](./jlens_qwen/lens.py)
  保存、加载和应用每层 Jacobian 矩阵。用于把公式映射到代码。
- [`jlens_qwen/fit_analytic.py`](./jlens_qwen/fit_analytic.py)
  逐层 Jacobian 的计算、链式相乘、提示词平均和检查点恢复。用于理解 lens 的拟合过程。
- [原始上游仓库：WeZZard/jlens-qwen36](https://github.com/WeZZard/jlens-qwen36)
  用于核对后续变化、问题讨论和发布文件。

### 论文与参考实现

- [Anthropic：Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
  当前项目直接借鉴的研究。用于理解 Jacobian lens 的研究动机、方法和实验边界。
- [Anthropic：jacobian-lens 参考实现](https://github.com/anthropics/jacobian-lens)
  论文配套代码。用于比较 MLX 实现与 PyTorch 参考实现。
- [Neuronpedia J-lens](https://neuronpedia.org/jlens)
  提供较大提示词规模拟合的公开 lens。用于比较演示级 lens 与更稳定的 lens。

### 模型与运行平台

- [Qwen3.6-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.6-27B)
  原仓库默认模型；27B 参数、64 层、隐藏维度 5120、混合 Gated DeltaNet/注意力架构。
- [Qwen3.5-2B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)
  第二阶段本地候选；同属 `qwen3_5` 架构家族，规模显著降低。
- [Qwen3.5-0.8B-Base 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
  第一阶段推荐模型；用于受控的下一词预测、激活捕获和小规模梯度实验。
- [MLX 官方仓库](https://github.com/ml-explore/mlx)
  解释 MLX 的延迟执行、自动微分和统一内存设计。用于理解原项目的技术选择。
- [NVIDIA CUDA on WSL 官方指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
  用于验证 WSL 2、Windows 驱动和 CUDA 是否正确连接。
- [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)
  用于按当前 CUDA 版本安装兼容的 PyTorch 构建。
- [Hugging Face Transformers 量化文档](https://huggingface.co/docs/transformers/main/en/quantization)
  用于理解 4-bit/8-bit 权重量化能节省什么，以及它与梯度型可解释性实验的差异。

### 前置知识

- [3Blue1Brown：Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
  通过空间变换理解向量、矩阵、基和线性映射。用于建立 Jacobian 公式的直觉。
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  用图解释词元、嵌入、注意力和逐层信息处理。用于进入代码前建立整体图景。
- [Neel Nanda：TransformerLens 演示与文档](https://transformerlensorg.github.io/TransformerLens/)
  介绍激活缓存、hook、logit attribution 和可解释性实验工作流。用于迁移到 CUDA/PyTorch 教学实验。

## Wisdom (Communities)

- [当前仓库 Issues](https://github.com/Nishijujuba/jlens-qwen36-sss/issues)
  用于报告复现问题、记录硬件差异并向维护者确认设计意图。
- [Anthropic jacobian-lens Issues](https://github.com/anthropics/jacobian-lens/issues)
  用于确认参考实现的限制、模型兼容性和研究复现细节。
- [TransformerLens Discussions](https://github.com/TransformerLensOrg/TransformerLens/discussions)
  用于询问 hook、激活缓存、归因和干预实验的实践问题。

## Gaps

- 当前未发现这个仓库经过验证的 Windows/WSL + NVIDIA CUDA 完整移植。
- 当前缺少 RTX 3080 Laptop 16 GB 上拟合 Qwen3.5 小模型 Jacobian lens 的可靠耗时和峰值显存基线。
- Qwen3.5 的混合 Gated DeltaNet 架构对通用可解释性工具的支持仍需逐项验证，不能假设 TransformerLens 已覆盖全部内部模块。
