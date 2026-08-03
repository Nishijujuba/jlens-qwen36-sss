# Teaching Notes

## Learner profile

- 模型可解释性基础：从零开始。
- 大模型基础：了解 Transformer、大模型训练和推理的大致概念，尚未系统掌握公式和实现细节。
- 编程基础：能阅读 Python；讲解代码时需要把张量形状、数据流和框架行为写清楚。
- 教学语言：中文为主，必要英文术语在第一次出现时给出中文解释。
- 教学顺序：直觉与例子 → 变量和形状 → 最小公式 → 代码位置 → 验证练习。
- 证据要求：每次明确区分观察、推断、因果验证和研究结论。

## Hardware snapshot

来自 2026-08-02 提供的 Windows 任务管理器截图：

- Windows 11 笔记本，已安装 WSL。
- NVIDIA GeForce RTX 3080 Laptop GPU。
- 专用 GPU 显存：16.0 GB。
- 共享 GPU 内存上限：15.9 GB。
- 系统内存：31.9 GB。

共享 GPU 内存来自系统内存，访问路径和性能与专用显存不同。它不能被当成额外的 15.9 GB 高速显存预算。

## Naming correction

仓库名中的 `qwen36` 指 **Qwen 3.6**。原默认模型是 **Qwen3.6-27B**，参数规模为 27B。它并非一个名为“Qwen 36B”的 36B 参数模型。

## Local model decision

### 第一阶段：`Qwen/Qwen3.5-0.8B-Base`

用途：理解前向传播、捕获隐藏状态、实现 Logit Lens、完成短序列的小规模梯度/Jacobian 实验。

选择理由：

- 与原模型同属 `qwen3_5` 架构家族，概念迁移成本较低。
- 24 层、隐藏维度 1024；单层 `1024 × 1024` Jacobian 远小于 27B 模型的 `5120 × 5120`。
- BF16/FP16 权重主体约为数 GB 量级，为激活、读出和临时张量留下显存空间。
- Base 模型适合受控的下一词预测实验，减少助手对齐行为带来的额外变量。

### 第二阶段：`Qwen/Qwen3.5-2B`

用途：验证在更有行为能力的模型上，读出图样和干预结果是否仍然出现。

- 16 GB 专用显存通常足以支持短上下文、batch size 1 的激活捕获和受控推理实验。
- 完整 Jacobian 拟合的峰值显存与耗时仍需实测，不能依据权重大小直接推断可行。

### 暂缓：Qwen3.5-4B 与 Qwen3.6-27B

- Qwen3.5-4B 可用于后续只读激活、Logit Lens 或量化推理；梯度型实验会明显压缩显存余量。
- Qwen3.6-27B 4-bit 的权重本体已经接近 16 GB 专用显存上限。原仓库还需要 lens 矩阵、激活、缓存、量化元数据和框架工作区。
- CPU/系统内存卸载可能让短推理启动，速度和稳定性不适合作为初学阶段基线。

## Platform decision

- `main` 保留原始 Apple Silicon / MLX 研究实现。
- `teach` 的默认活动代码采用 WSL 2 + CUDA + PyTorch + Hugging Face Transformers。
- 原仓库的自定义 GDN 反向路径使用 Metal 内核，完整 CUDA 移植属于独立工程任务。
- 教学版用官方多模态模型类加载权重，文本实验中释放未使用的视觉塔。
- 默认运行 Logit Lens；只有提供与 24 层、1024 隐藏维度严格匹配的 NPZ 文件时才启用 Jacobian Lens。

## Implementation status

已建立 `jlens_wsl/` 运行路径，包含：

- 官方 `Qwen/Qwen3.5-0.8B-Base` 加载；
- 24 层 PyTorch forward hook；
- 分层 Top-K Logit/Jacobian 读出；
- `add`、`remove`、`replace` 残差干预；
- FastAPI + SSE 网页调试器；
- WSL/CUDA 预检、安装、真实模型冒烟脚本；
- VS Code Remote - WSL 调试配置；
- 不下载模型的假模型单元测试。

教学版故意采用逐词完整重放。它牺牲速度，换取清晰的层钩子语义、可重复 A/B 实验和容易下断点的代码路径。

## Unknown unknowns to check on the target laptop

- Windows NVIDIA 驱动、WSL 内核与 PyTorch CUDA wheel 的实际兼容性。
- RTX 3080 Laptop 的实际 TGP、散热和持续频率。
- WSL 的内存上限、swap 和 Windows 后台占用。
- Qwen3.5 Gated DeltaNet 在未安装可选 FLA/causal-conv1d 内核时的实际速度。
- 真实模型加载后的专用显存、生成每词耗时和峰值显存。
- 多模态包装层在当前 Transformers 版本中的内部属性是否变化。
- lens 的可读性随提示词数量、提示词分布、模型版本和精度的变化。
