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

仍需在 WSL 内验证：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

共享 GPU 内存来自系统内存，访问路径和性能与专用显存不同。它不应被当成额外的 15.9 GB 高速显存预算。

## Naming correction

仓库名中的 `qwen36` 指 **Qwen 3.6**。默认模型是 **Qwen3.6-27B**，参数规模为 27B。它并非一个名为“Qwen 36B”的 36B 参数模型。

## Local model decision

### 第一阶段：`Qwen/Qwen3.5-0.8B-Base`

用途：理解前向传播、捕获隐藏状态、实现 logit lens、完成短序列的小规模梯度/Jacobian 实验。

选择理由：

- 与原模型同属 `qwen3_5` 架构家族，概念迁移成本低。
- 24 层、隐藏维度 1024；单层 `1024 × 1024` Jacobian 远小于 27B 模型的 `5120 × 5120`。
- BF16 权重的理论主体体积约 1.6 GB，为自动微分、激活和临时张量留下显存空间。
- Base 模型适合受控的下一词预测实验，减少聊天模板和后训练行为带来的额外变量。

### 第二阶段：`Qwen/Qwen3.5-2B`

用途：验证在更有行为能力的模型上，读出图样和干预结果是否仍然出现。

选择理由：

- 24 层、隐藏维度 2048，BF16 权重理论主体体积约 4 GB。
- 16 GB 专用显存通常足以支持短上下文、batch size 1 的激活捕获和受控实验。
- 完整 Jacobian 拟合的峰值显存与耗时仍需实测，不能依据权重大小直接推断可行。

### 暂缓：Qwen3.5-4B 与 Qwen3.6-27B

- Qwen3.5-4B 适合后续只读激活、logit lens 或量化推理；梯度型实验会明显压缩显存余量。
- Qwen3.6-27B 4-bit 的权重本体已经接近 16 GB 专用显存上限。原仓库还需要 lens 矩阵、激活、缓存、量化元数据和框架工作区。
- CPU/系统内存卸载可能让短推理勉强启动，速度和稳定性不适合作为初学阶段的实验基线。

## Platform decision

- 原仓库继续作为“研究实现阅读对象”。
- 本地教学实验优先采用 WSL 2 + CUDA + PyTorch + Hugging Face Transformers。
- 原仓库的自定义 GDN 反向路径使用 Metal 内核，完整 CUDA 移植属于独立工程任务。
- MLX 已出现 Linux/CUDA 支持路径，当前仓库仍明确标记 Apple/MLX only，且其性能关键路径依赖 Metal。教学阶段不把未经验证的平台兼容性当作可用能力。

## Unknown unknowns to check later

- RTX 3080 Laptop 的实际 TGP、散热和持续频率会显著影响拟合时间。
- WSL 的内存上限、swap 和 Windows 后台占用可能导致系统先发生内存压力，再出现明确的 CUDA OOM。
- Qwen3.5 的多模态包装层可能增加加载复杂度；首个实验应只走文本路径，并记录实际加载的模块。
- 量化权重适合节省推理显存；梯度、Jacobian 和精细因果干预通常需要更谨慎地处理精度和可微性。
- lens 的可读性会随拟合提示词数量、提示词分布、模型版本和精度变化。单次漂亮结果不构成稳健结论。
