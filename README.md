# J-lens Qwen3.5-0.8B — Windows 11 / WSL 教学版

这个 `teach` 分支把仓库的默认运行路径迁移到 **Windows 11 + WSL 2 + NVIDIA CUDA + PyTorch/Transformers**，默认模型为官方的 [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)。

它提供一个本地网页调试器，用来观察每个 Transformer 层在当前词元位置上更倾向哪些输出词元，并对残差流做可复现的探索性干预。

> `main` 分支保留原始 Apple Silicon / MLX / Qwen3.6-27B 实现。`teach` 分支的活动代码位于 `jlens_wsl/`；原来的 `jlens_qwen/` 保留为架构对照材料，不再是默认安装入口。

## 本地硬件结论

截图中的设备为 RTX 3080 Laptop GPU，16 GiB 专用显存，系统内存约 32 GiB。该配置运行 0.8B 模型的 FP16 推理和 24 层 Logit Lens 读出有充足余量。原版 27B MLX 路径依赖 Apple Metal，无法直接在 WSL/CUDA 上运行；即使改成 4-bit CUDA 权重，完整 27B Jacobian 拟合仍不适合作为这台笔记本的入门实验。

## 一次性安装

### 1. Windows PowerShell

```powershell
wsl --update
wsl --status
```

NVIDIA 驱动安装在 Windows 主机。WSL 内不要安装 Linux 显示驱动。

### 2. WSL 终端

建议把仓库放在 WSL 的 Linux 文件系统，例如 `~/src`，避免 `/mnt/c` 的小文件 I/O 开销。

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/Nishijujuba/jlens-qwen36-sss.git
cd jlens-qwen36-sss
git switch teach

bash scripts/setup_wsl.sh
```

安装脚本会：

1. 创建 `.venv`；
2. 安装 CUDA 版 PyTorch；
3. 安装本仓库及开发依赖；
4. 检查 WSL、CUDA、显存和磁盘；
5. 运行不依赖模型下载的单元测试。

## 启动

```bash
bash scripts/run_wsl.sh
```

浏览器打开：

```text
http://localhost:8765
```

首次启动会从 Hugging Face 下载官方模型。模型缓存通常位于 `~/.cache/huggingface/`。

## 拉取更新后直接调试

```bash
cd ~/src/jlens-qwen36-sss
git switch teach
git pull --ff-only
source .venv/bin/activate
python -m pytest tests_wsl
bash scripts/run_wsl.sh
```

VS Code 中使用 **Remote - WSL** 打开仓库后，选择运行配置：

```text
J-lens WSL: FastAPI debugger
```

随后可在 `jlens_wsl/runtime.py` 的 `_forward_capture()`、`_apply_intervention()`、`analyze()` 或 `generate()` 中下断点。

## 真实模型冒烟测试

```bash
source .venv/bin/activate
python scripts/smoke_qwen.py --device cuda --dtype fp16 --max-new-tokens 1
```

该命令至少完成：官方权重加载、一次真实前向传播、两个层的词元读出和一个生成词元。

Windows PowerShell 也可执行完整目标环境验证：

```powershell
.\scripts\verify_from_windows.ps1
```

## 运行模式

### 1. Logit Lens：默认、立即可用

中间层产生残差向量 `h_l`。Logit Lens 将最终归一化层与输出词表矩阵直接应用到该向量：

```text
score_l = W_U · norm(h_l)
```

直觉上，它像把每层的“草稿向量”拿到最终阅卷器上提前评分。早期层和最终输出使用的向量坐标可能尚未对齐，因此读出是一种探针证据，不能直接等同于模型的真实想法。

### 2. Jacobian Lens：需要匹配的拟合文件

如果设置：

```bash
export JLENS_PATH=data/lens/qwen35-0.8b.npz
```

运行时会读取与原仓库兼容的 `J_<layer>` 矩阵，并计算：

```text
score_l = W_U · norm(J_l · h_l)
```

矩阵必须是 `1024 × 1024`，并与 Qwen3.5-0.8B 的层编号匹配。这个版本没有伪造或捆绑低质量的 0.8B Jacobian lens，因此默认明确运行 Logit Lens。

## 为什么使用 Transformers，而没有选 vLLM / SGLang

vLLM 和 SGLang 擅长高吞吐服务。这个仓库需要层级激活、PyTorch forward hook、残差修改和逐步复现实验；Transformers 暴露这些内部结构，调试路径更短，也更容易验证。

## 为什么逐词完整重放，而没有使用缓存解码

Qwen3.5 是 3:1 的混合架构：三层 Gated DeltaNet 配一层全注意力。缓存解码会同时维护循环状态和 KV 缓存，速度更快，干预语义更复杂。教学版每生成一个词元都完整重放当前序列：

- 每次层钩子看到同一套完整输入；
- 基线与干预更容易做确定性比较；
- 可以直接在 Python 调试器里检查每一层；
- 代价是总生成复杂度随长度近似二次增长。

默认限制为 512 个输入词元、64 个输出词元，适合学习和小实验。

## Base 模型的隐藏风险

`Qwen3.5-0.8B-Base` 是预训练基础模型。它的目标是预测下一个词元，适用于补全、微调和研究。它没有完整的助手式指令对齐。网页默认选择“文本补全”；“聊天模板”仅用于比较模板对内部表示的影响，输出质量不应按对话助手标准评估。

## 可解释性证据层级

| 观察 | 能说明什么 | 不能说明什么 |
|---|---|---|
| 某层 Top-1 出现一个词 | 该探针在此位置给这个词较高分 | 模型一定在“想”这个词 |
| 多个相邻层稳定出现同类词 | 表示具有跨层稳定性 | 该表示必然决定输出 |
| 修改残差后输出稳定变化 | 提供一定因果证据 | 方向只有一个自然语言含义 |
| 多提示词、多随机种子复现 | 结论更稳健 | 已达到研究级机制解释 |

## 环境变量

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `JLENS_MODEL_ID` | `Qwen/Qwen3.5-0.8B-Base` | 模型仓库 |
| `JLENS_DEVICE` | `cuda`（启动脚本） | `auto` / `cuda` / `cpu` |
| `JLENS_DTYPE` | `fp16`（启动脚本） | `auto` / `fp16` / `bf16` / `fp32` |
| `JLENS_PATH` | `none` | Jacobian lens NPZ，`none` 表示 Logit Lens |
| `JLENS_MAX_INPUT_TOKENS` | `512` | 输入上限；超限会报错，不会静默截断 |
| `JLENS_MAX_NEW_TOKENS` | `64` | 生成上限 |
| `JLENS_EAGER_LOAD` | `1` | 服务启动时加载模型 |
| `JLENS_ATTN` | `sdpa` | 全注意力后端 |

## 测试

```bash
source .venv/bin/activate
python -m pytest tests_wsl
python -m compileall -q jlens_wsl scripts
```

测试分为两层：

- `tests_wsl/` 使用小型 Qwen 形状假模型，验证层钩子、读出数学、SSE API 和干预路径；
- `scripts/smoke_qwen.py` 下载并运行真实官方模型，验证版本/API/权重兼容性。

## 目录

```text
jlens_wsl/
  config.py          环境与限制
  lens.py            Logit/Jacobian transport 和批量 Top-K
  runtime.py         模型加载、层钩子、生成、残差干预
  server.py          FastAPI / SSE
  static/index.html  本地网页调试器
scripts/
  setup_wsl.sh       一次性环境安装
  run_wsl.sh         启动服务
  check_wsl.py       WSL/CUDA 预检
  smoke_qwen.py      真实模型冒烟测试
  verify_from_windows.ps1  从 Windows 调用 WSL 完整验证
tests_wsl/           无需下载模型的自动化测试
jlens_qwen/          原始 MLX 实现，保留作对照阅读
```

## 进一步阅读

- [`docs/WSL.md`](docs/WSL.md)：WSL、CUDA、驱动和常见故障
- [`docs/ARCHITECTURE-WSL.md`](docs/ARCHITECTURE-WSL.md)：代码路径与关键设计选择
- [`docs/DEBUGGING.md`](docs/DEBUGGING.md)：VS Code 断点练习
- [`TEACHING-START.md`](TEACHING-START.md)：从零开始的教学路线

## License

Apache-2.0。原仓库来源和第三方致谢保留在 `main` 分支及原始代码文档中。
