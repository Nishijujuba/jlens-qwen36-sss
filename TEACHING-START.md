# J-lens 教学工作区入口

该分支同时承担两件事：保存从零开始的可解释性课程，并提供可在 Windows 11 + WSL + NVIDIA CUDA 上调试的 Qwen3.5-0.8B 实现。`main` 继续保存原始 Apple Silicon / MLX / Qwen3.6-27B 路径。

## 当前可运行路径

```text
浏览器
  ↓ HTTP / SSE
jlens_wsl/server.py
  ↓ 调用
jlens_wsl/runtime.py
  ↓ 捕获层激活、生成、干预
jlens_wsl/lens.py
  ↓ 最终 RMSNorm + lm_head
每层 Top-K 词元读出
```

默认模型是 `Qwen/Qwen3.5-0.8B-Base`，默认探针是 Logit Lens。真实 Jacobian Lens 需要另行提供与 24 层、隐藏维度 1024 完全匹配的拟合矩阵。

## 第一次启动

在 Windows PowerShell 更新并确认 WSL：

```powershell
wsl --update
wsl --status
```

在 WSL 终端执行：

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/Nishijujuba/jlens-qwen36-sss.git
cd jlens-qwen36-sss
git switch teach
bash scripts/setup_wsl.sh
bash scripts/run_wsl.sh
```

浏览器打开 `http://localhost:8765`。

## 学习顺序

| 顺序 | 文件或动作 | 本阶段成果 |
|---|---|---|
| 0 | [`MISSION.md`](MISSION.md) | 明确学习目标、完成标准和边界 |
| 1 | [`lessons/0001-from-text-to-residual-stream.html`](lessons/0001-from-text-to-residual-stream.html) | 理解词元、向量、层、残差流和主公式 |
| 2 | [`lessons/0002-read-the-code-path.html`](lessons/0002-read-the-code-path.html) | 定位原 MLX 实现的读取、拟合和写入路径 |
| 3 | [`lessons/0003-wsl-and-model-choice.html`](lessons/0003-wsl-and-model-choice.html) | 理解硬件约束和 0.8B 选择依据 |
| 4 | [`docs/ARCHITECTURE-WSL.md`](docs/ARCHITECTURE-WSL.md) | 阅读新的 PyTorch 数据流与设计取舍 |
| 5 | [`docs/DEBUGGING.md`](docs/DEBUGGING.md) | 在 VS Code 中逐层下断点并检查张量 |
| 验收 | `python scripts/smoke_qwen.py --device cuda --dtype fp16` | 完成真实权重、前向、读出和生成 |
| 随时 | [`reference/0001-glossary.html`](reference/0001-glossary.html) | 查阅术语和公式 |
| 持续 | [`learning-records/`](learning-records/) | 记录真正掌握的内容 |

## HTML 课程

```bash
source .venv/bin/activate
python -m http.server 8000
```

浏览器打开：

```text
http://localhost:8000/lessons/0001-from-text-to-residual-stream.html
```

## 当前文件结构

```text
MISSION.md                         学习任务与完成标准
RESOURCES.md                       经过筛选的知识来源与社区
NOTES.md                           学习偏好、硬件、实现决策和风险
TEACHING-START.md                  本入口
jlens_wsl/
  config.py                        环境配置
  lens.py                          Logit/Jacobian 读出
  runtime.py                       模型、钩子、生成和残差干预
  server.py                        FastAPI / SSE
  static/index.html                网页调试器
scripts/
  setup_wsl.sh                     一次性安装
  check_wsl.py                     WSL/CUDA 预检
  run_wsl.sh                       启动服务
  smoke_qwen.py                    真实模型冒烟测试
  verify_from_windows.ps1          Windows 到 WSL 的整体验证
tests_wsl/                         无需模型下载的自动测试
docs/                              WSL、架构和调试说明
assets/ lessons/ reference/        教学材料
jlens_qwen/                        原 MLX 实现，作为对照阅读
```

## 学习闭环

1. 先预测某段代码将产生什么张量和输出。
2. 在调试器中观察实际形状、数值范围和 Top-K。
3. 只改变一个变量，例如层、提示词、精度或干预强度。
4. 保存基线与干预结果。
5. 判断证据属于观察、相关性、因果干预，还是跨样本稳健结论。
6. 能独立解释后新增 learning record。

## 第一轮建议断点

- `jlens_wsl/runtime.py::_forward_capture`
- `jlens_wsl/lens.py::readout_last_position`
- `jlens_wsl/runtime.py::_apply_intervention`
- `jlens_wsl/runtime.py::generate`

每个断点先检查 `[batch, sequence, hidden]` 三个维度。Qwen3.5-0.8B 的隐藏维度应为 1024，层数应为 24；若实际值不同，应先停止实验并检查模型版本与内部 API。
