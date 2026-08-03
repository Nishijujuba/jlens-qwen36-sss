# J-lens 教学工作区入口

该分支把 `jlens-qwen36-sss` 转换为一个可持续学习的工作区。`main` 保留原项目；`teach` 增加任务目标、学习记录、参考页、分阶段课程和硬件决策。

## 当前结论

1. 仓库名中的 `qwen36` 指 **Qwen 3.6**。默认模型为 **Qwen3.6-27B**，参数量为 27B。
2. 原项目围绕 Apple Silicon、MLX 和自定义 Metal Gated DeltaNet 反向内核设计。
3. 当前笔记本为 Windows 11 + WSL、RTX 3080 Laptop 16 GB 专用显存、约 32 GB 系统内存。
4. 该设备适合从 `Qwen/Qwen3.5-0.8B-Base` 开始 CUDA/PyTorch 教学实验，再升级到 `Qwen/Qwen3.5-2B`。
5. Qwen3.6-27B 的完整 J-lens 路径需要匹配的平台与更宽裕的内存；本地 CPU/共享内存卸载不作为学习基线。

## 学习顺序

| 顺序 | 文件 | 本阶段成果 |
|---|---|---|
| 0 | [`MISSION.md`](MISSION.md) | 明确学习目标、完成标准和边界 |
| 1 | [`lessons/0001-from-text-to-residual-stream.html`](lessons/0001-from-text-to-residual-stream.html) | 理解词元、向量、层、残差流、logit 和 J-lens 主公式 |
| 2 | [`lessons/0002-read-the-code-path.html`](lessons/0002-read-the-code-path.html) | 沿读取、拟合、写入三条路径定位核心代码 |
| 3 | [`lessons/0003-wsl-and-model-choice.html`](lessons/0003-wsl-and-model-choice.html) | 完成 WSL/CUDA 验收并理解模型选择依据 |
| 随时 | [`reference/0001-glossary.html`](reference/0001-glossary.html) | 查阅术语、公式、架构和硬件概念 |
| 持续 | [`learning-records/`](learning-records/) | 记录已真正掌握的内容，决定下一课难度 |

## 怎样打开 HTML 课程

在本地仓库执行：

```bash
git fetch origin
git switch teach
python -m http.server 8000
```

随后在浏览器打开：

```text
http://localhost:8000/lessons/0001-from-text-to-residual-stream.html
```

直接双击 HTML 也可以阅读；本地 HTTP 服务在链接跳转和后续交互实验上更稳定。

## 当前文件结构

```text
MISSION.md                         学习任务与完成标准
RESOURCES.md                       经过筛选的知识来源与社区
NOTES.md                           学习偏好、硬件和模型决策
TEACHING-START.md                  本入口
assets/
  teaching.css                     所有课程共享的样式
  lesson.js                        所有课程共享的测验组件
reference/
  0001-glossary.html               可打印的入门术语表
lessons/
  0001-from-text-to-residual-stream.html
  0002-read-the-code-path.html
  0003-wsl-and-model-choice.html
learning-records/
  0001-starting-point-and-hardware.md
```

## 学习循环

每个主题采用同一闭环：

1. **获取知识**：阅读一节短课和对应主资料。
2. **主动回忆**：关闭页面，用自己的话复述数据流或公式。
3. **代码定位**：在仓库中找到公式对应的函数和张量形状。
4. **最小实验**：只改变一个变量，保留基线结果。
5. **证据分级**：区分观察、相关性、干预结果和可推广结论。
6. **记录掌握**：学习者能独立解释或完成任务后，新增 learning record。

## 第一个动手检查点

完成前三课后，在 Windows PowerShell 和 WSL 中执行第 3 课的环境命令，并记录：

- WSL 版本与发行版是否为 WSL 2。
- `nvidia-smi` 是否看到 RTX 3080 Laptop。
- 驱动版本与专用显存。
- PyTorch 版本、CUDA runtime 和 `torch.cuda.is_available()`。
- FP16 CUDA 矩阵乘法是否成功。

这些结果将决定下一步环境文件和实验代码采用哪个 PyTorch/CUDA 组合。未经实测，不预设完整梯度实验一定可行。

## 后续课程规划

- 第 4 课：用 0.8B 模型捕获指定层隐藏状态，逐项解释张量形状。
- 第 5 课：实现最小 logit lens，并验证最终层读出与模型 logits 一致。
- 第 6 课：用有限差分、JVP 和 VJP 建立 Jacobian 直觉。
- 第 7 课：完成一次残差 steering 和基线/干预 A/B 对照。
- 第 8 课：把教学实现与原仓库的 MLX/Metal 实现逐函数对应。

后续代码将在环境验收后添加，避免提前锁定与本机驱动或 CUDA 版本不兼容的依赖。
