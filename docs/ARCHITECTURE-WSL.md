# WSL 版本架构

## 一次生成步骤

```text
提示词
  ↓ tokenizer
input_ids [1, T]
  ↓ Qwen3.5 24 个解码层
每层 residual [1, T, 1024] ── forward hook 捕获/可选修改
  ↓ 最终 RMSNorm
  ↓ lm_head [1024 → 248320]
下一个词元 logits
```

同一次前向中，捕获的每层最后位置向量被堆叠为 `[layers, 1024]`，批量经过最终归一化和输出头。这样只调用一次大词表矩阵乘法，避免 Python 循环逐层执行 24 次输出头。

## 代码路径

1. `jlens_wsl/server.py` 校验 HTTP 请求并创建 SSE 流。
2. `jlens_wsl/runtime.py::tokenize()` 构造补全或聊天模板输入。
3. `runtime.py::_forward_capture()` 给指定层注册临时 forward hook。
4. Qwen 完整前向；hook 捕获层输出，必要时改写最后位置。
5. `jlens_wsl/lens.py` 批量执行 transport、最终归一化、unembedding 与 Top-K。
6. `runtime.py::generate()` 采样一个词元，将其追加到输入并进入下一轮完整重放。

## 张量维度

| 名称 | 典型形状 | 含义 |
|---|---|---|
| `input_ids` | `[1, T]` | 词元编号 |
| 层残差 `h_l` | `[1, T, 1024]` | 第 l 层的连续表示 |
| 选层最后位置 | `[24, 1024]` | 24 层各一条向量 |
| Logit Lens logits | `[24, 248320]` | 各层对完整词表的原始分数 |
| Top-K ids | `[24, K]` | 每层最高分词元 |
| Jacobian `J_l` | `[1024, 1024]` | 中间层到最终层坐标的局部线性映射 |

## 干预

`add/remove` 使用输出词向量或 `J_l^T W_U[token]` 作为方向，并用层残差 RMS 对强度归一化：

```text
h' = h ± α · RMS(h) · unit(v)
```

`replace` 测量当前向量在来源方向上的投影，然后将该投影从来源方向转向目标方向。它是低维、可审计的编辑，仍可能伴随副作用，因为自然语言概念在模型中通常是分布式表示。

## 被保留的旧实现

`jlens_qwen/` 的 MLX 代码含 Qwen3.6-27B 专用 Gated DeltaNet Metal 反向内核和完整 Jacobian 拟合优化。WSL 版本没有机械翻译 Metal 内核；该路径会产生错误的后端假设。旧代码保留用于比较“模型无关思想”和“硬件/架构专用优化”。
