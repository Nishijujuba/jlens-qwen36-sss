# VS Code WSL 调试练习

## 准备

1. 在 Windows 安装 VS Code 与 Remote - WSL 扩展。
2. 在 WSL 仓库目录运行 `code .`。
3. 选择 `.venv/bin/python`。
4. 在 Run and Debug 中选择 `J-lens WSL: FastAPI debugger`。

## 建议断点

### 练习 1：看一个词如何变成向量

在 `Qwen35Runtime.tokenize()` 返回前下断点，检查：

- `input_ids.shape`
- 每个整数词元编号
- `attention_mask`

随后进入 `_forward_capture()`，比较 `captured[0]` 与 `captured[23]` 的形状和数值尺度。

### 练习 2：理解 Logit Lens

在 `readout_last_position()` 中查看：

- `stack.shape == [选中层数, 1024]`
- `logits.shape == [选中层数, 248320]`
- `torch.topk` 返回的编号与分数

这里的 `248320` 是词表大小。每个分数叫 logit，是 softmax 之前的相对偏好。

### 练习 3：区分相关读出与因果干预

先关闭干预运行一次，再在 `_apply_intervention()` 下断点运行 `add`：

- 比较 `row32` 与 `edited`
- 检查 `residual_rms`
- 记录最终输出是否变化

一次变化只能说明这个编辑影响了输出。更强结论需要多提示词、对照方向、不同强度和重复实验。

## 典型启动配置

`.vscode/launch.json` 会设置：

```text
JLENS_DEVICE=cuda
JLENS_DTYPE=fp16
JLENS_PATH=none
JLENS_EAGER_LOAD=0
```

懒加载能让调试器先进入 FastAPI，再由网页的“加载模型”按钮触发模型加载。
