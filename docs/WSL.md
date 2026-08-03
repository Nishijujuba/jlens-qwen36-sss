# WSL 与 CUDA 运行说明

## 驱动边界

Windows 11 主机上的 NVIDIA 驱动通过 WSL 映射 CUDA 驱动接口。WSL 中的 PyTorch wheel 自带所需 CUDA 用户态库。这个项目无需安装完整 CUDA Toolkit，也不需要在 WSL 内安装 NVIDIA Linux 显示驱动。

只有在编译 `causal-conv1d`、Flash Linear Attention 或自定义 CUDA 内核时，才需要 toolkit 编译器。此时应安装 `cuda-toolkit-<version>`，避免 `cuda`、`cuda-drivers` 等会尝试安装 Linux 驱动的元包。

## 预检

```bash
nvidia-smi
source .venv/bin/activate
python scripts/check_wsl.py --require-wsl --require-cuda
```

应看到：

- `wsl: true`
- `cuda_available: true`
- GPU 为 RTX 3080 Laptop GPU
- 显存接近 16 GiB
- Compute Capability 为 8.6

## 常见问题

### `torch.cuda.is_available()` 为 false

按顺序检查：

```powershell
wsl --update
wsl --shutdown
```

重新进入 WSL，再执行 `nvidia-smi`。如果 WSL 中的 `nvidia-smi` 也失败，应先修复 Windows NVIDIA 驱动或 WSL 内核，继续重装 Python 包通常无效。

### 模型下载慢或失败

Hugging Face 缓存位于：

```bash
~/.cache/huggingface/hub
```

可设置：

```bash
export HF_HOME="$HOME/.cache/huggingface"
export HF_TOKEN="..."   # 仅在模型或网络策略要求时
```

中断后的分片通常可继续下载。

### 运行很慢

先确认服务日志显示 `device=cuda` 和 `dtype=float16`。其次缩短提示词、减少显示层和生成长度。默认完整重放路径优先可解释性，速度低于带缓存的普通推理服务。

可选快速 GDN 内核：

```bash
INSTALL_FAST_KERNELS=1 bash scripts/setup_wsl.sh
```

这些包含本地编译与 GPU 架构依赖，安装失败不影响默认 PyTorch 参考实现的正确性。

### 浏览器无法打开

服务监听 `0.0.0.0:8765`，Windows 通常可通过 `http://localhost:8765` 访问。检查：

```bash
curl http://127.0.0.1:8765/api/health
ss -ltnp | grep 8765
```
