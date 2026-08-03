# 选择 0.8B PyTorch 教学路径

已确定 Windows 11 + WSL + RTX 3080 Laptop 16GB 的第一阶段实验使用官方 Qwen3.5-0.8B-Base、PyTorch forward hook 和默认 Logit Lens。原 27B MLX/Metal 实现保留作架构对照；真实 Jacobian lens 只有在获得与 1024 维、24 层完全匹配的拟合矩阵后才启用，避免把临时线性探针误称为 Jacobian lens。
