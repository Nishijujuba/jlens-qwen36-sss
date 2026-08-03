#!/usr/bin/env python3
"""Preflight check for the target Windows 11 + WSL environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def is_wsl() -> bool:
    markers = " ".join(
        [
            platform.release(),
            platform.version(),
            Path("/proc/version").read_text(errors="ignore") if Path("/proc/version").exists() else "",
        ]
    ).lower()
    return "microsoft" in markers or "wsl" in markers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-wsl", action="store_true")
    args = parser.parse_args()

    import torch

    wsl = is_wsl()
    cuda = torch.cuda.is_available()
    report: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wsl": wsl,
        "repo_on_windows_mount": (len(Path.cwd().parts) >= 3 and Path.cwd().parts[1] == "mnt" and len(Path.cwd().parts[2]) == 1 and Path.cwd().parts[2].isalpha()),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "nvidia_smi_found": shutil.which("nvidia-smi") is not None,
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "free_disk_gib": round(shutil.disk_usage(Path.cwd()).free / 1024**3, 2),
    }
    if cuda:
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "vram_gib": round(props.total_memory / 1024**3, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "torch_cuda": torch.version.cuda,
        }

    problems: list[str] = []
    warnings: list[str] = []
    if args.require_wsl and not wsl:
        problems.append("当前 shell 没有检测到 WSL 标记")
    if args.require_cuda and not cuda:
        problems.append("PyTorch 无法访问 CUDA")
    if report["free_disk_gib"] < 8:
        problems.append("可用磁盘不足 8 GiB；模型缓存和 Python 环境可能无法完整安装")
    if report["repo_on_windows_mount"]:
        warnings.append("仓库位于 /mnt/c 等 Windows 挂载盘；建议放到 ~/src 以减少大量小文件 I/O 延迟")
    if cuda and report["gpu"]["vram_gib"] < 6:
        warnings.append("显存低于 6 GiB；需减少层数、位置数和生成长度")
    if not cuda:
        warnings.append("CPU 模式可以做正确性检查，交互速度会明显下降")

    report["warnings"] = warnings
    report["problems"] = problems
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if problems:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
