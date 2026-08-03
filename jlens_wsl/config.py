"""Runtime configuration for the Windows 11 + WSL port.

The defaults deliberately favor a debuggable experiment over maximum serving
throughput.  A full forward pass is replayed for every generated token so that
layer hooks and interventions have simple, deterministic semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, on/off")


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Settings shared by the HTTP server and command-line smoke test."""

    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    dtype: str = "auto"
    attention_backend: str = "sdpa"
    lens_path: str | None = None
    eager_load: bool = True
    text_only: bool = True
    max_input_tokens: int = 512
    max_new_tokens: int = 64
    readout_position_chunk: int = 4

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        lens_raw = os.environ.get("JLENS_PATH", "none").strip()
        lens_path = None if lens_raw.lower() in {"", "none", "logit"} else lens_raw
        return cls(
            model_id=os.environ.get("JLENS_MODEL_ID", DEFAULT_MODEL_ID).strip(),
            device=os.environ.get("JLENS_DEVICE", "auto").strip().lower(),
            dtype=os.environ.get("JLENS_DTYPE", "auto").strip().lower(),
            attention_backend=os.environ.get("JLENS_ATTN", "sdpa").strip().lower(),
            lens_path=lens_path,
            eager_load=_env_bool("JLENS_EAGER_LOAD", True),
            text_only=_env_bool("JLENS_TEXT_ONLY", True),
            max_input_tokens=_env_int("JLENS_MAX_INPUT_TOKENS", 512),
            max_new_tokens=_env_int("JLENS_MAX_NEW_TOKENS", 64),
            readout_position_chunk=_env_int("JLENS_READOUT_POSITION_CHUNK", 4),
        )

    def resolved_lens_path(self, repo_root: Path | None = None) -> Path | None:
        if self.lens_path is None:
            return None
        path = Path(self.lens_path).expanduser()
        if not path.is_absolute() and repo_root is not None:
            path = repo_root / path
        return path.resolve()
