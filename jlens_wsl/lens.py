"""Layer readout and optional Jacobian transport.

A logit lens applies the model's final normalization and unembedding directly
to an intermediate residual vector.  A Jacobian lens first transports that
vector with a fitted per-layer matrix ``J_l``.  The latter is optional in this
WSL port because the repository does not ship a fitted 0.8B lens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor, nn


class TokenDecoder(Protocol):
    def decode(self, token_ids, **kwargs) -> str: ...


@dataclass(frozen=True, slots=True)
class TokenScore:
    token_id: int
    token: str
    score: float


@dataclass(frozen=True, slots=True)
class LayerReadout:
    layer: int
    top: tuple[TokenScore, ...]


class TransportLens:
    """Optional full-rank transport matrices using the original NPZ schema.

    The file must contain keys named ``J_0``, ``J_1``, ... and may contain
    scalar metadata keys ``n_prompts`` and ``d_model``.  With no file, the
    transport is the identity and the runtime is explicitly in logit-lens
    mode.
    """

    def __init__(
        self,
        jacobians: dict[int, np.ndarray] | None = None,
        *,
        n_prompts: int | None = None,
        d_model: int | None = None,
        source: str | None = None,
    ) -> None:
        self._jacobians = jacobians or {}
        self.n_prompts = n_prompts
        self.d_model = d_model
        self.source = source
        self._device_cache: dict[tuple[int, str, torch.dtype], Tensor] = {}

    @property
    def mode(self) -> str:
        return "jacobian" if self._jacobians else "logit"

    @property
    def source_layers(self) -> list[int]:
        return sorted(self._jacobians)

    @classmethod
    def identity(cls) -> "TransportLens":
        return cls()

    @classmethod
    def load(cls, path: str | Path, *, expected_d_model: int | None = None) -> "TransportLens":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"lens file does not exist: {path}")
        with np.load(path, allow_pickle=False) as data:
            jacobians: dict[int, np.ndarray] = {}
            for key in data.files:
                if key.startswith("J_"):
                    layer = int(key[2:])
                    matrix = np.asarray(data[key])
                    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                        raise ValueError(f"{key} must be a square matrix, got {matrix.shape}")
                    jacobians[layer] = matrix.astype(np.float32, copy=False)
            if not jacobians:
                raise ValueError(f"no J_<layer> matrices found in {path}")
            inferred_d = next(iter(jacobians.values())).shape[0]
            for layer, matrix in jacobians.items():
                if matrix.shape != (inferred_d, inferred_d):
                    raise ValueError(
                        f"J_{layer} has shape {matrix.shape}; expected {(inferred_d, inferred_d)}"
                    )
            meta_d = int(data["d_model"]) if "d_model" in data.files else inferred_d
            if meta_d != inferred_d:
                raise ValueError(f"lens metadata d_model={meta_d}, matrix width={inferred_d}")
            if expected_d_model is not None and meta_d != expected_d_model:
                raise ValueError(
                    f"lens d_model={meta_d} does not match model d_model={expected_d_model}"
                )
            n_prompts = int(data["n_prompts"]) if "n_prompts" in data.files else None
        return cls(
            jacobians,
            n_prompts=n_prompts,
            d_model=meta_d,
            source=str(path),
        )

    def _matrix(self, layer: int, reference: Tensor) -> Tensor:
        if layer not in self._jacobians:
            raise KeyError(
                f"layer {layer} is absent from the fitted lens; available={self.source_layers}"
            )
        key = (layer, str(reference.device), reference.dtype)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = torch.as_tensor(
                self._jacobians[layer],
                device=reference.device,
                dtype=reference.dtype,
            )
            self._device_cache[key] = cached
        return cached

    def transport(self, hidden: Tensor, layer: int) -> Tensor:
        """Map row-vector residuals into the final-layer basis.

        The repository convention is ``J @ h`` for a column vector.  PyTorch
        activations use row vectors, so the equivalent operation is
        ``h @ J.T``.
        """
        if self.mode == "logit":
            return hidden
        matrix = self._matrix(layer, hidden)
        return hidden @ matrix.transpose(-1, -2)

    def concept_direction(self, unembedding_row: Tensor, layer: int) -> Tensor:
        """Return a residual-space direction that increases one token logit.

        For ``logit = w @ (J @ h)``, the gradient direction in ``h`` is
        ``J.T @ w``.  With row-vector notation this becomes ``w @ J``.
        The derivative of final RMSNorm is intentionally omitted; this keeps
        the intervention transparent and matches the common lens-direction
        approximation used for exploratory steering.
        """
        if self.mode == "logit":
            return unembedding_row
        matrix = self._matrix(layer, unembedding_row)
        return unembedding_row @ matrix


def _decode_token(tokenizer: TokenDecoder, token_id: int) -> str:
    kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    try:
        text = tokenizer.decode([token_id], **kwargs)
    except TypeError:
        text = tokenizer.decode([token_id])
    if text == " ":
        return "␣"
    if text == "\n":
        return "⏎"
    if text == "\t":
        return "⇥"
    if text and text.isspace():
        return text.replace(" ", "␣").replace("\n", "⏎").replace("\t", "⇥")
    return text


def _validate_layers(captured: dict[int, Tensor], layers: Sequence[int]) -> list[int]:
    ordered = sorted(set(int(layer) for layer in layers))
    missing = [layer for layer in ordered if layer not in captured]
    if missing:
        raise KeyError(f"captured activations are missing layers {missing}")
    return ordered


def readout_last_position(
    captured: dict[int, Tensor],
    *,
    layers: Sequence[int],
    final_norm: nn.Module,
    lm_head: nn.Module,
    tokenizer: TokenDecoder,
    lens: TransportLens,
    top_k: int,
) -> list[LayerReadout]:
    """Compute top-k token scores for the final position at several layers."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    ordered = _validate_layers(captured, layers)
    transported: list[Tensor] = []
    for layer in ordered:
        hidden = captured[layer][:, -1, :]
        hidden = lens.transport(hidden, layer)
        transported.append(hidden)
    stack = torch.cat(transported, dim=0)
    logits = lm_head(final_norm(stack)).float()
    values, ids = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
    result: list[LayerReadout] = []
    for row, layer in enumerate(ordered):
        top = tuple(
            TokenScore(
                token_id=int(token_id),
                token=_decode_token(tokenizer, int(token_id)),
                score=float(score),
            )
            for token_id, score in zip(ids[row].tolist(), values[row].tolist())
        )
        result.append(LayerReadout(layer=layer, top=top))
    return result


def readout_positions(
    captured: dict[int, Tensor],
    *,
    layers: Sequence[int],
    positions: Sequence[int],
    final_norm: nn.Module,
    lm_head: nn.Module,
    tokenizer: TokenDecoder,
    lens: TransportLens,
    top_k: int,
    position_chunk: int = 4,
) -> dict[int, list[LayerReadout]]:
    """Compute readouts for selected global positions with bounded memory."""
    if position_chunk < 1:
        raise ValueError("position_chunk must be >= 1")
    ordered_layers = _validate_layers(captured, layers)
    if not ordered_layers:
        return {}
    seq_len = captured[ordered_layers[0]].shape[1]
    ordered_positions = [int(position) for position in positions]
    bad = [position for position in ordered_positions if position < 0 or position >= seq_len]
    if bad:
        raise IndexError(f"positions out of range for sequence length {seq_len}: {bad}")

    output: dict[int, list[LayerReadout]] = {position: [] for position in ordered_positions}
    for start in range(0, len(ordered_positions), position_chunk):
        chunk = ordered_positions[start : start + position_chunk]
        tensors: list[Tensor] = []
        row_map: list[tuple[int, int]] = []
        for position in chunk:
            for layer in ordered_layers:
                hidden = captured[layer][:, position, :]
                tensors.append(lens.transport(hidden, layer))
                row_map.append((position, layer))
        if not tensors:
            continue
        stack = torch.cat(tensors, dim=0)
        logits = lm_head(final_norm(stack)).float()
        values, ids = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        for row, (position, layer) in enumerate(row_map):
            top = tuple(
                TokenScore(
                    token_id=int(token_id),
                    token=_decode_token(tokenizer, int(token_id)),
                    score=float(score),
                )
                for token_id, score in zip(ids[row].tolist(), values[row].tolist())
            )
            output[position].append(LayerReadout(layer=layer, top=top))
    return output


def layer_readouts_to_json(readouts: Iterable[LayerReadout]) -> list[dict]:
    return [
        {
            "layer": item.layer,
            "top": [
                {"token_id": token.token_id, "token": token.token, "score": token.score}
                for token in item.top
            ],
        }
        for item in readouts
    ]
