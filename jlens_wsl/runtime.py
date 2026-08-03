"""PyTorch runtime for Qwen3.5-0.8B-Base on Windows 11 + WSL.

Why replay the full sequence for each generated token instead of using a KV/GDN
cache?  Cached decoding is faster, while full replay gives layer hooks a clear
meaning, makes intervention A/B runs deterministic, and avoids mutating opaque
hybrid-attention state.  The 0.8B model keeps this deliberate O(T^2) teaching
path practical for short prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from pathlib import Path
import threading
import time
from typing import Any, Iterator, Literal, Sequence

import torch
from torch import Tensor, nn

from .config import RuntimeSettings
from .lens import (
    TransportLens,
    layer_readouts_to_json,
    readout_last_position,
    readout_positions,
)


PromptMode = Literal["completion", "chat"]
InterventionMode = Literal["add", "remove", "replace"]


class RuntimeConfigurationError(RuntimeError):
    """The requested hardware or model configuration is unavailable."""


@dataclass(frozen=True, slots=True)
class Intervention:
    """One exploratory residual-stream edit applied to the current last token.

    ``alpha`` is measured in units of the layer residual RMS.  This makes a
    strength of 1 roughly comparable across layers whose raw vector scales
    differ.  ``replace`` removes the current projection on ``source`` and adds
    the same projected strength toward ``token``.
    """

    mode: InterventionMode
    layer: int
    token: str
    alpha: float = 1.0
    source: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedIntervention:
    mode: InterventionMode
    layer: int
    alpha: float
    token: str
    token_id: int
    token_ids: tuple[int, ...]
    target_direction: Tensor
    source: str | None = None
    source_id: int | None = None
    source_ids: tuple[int, ...] = ()
    source_direction: Tensor | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "layer": self.layer,
            "alpha": self.alpha,
            "token": self.token,
            "token_id": self.token_id,
            "token_ids": list(self.token_ids),
            "source": self.source,
            "source_id": self.source_id,
            "source_ids": list(self.source_ids),
            "scope": "current last token",
            "direction": "J^T W_U[token]" if self.source_direction is not None else "W_U[token] or J^T W_U[token]",
        }


def _first_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, Tensor):
                return item
    raise TypeError(f"decoder layer returned unsupported output type: {type(value)!r}")


def _replace_first_tensor(value: Any, tensor: Tensor) -> Any:
    if isinstance(value, Tensor):
        return tensor
    if isinstance(value, tuple):
        items = list(value)
        for index, item in enumerate(items):
            if isinstance(item, Tensor):
                items[index] = tensor
                return tuple(items)
    if isinstance(value, list):
        items = list(value)
        for index, item in enumerate(items):
            if isinstance(item, Tensor):
                items[index] = tensor
                return items
    raise TypeError(f"decoder layer returned unsupported output type: {type(value)!r}")


def _decode(tokenizer: Any, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str:
    kwargs = {
        "skip_special_tokens": skip_special_tokens,
        "clean_up_tokenization_spaces": False,
    }
    try:
        return tokenizer.decode(list(token_ids), **kwargs)
    except TypeError:
        return tokenizer.decode(list(token_ids))


def _token_segments(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    """Return display segments that concatenate to the cumulative decode."""
    segments: list[str] = []
    previous = ""
    for index in range(len(token_ids)):
        current = _decode(tokenizer, token_ids[: index + 1], skip_special_tokens=False)
        if current.startswith(previous):
            segments.append(current[len(previous) :])
        else:
            segments.append(_decode(tokenizer, [token_ids[index]], skip_special_tokens=False))
        previous = current
    return segments


class Qwen35Runtime:
    """Owns the model, tokenizer, layer hooks, readout and generation loop."""

    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings or RuntimeSettings.from_env()
        self._load_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._loaded = False
        self._model: nn.Module | None = None
        self._tokenizer: Any = None
        self._text_model: nn.Module | None = None
        self._layers: Sequence[nn.Module] = ()
        self._final_norm: nn.Module | None = None
        self._lm_head: nn.Module | None = None
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._lens = TransportLens.identity()
        self._layer_types: list[str] = []
        self._model_load_seconds: float | None = None

    @classmethod
    def from_components(
        cls,
        model: nn.Module,
        tokenizer: Any,
        *,
        settings: RuntimeSettings | None = None,
        lens: TransportLens | None = None,
    ) -> "Qwen35Runtime":
        """Construct a runtime from an in-memory Qwen-shaped model for tests."""
        runtime = cls(settings or RuntimeSettings(eager_load=False, device="cpu", dtype="float32"))
        runtime._bind_components(model, tokenizer, lens=lens)
        runtime._loaded = True
        runtime._model_load_seconds = 0.0
        return runtime

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    @property
    def d_model(self) -> int:
        if self._final_norm is None:
            return 0
        weight = getattr(self._final_norm, "weight", None)
        if isinstance(weight, Tensor):
            return int(weight.shape[0])
        if self._lm_head is not None and hasattr(self._lm_head, "weight"):
            return int(self._lm_head.weight.shape[1])
        return 0

    def _resolve_device(self) -> torch.device:
        requested = self.settings.device
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeConfigurationError(
                "JLENS_DEVICE requests CUDA, but torch.cuda.is_available() is false. "
                "Run scripts/check_wsl.py and verify that WSL can see nvidia-smi."
            )
        return device

    def _resolve_dtype(self, device: torch.device) -> torch.dtype:
        requested = self.settings.dtype
        choices = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if requested == "auto":
            # FP16 is the conservative default for an RTX 3080 Laptop GPU.
            # BF16 can be selected explicitly after check_wsl.py confirms it.
            return torch.float16 if device.type == "cuda" else torch.float32
        if requested not in choices:
            raise RuntimeConfigurationError(
                f"unsupported JLENS_DTYPE={requested!r}; use auto, fp16, bf16 or fp32"
            )
        dtype = choices[requested]
        if device.type == "cpu" and dtype == torch.float16:
            raise RuntimeConfigurationError(
                "FP16 CPU execution is poorly supported; use JLENS_DTYPE=fp32 on CPU"
            )
        return dtype

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            started = time.perf_counter()
            device = self._resolve_device()
            dtype = self._resolve_dtype(device)
            try:
                from transformers import AutoModelForMultimodalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeConfigurationError(
                    "Transformers is missing. Run `bash scripts/setup_wsl.sh` inside WSL."
                ) from exc

            load_kwargs: dict[str, Any] = {
                "dtype": dtype,
                "low_cpu_mem_usage": True,
            }
            if self.settings.attention_backend:
                load_kwargs["attn_implementation"] = self.settings.attention_backend

            tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id)
            model = AutoModelForMultimodalLM.from_pretrained(
                self.settings.model_id,
                **load_kwargs,
            )
            model.to(device)
            model.eval()

            # This workspace studies the language residual stream.  Dropping
            # the unused vision tower releases VRAM after the official
            # multimodal checkpoint has loaded.  Text-only forward calls do
            # not touch Qwen3_5Model.visual when pixel_values is absent.
            if self.settings.text_only:
                base = getattr(model, "model", None)
                if base is not None and hasattr(base, "visual"):
                    base.visual = None
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            self._device = device
            self._dtype = dtype
            self._bind_components(model, tokenizer)
            self._model_load_seconds = time.perf_counter() - started
            self._loaded = True

    def _bind_components(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        lens: TransportLens | None = None,
    ) -> None:
        base = getattr(model, "model", None)
        if base is not None and hasattr(base, "language_model"):
            text_model = base.language_model
        elif base is not None and hasattr(base, "layers"):
            text_model = base
        elif hasattr(model, "language_model"):
            text_model = model.language_model
        else:
            raise RuntimeConfigurationError(
                "could not locate the Qwen text backbone; expected model.model.language_model"
            )
        layers = getattr(text_model, "layers", None)
        final_norm = getattr(text_model, "norm", None)
        lm_head = getattr(model, "lm_head", None)
        if layers is None or final_norm is None or lm_head is None:
            raise RuntimeConfigurationError(
                "model is missing layers, final norm, or lm_head required for layer readout"
            )

        self._model = model
        self._tokenizer = tokenizer
        self._text_model = text_model
        self._layers = list(layers)
        self._final_norm = final_norm
        self._lm_head = lm_head

        try:
            parameter = next(model.parameters())
            self._device = parameter.device
            self._dtype = parameter.dtype
        except StopIteration:
            pass

        if getattr(tokenizer, "pad_token_id", None) is None:
            eos = getattr(tokenizer, "eos_token_id", None)
            if isinstance(eos, int):
                tokenizer.pad_token_id = eos

        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", config)
        layer_types = getattr(text_config, "layer_types", None)
        if isinstance(layer_types, (list, tuple)) and len(layer_types) == len(self._layers):
            self._layer_types = [str(value) for value in layer_types]
        else:
            self._layer_types = ["decoder" for _ in self._layers]

        if lens is not None:
            self._lens = lens
        else:
            lens_path = self.settings.resolved_lens_path(Path.cwd())
            self._lens = (
                TransportLens.load(lens_path, expected_d_model=self.d_model)
                if lens_path is not None
                else TransportLens.identity()
            )

    def _require_components(self) -> tuple[nn.Module, Any, nn.Module, nn.Module]:
        self.ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._final_norm is not None
        assert self._lm_head is not None
        return self._model, self._tokenizer, self._final_norm, self._lm_head

    def _normalise_layers(self, layers: Sequence[int] | None) -> list[int]:
        selected = list(range(self.n_layers)) if layers is None else sorted(set(int(x) for x in layers))
        bad = [layer for layer in selected if layer < 0 or layer >= self.n_layers]
        if bad:
            raise ValueError(f"layers out of range 0..{self.n_layers - 1}: {bad}")
        if self._lens.mode == "jacobian":
            missing = [layer for layer in selected if layer not in self._lens.source_layers]
            if missing:
                raise ValueError(
                    f"fitted lens has no matrices for layers {missing}; available={self._lens.source_layers}"
                )
        return selected

    def _normalise_encoded(self, encoded: Any) -> dict[str, Tensor]:
        if isinstance(encoded, Tensor):
            data: dict[str, Tensor] = {"input_ids": encoded}
        elif isinstance(encoded, dict) or hasattr(encoded, "items"):
            data = {key: value for key, value in encoded.items() if isinstance(value, Tensor)}
        else:
            raise TypeError(f"tokenizer returned unsupported type: {type(encoded)!r}")
        if "input_ids" not in data:
            raise ValueError("tokenizer result does not contain input_ids")
        if data["input_ids"].ndim == 1:
            data["input_ids"] = data["input_ids"].unsqueeze(0)
        if "attention_mask" not in data:
            data["attention_mask"] = torch.ones_like(data["input_ids"])
        return {key: value.to(self._device) for key, value in data.items()}

    def tokenize(self, prompt: str, *, mode: PromptMode = "completion") -> dict[str, Tensor]:
        _, tokenizer, _, _ = self._require_components()
        if not prompt:
            raise ValueError("prompt is empty")
        if mode == "completion":
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        elif mode == "chat":
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
            }
            try:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **kwargs,
                )
            except TypeError:
                encoded = tokenizer.apply_chat_template(messages, **kwargs)
        else:
            raise ValueError(f"unsupported prompt mode: {mode!r}")
        data = self._normalise_encoded(encoded)
        length = int(data["input_ids"].shape[1])
        if length > self.settings.max_input_tokens:
            raise ValueError(
                f"prompt has {length} tokens; configured limit is {self.settings.max_input_tokens}. "
                "The runtime refuses silent truncation because it would move every layer/position coordinate."
            )
        return data

    def inspect_tokenization(self, text: str) -> dict[str, Any]:
        _, tokenizer, _, _ = self._require_components()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        ids = [int(token_id) for token_id in token_ids]
        segments = _token_segments(tokenizer, ids)
        return {
            "text": text,
            "tokens": [
                {"token_id": token_id, "text": segments[index]}
                for index, token_id in enumerate(ids)
            ],
        }

    def _token_ids_for_concept(self, text: str) -> tuple[int, ...]:
        if not text:
            raise ValueError("intervention concept is empty")
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        ids = tuple(int(token_id) for token_id in token_ids)
        if not ids:
            raise ValueError(f"concept {text!r} encodes to no tokens")
        return ids

    def _prepare_intervention(self, spec: Intervention | None) -> PreparedIntervention | None:
        if spec is None:
            return None
        if spec.layer < 0 or spec.layer >= self.n_layers:
            raise ValueError(f"intervention layer must be in 0..{self.n_layers - 1}")
        if not math.isfinite(spec.alpha) or abs(spec.alpha) > 20:
            raise ValueError("intervention alpha must be finite and within [-20, 20]")
        target_ids = self._token_ids_for_concept(spec.token)
        assert self._lm_head is not None
        target_row = self._lm_head.weight[target_ids[0]].detach()
        target_direction = self._lens.concept_direction(target_row, spec.layer)

        source_ids: tuple[int, ...] = ()
        source_id: int | None = None
        source_direction: Tensor | None = None
        if spec.mode == "replace":
            if not spec.source:
                raise ValueError("replace intervention requires source text")
            source_ids = self._token_ids_for_concept(spec.source)
            source_id = source_ids[0]
            source_row = self._lm_head.weight[source_id].detach()
            source_direction = self._lens.concept_direction(source_row, spec.layer)
        elif spec.mode not in {"add", "remove"}:
            raise ValueError(f"unsupported intervention mode: {spec.mode!r}")

        return PreparedIntervention(
            mode=spec.mode,
            layer=spec.layer,
            alpha=float(spec.alpha),
            token=spec.token,
            token_id=target_ids[0],
            token_ids=target_ids,
            target_direction=target_direction,
            source=spec.source,
            source_id=source_id,
            source_ids=source_ids,
            source_direction=source_direction,
        )

    @staticmethod
    def _unit(vector: Tensor, reference: Tensor) -> Tensor:
        vector = vector.to(device=reference.device, dtype=torch.float32)
        return vector / vector.norm().clamp_min(1e-6)

    def _apply_intervention(self, hidden: Tensor, spec: PreparedIntervention) -> Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"expected layer hidden state [batch, sequence, d], got {hidden.shape}")
        output = hidden.clone()
        row = output[:, -1, :]
        row32 = row.float()
        residual_rms = row32.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        target = self._unit(spec.target_direction, row32)
        if spec.mode == "add":
            edited = row32 + spec.alpha * residual_rms * target
        elif spec.mode == "remove":
            edited = row32 - spec.alpha * residual_rms * target
        else:
            assert spec.source_direction is not None
            source = self._unit(spec.source_direction, row32)
            source_loading = (row32 * source).sum(dim=-1, keepdim=True)
            edited = row32 + spec.alpha * source_loading * (target - source)
        output[:, -1, :] = edited.to(dtype=output.dtype)
        return output

    def _forward_capture(
        self,
        encoded: dict[str, Tensor],
        *,
        layers: Sequence[int],
        intervention: PreparedIntervention | None = None,
    ) -> tuple[Tensor, dict[int, Tensor]]:
        model, _, _, _ = self._require_components()
        requested = sorted(set(layers) | ({intervention.layer} if intervention else set()))
        captured: dict[int, Tensor] = {}
        handles: list[Any] = []

        for layer_index in requested:
            def hook(_module, _args, output, *, index=layer_index):
                tensor = _first_tensor(output)
                if intervention is not None and index == intervention.layer:
                    tensor = self._apply_intervention(tensor, intervention)
                    output = _replace_first_tensor(output, tensor)
                captured[index] = tensor.detach()
                return output

            handles.append(self._layers[layer_index].register_forward_hook(hook))

        try:
            with torch.no_grad():
                outputs = model(
                    **encoded,
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            logits = getattr(outputs, "logits", None)
            if logits is None:
                logits = outputs[0]
            if not isinstance(logits, Tensor):
                raise TypeError("model forward did not return tensor logits")
        finally:
            for handle in handles:
                handle.remove()

        missing = [layer for layer in requested if layer not in captured]
        if missing:
            raise RuntimeError(f"layer hooks did not capture {missing}")
        return logits, captured

    def analyze(
        self,
        prompt: str,
        *,
        mode: PromptMode = "completion",
        layers: Sequence[int] | None = None,
        top_k: int = 5,
        max_positions: int = 16,
    ) -> dict[str, Any]:
        if max_positions < 1 or max_positions > 64:
            raise ValueError("max_positions must be in [1, 64]")
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be in [1, 20]")
        with self._request_lock:
            encoded = self.tokenize(prompt, mode=mode)
            selected = self._normalise_layers(layers)
            started = time.perf_counter()
            _, captured = self._forward_capture(encoded, layers=selected)
            token_ids = [int(token_id) for token_id in encoded["input_ids"][0].tolist()]
            start = max(0, len(token_ids) - max_positions)
            positions = list(range(start, len(token_ids)))
            assert self._final_norm is not None and self._lm_head is not None
            cells = readout_positions(
                captured,
                layers=selected,
                positions=positions,
                final_norm=self._final_norm,
                lm_head=self._lm_head,
                tokenizer=self._tokenizer,
                lens=self._lens,
                top_k=top_k,
                position_chunk=self.settings.readout_position_chunk,
            )
            segments = _token_segments(self._tokenizer, token_ids)
            elapsed = time.perf_counter() - started
            return {
                "prompt_mode": mode,
                "input_tokens": len(token_ids),
                "positions": [
                    {
                        "position": position,
                        "token_id": token_ids[position],
                        "token": segments[position],
                        "readouts": layer_readouts_to_json(cells[position]),
                    }
                    for position in positions
                ],
                "layers": selected,
                "layer_types": {str(layer): self._layer_types[layer] for layer in selected},
                "lens_mode": self._lens.mode,
                "elapsed_seconds": elapsed,
            }

    @staticmethod
    def _sample_token(
        logits: Tensor,
        *,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None,
    ) -> int:
        scores = logits.float()
        if temperature <= 0:
            return int(torch.argmax(scores).item())
        scores = scores / temperature
        probabilities = torch.softmax(scores, dim=-1)
        if top_p < 1.0:
            sorted_prob, sorted_ids = torch.sort(probabilities, descending=True)
            cumulative = torch.cumsum(sorted_prob, dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_prob = sorted_prob.masked_fill(remove, 0)
            sorted_prob = sorted_prob / sorted_prob.sum(dim=-1, keepdim=True)
            sampled_index = torch.multinomial(sorted_prob, num_samples=1, generator=generator)
            return int(sorted_ids.gather(-1, sampled_index).item())
        return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())

    def _eos_ids(self) -> set[int]:
        values: list[Any] = []
        if self._tokenizer is not None:
            values.append(getattr(self._tokenizer, "eos_token_id", None))
        if self._model is not None:
            generation_config = getattr(self._model, "generation_config", None)
            values.append(getattr(generation_config, "eos_token_id", None))
        output: set[int] = set()
        for value in values:
            if isinstance(value, int):
                output.add(value)
            elif isinstance(value, (list, tuple, set)):
                output.update(int(item) for item in value)
        return output

    def generate(
        self,
        prompt: str,
        *,
        mode: PromptMode = "completion",
        layers: Sequence[int] | None = None,
        top_k: int = 5,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        intervention: Intervention | None = None,
    ) -> Iterator[dict[str, Any]]:
        if max_new_tokens < 1 or max_new_tokens > self.settings.max_new_tokens:
            raise ValueError(
                f"max_new_tokens must be in [1, {self.settings.max_new_tokens}]"
            )
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be in [1, 20]")
        if temperature < 0 or temperature > 2:
            raise ValueError("temperature must be in [0, 2]")
        if top_p <= 0 or top_p > 1:
            raise ValueError("top_p must be in (0, 1]")

        with self._request_lock:
            encoded = self.tokenize(prompt, mode=mode)
            selected = self._normalise_layers(layers)
            prepared = self._prepare_intervention(intervention)
            if prepared is not None and self._lens.mode == "jacobian":
                if prepared.layer not in self._lens.source_layers:
                    raise ValueError(f"fitted lens has no matrix for intervention layer {prepared.layer}")

            generator: torch.Generator | None = None
            if temperature > 0:
                generator = torch.Generator(device=self._device.type)
                generator.manual_seed(seed)

            input_count = int(encoded["input_ids"].shape[1])
            generated_ids: list[int] = []
            previous_text = ""
            eos_ids = self._eos_ids()
            request_started = time.perf_counter()
            yield {
                "event": "start",
                "model": self.model_info(load=False),
                "prompt_mode": mode,
                "input_tokens": input_count,
                "layers": selected,
                "intervention": prepared.public_dict() if prepared else None,
                "decode_strategy": "full-sequence replay per generated token",
            }

            stop_reason = "max_new_tokens"
            for step in range(max_new_tokens):
                step_started = time.perf_counter()
                logits, captured = self._forward_capture(
                    encoded,
                    layers=selected,
                    intervention=prepared,
                )
                assert self._final_norm is not None and self._lm_head is not None
                readouts = readout_last_position(
                    captured,
                    layers=selected,
                    final_norm=self._final_norm,
                    lm_head=self._lm_head,
                    tokenizer=self._tokenizer,
                    lens=self._lens,
                    top_k=top_k,
                )
                token_id = self._sample_token(
                    logits[0, -1],
                    temperature=temperature,
                    top_p=top_p,
                    generator=generator,
                )
                generated_ids.append(token_id)
                full_text = _decode(
                    self._tokenizer,
                    generated_ids,
                    skip_special_tokens=True,
                )
                delta = full_text[len(previous_text) :] if full_text.startswith(previous_text) else full_text
                previous_text = full_text

                new_token = torch.tensor([[token_id]], device=self._device, dtype=torch.long)
                encoded["input_ids"] = torch.cat([encoded["input_ids"], new_token], dim=1)
                encoded["attention_mask"] = torch.cat(
                    [encoded["attention_mask"], torch.ones_like(new_token)],
                    dim=1,
                )
                step_seconds = time.perf_counter() - step_started
                yield {
                    "event": "token",
                    "step": step,
                    "position": input_count + step,
                    "token_id": token_id,
                    "token": delta,
                    "text": full_text,
                    "readouts": layer_readouts_to_json(readouts),
                    "step_seconds": step_seconds,
                }
                if token_id in eos_ids:
                    stop_reason = "eos"
                    break

            yield {
                "event": "end",
                "text": previous_text,
                "token_ids": generated_ids,
                "output_tokens": len(generated_ids),
                "stop_reason": stop_reason,
                "elapsed_seconds": time.perf_counter() - request_started,
            }

    def model_info(self, *, load: bool = True) -> dict[str, Any]:
        if load:
            self.ensure_loaded()
        cuda: dict[str, Any] | None = None
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            cuda = {
                "name": torch.cuda.get_device_name(index),
                "total_vram_gib": round(props.total_memory / 1024**3, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        return {
            "loaded": self._loaded,
            "model_id": self.settings.model_id,
            "device": str(self._device),
            "dtype": str(self._dtype).replace("torch.", ""),
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "layer_types": self._layer_types,
            "lens_mode": self._lens.mode,
            "lens_source": self._lens.source,
            "lens_source_layers": self._lens.source_layers,
            "lens_n_prompts": self._lens.n_prompts,
            "text_only": self.settings.text_only,
            "attention_backend": self.settings.attention_backend,
            "model_load_seconds": self._model_load_seconds,
            "torch_version": torch.__version__,
            "cuda": cuda,
            "limits": {
                "max_input_tokens": self.settings.max_input_tokens,
                "max_new_tokens": self.settings.max_new_tokens,
            },
        }
