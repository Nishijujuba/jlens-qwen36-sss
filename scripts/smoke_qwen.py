#!/usr/bin/env python3
"""Load the official checkpoint and execute one real forward/readout step."""

from __future__ import annotations

import argparse
import json

from jlens_wsl.config import DEFAULT_MODEL_ID, RuntimeSettings
from jlens_wsl.runtime import Qwen35Runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--layers", default="0,23")
    args = parser.parse_args()
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    settings = RuntimeSettings(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        eager_load=False,
        lens_path=None,
        max_input_tokens=64,
        max_new_tokens=max(1, args.max_new_tokens),
        readout_position_chunk=2,
    )
    runtime = Qwen35Runtime(settings)
    runtime.ensure_loaded()
    analysis = runtime.analyze(
        "The capital of France is",
        layers=layers,
        top_k=3,
        max_positions=1,
    )
    events = list(
        runtime.generate(
            "The capital of France is",
            layers=layers,
            top_k=3,
            max_new_tokens=args.max_new_tokens,
            temperature=0,
        )
    )
    print(
        json.dumps(
            {
                "model": runtime.model_info(load=False),
                "analysis": analysis,
                "generation_end": events[-1],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
