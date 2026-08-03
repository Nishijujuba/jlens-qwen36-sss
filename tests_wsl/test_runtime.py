from __future__ import annotations

from jlens_wsl.config import RuntimeSettings
from jlens_wsl.runtime import Intervention, Qwen35Runtime
from tests_wsl.fakes import FakeModel, FakeTokenizer


def make_runtime() -> Qwen35Runtime:
    return Qwen35Runtime.from_components(
        FakeModel(),
        FakeTokenizer(),
        settings=RuntimeSettings(
            eager_load=False,
            device="cpu",
            dtype="float32",
            max_input_tokens=32,
            max_new_tokens=8,
            readout_position_chunk=2,
        ),
    )


def test_analyze_captures_requested_layers_and_positions():
    runtime = make_runtime()
    result = runtime.analyze("ab", layers=[0, 2], top_k=2, max_positions=2)
    assert result["input_tokens"] == 2
    assert result["layers"] == [0, 2]
    assert len(result["positions"]) == 2
    assert [item["layer"] for item in result["positions"][0]["readouts"]] == [0, 2]


def test_generation_emits_start_tokens_and_end():
    runtime = make_runtime()
    events = list(
        runtime.generate(
            "ab",
            layers=[0, 1, 2],
            top_k=2,
            max_new_tokens=2,
            temperature=0,
        )
    )
    assert events[0]["event"] == "start"
    assert events[-1]["event"] == "end"
    token_events = [event for event in events if event["event"] == "token"]
    assert 1 <= len(token_events) <= 2
    assert len(token_events[0]["readouts"]) == 3


def test_add_intervention_is_resolved_and_applied():
    runtime = make_runtime()
    events = list(
        runtime.generate(
            "ab",
            layers=[0, 2],
            top_k=2,
            max_new_tokens=1,
            intervention=Intervention(mode="add", layer=1, token=" c", alpha=0.5),
        )
    )
    assert events[0]["intervention"]["layer"] == 1
    assert events[0]["intervention"]["token_ids"]
