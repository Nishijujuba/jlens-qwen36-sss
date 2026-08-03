from __future__ import annotations

from fastapi.testclient import TestClient

from jlens_wsl.config import RuntimeSettings
from jlens_wsl.runtime import Qwen35Runtime
from jlens_wsl.server import create_app
from tests_wsl.fakes import FakeModel, FakeTokenizer


def make_client() -> TestClient:
    settings = RuntimeSettings(
        eager_load=False,
        device="cpu",
        dtype="float32",
        max_input_tokens=32,
        max_new_tokens=8,
    )
    runtime = Qwen35Runtime.from_components(FakeModel(), FakeTokenizer(), settings=settings)
    return TestClient(create_app(runtime=runtime, settings=settings))


def test_health_and_model_info():
    with make_client() as client:
        assert client.get("/api/health").json()["status"] == "ok"
        info = client.get("/api/model").json()
        assert info["n_layers"] == 3
        assert info["lens_mode"] == "logit"


def test_analyze_endpoint():
    with make_client() as client:
        response = client.post(
            "/api/analyze",
            json={"prompt": "ab", "layers": [0, 2], "top_k": 2, "max_positions": 2},
        )
        assert response.status_code == 200
        assert response.json()["layers"] == [0, 2]


def test_stream_endpoint_emits_sse():
    with make_client() as client:
        response = client.post(
            "/api/generate_stream",
            json={"prompt": "ab", "layers": [0, 2], "top_k": 2, "max_new_tokens": 1},
        )
        assert response.status_code == 200
        assert "event: start" in response.text
        assert "event: token" in response.text
        assert "event: end" in response.text
