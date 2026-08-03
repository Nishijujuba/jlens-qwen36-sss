"""FastAPI application for the WSL Qwen3.5 interpretability workspace."""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import RuntimeSettings
from .runtime import Intervention, Qwen35Runtime


class AnalyzeRequest(BaseModel):
    prompt: str = Field(min_length=1)
    mode: Literal["completion", "chat"] = "completion"
    layers: list[int] | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    max_positions: int = Field(default=16, ge=1, le=64)


class InterventionRequest(BaseModel):
    mode: Literal["add", "remove", "replace"]
    layer: int = Field(ge=0)
    token: str = Field(min_length=1)
    alpha: float = Field(default=1.0, ge=-20, le=20)
    source: str | None = None

    def to_runtime(self) -> Intervention:
        return Intervention(
            mode=self.mode,
            layer=self.layer,
            token=self.token,
            alpha=self.alpha,
            source=self.source,
        )


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    mode: Literal["completion", "chat"] = "completion"
    layers: list[int] | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    max_new_tokens: int = Field(default=32, ge=1, le=128)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: int = 0
    intervention: InterventionRequest | None = None


class TokenizeRequest(BaseModel):
    text: str


def _sse(event: str, data: dict) -> bytes:
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n".encode("utf-8")


def create_app(
    *,
    runtime: Qwen35Runtime | None = None,
    settings: RuntimeSettings | None = None,
) -> FastAPI:
    settings = settings or (runtime.settings if runtime is not None else RuntimeSettings.from_env())
    runtime = runtime or Qwen35Runtime(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if settings.eager_load and not runtime.loaded:
            await asyncio.to_thread(runtime.ensure_loaded)
        yield

    app = FastAPI(
        title="J-lens Qwen3.5 WSL",
        version="0.3.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        path = Path(__file__).with_name("static") / "index.html"
        return HTMLResponse(path.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "loaded": runtime.loaded,
            "model_id": settings.model_id,
            "eager_load": settings.eager_load,
        }

    @app.post("/api/load")
    async def load_model() -> dict:
        try:
            await asyncio.to_thread(runtime.ensure_loaded)
            return runtime.model_info(load=False)
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.get("/api/model")
    async def model_info() -> dict:
        try:
            return await asyncio.to_thread(runtime.model_info)
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.post("/api/tokenize")
    async def tokenize(req: TokenizeRequest) -> dict:
        try:
            return await asyncio.to_thread(runtime.inspect_tokenization, req.text)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/analyze")
    async def analyze(req: AnalyzeRequest) -> dict:
        try:
            return await asyncio.to_thread(
                runtime.analyze,
                req.prompt,
                mode=req.mode,
                layers=req.layers,
                top_k=req.top_k,
                max_positions=req.max_positions,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.post("/api/generate_stream")
    async def generate_stream(req: GenerateRequest) -> StreamingResponse:
        # Validation that depends on runtime limits occurs before headers are
        # emitted by advancing the generator once.
        try:
            iterator = runtime.generate(
                req.prompt,
                mode=req.mode,
                layers=req.layers,
                top_k=req.top_k,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                seed=req.seed,
                intervention=(req.intervention.to_runtime() if req.intervention else None),
            )
            first = await asyncio.to_thread(next, iterator)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

        def stream():
            yield _sse(first.get("event", "message"), first)
            try:
                for item in iterator:
                    yield _sse(item.get("event", "message"), item)
            except Exception as exc:  # headers are already sent; surface as SSE
                yield _sse("error", {"event": "error", "error": str(exc)})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()
