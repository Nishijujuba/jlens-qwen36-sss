"""Command-line entry point."""

from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B J-lens WSL workspace")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="start the local web debugger")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--lazy", action="store_true", help="load the model on the first API call")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "serve":
        if args.lazy:
            os.environ["JLENS_EAGER_LOAD"] = "0"
        import uvicorn

        uvicorn.run(
            "jlens_wsl.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )


if __name__ == "__main__":
    main()
