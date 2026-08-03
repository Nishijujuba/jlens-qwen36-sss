.PHONY: setup-wsl check test smoke run
setup-wsl:
	bash scripts/setup_wsl.sh
check:
	. .venv/bin/activate && python scripts/check_wsl.py --require-wsl --require-cuda
test:
	. .venv/bin/activate && python -m pytest tests_wsl
smoke:
	. .venv/bin/activate && python scripts/smoke_qwen.py --device cuda --dtype fp16
run:
	bash scripts/run_wsl.sh
