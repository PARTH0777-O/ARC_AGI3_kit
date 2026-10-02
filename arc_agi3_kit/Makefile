PYTHON ?= ..\.venv\Scripts\python.exe
GAME ?= ls20
AGENT ?= explorer
MAX_ACTIONS ?= 500

.PHONY: help play-local test list-games run run-offline

help:
	@echo "ARC-AGI-3 Makefile Commands:"
	@echo "  make play-local     - Launch interactive local/terminal player"
	@echo "  make play-offline   - Launch offline mock simulator (no network required)"
	@echo "  make run            - Run agent on a game (default: GAME=ls20 AGENT=explorer)"
	@echo "  make list-games     - List games on ARC API"
	@echo "  make test           - Run offline unit tests"

play-local:
	$(PYTHON) scripts/play_local.py --game $(GAME)

play-offline:
	$(PYTHON) scripts/play_local.py --game $(GAME) --offline

run:
	$(PYTHON) scripts/run.py --game $(GAME) --agent $(AGENT) --max-actions $(MAX_ACTIONS) -v

list-games:
	$(PYTHON) scripts/run.py --list-games

test:
	$(PYTHON) tests/test_world_model.py
