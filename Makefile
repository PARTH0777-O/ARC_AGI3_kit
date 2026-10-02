PYTHON ?= .venv\Scripts\python.exe
KAGGLE ?= .venv\Scripts\kaggle.exe
GAME ?= ls20
AGENT ?= explorer
MAX_ACTIONS ?= 500

.PHONY: help play-local play-offline test list-games run build-notebook submit status

help:
	@echo "ARC-AGI-3 Commands:"
	@echo "  make play-local     - Launch interactive local/terminal player"
	@echo "  make play-offline   - Launch offline mock simulator"
	@echo "  make build-notebook - Build notebooks/submission.ipynb"
	@echo "  make submit         - Build notebook and push to Kaggle"
	@echo "  make status         - Check Kaggle kernel run status"
	@echo "  make run            - Run agent online on a game"
	@echo "  make list-games     - List games on ARC API"
	@echo "  make test           - Run offline unit tests"

build-notebook:
	$(PYTHON) scripts/build_notebook.py

submit: build-notebook
	$(KAGGLE) kernels push -p notebooks

status:
	$(KAGGLE) kernels status your-username/arc-agi-3-agent

play-local:
	$(PYTHON) arc_agi3_kit/scripts/play_local.py --game $(GAME)

play-offline:
	$(PYTHON) arc_agi3_kit/scripts/play_local.py --game $(GAME) --offline

run:
	$(PYTHON) arc_agi3_kit/scripts/run.py --game $(GAME) --agent $(AGENT) --max-actions $(MAX_ACTIONS) -v

list-games:
	$(PYTHON) arc_agi3_kit/scripts/run.py --list-games

test:
	$(PYTHON) arc_agi3_kit/tests/test_world_model.py
