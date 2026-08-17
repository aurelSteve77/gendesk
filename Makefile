.DEFAULT_GOAL := help
PY := .venv/bin/python
UV := VIRTUAL_ENV=.venv uv

.PHONY: help setup lint fmt type test test-fast cov data features corpus train rl backtest ablations report app pipeline clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install the package with dev extras
	uv venv --python 3.12 .venv
	$(UV) pip install torch --index-url https://download.pytorch.org/whl/cu124
	$(UV) pip install -e ".[dev]"
	$(PY) -m pre_commit install || true

lint: ## Ruff lint
	.venv/bin/ruff check src tests app

fmt: ## Ruff format + autofix
	.venv/bin/ruff format src tests app
	.venv/bin/ruff check --fix src tests app

type: ## mypy
	.venv/bin/mypy

test: ## Full test suite
	.venv/bin/pytest

test-fast: ## Test suite without slow/network tests
	.venv/bin/pytest -m "not slow and not network"

cov: ## Test suite with coverage report
	.venv/bin/pytest --cov=src/gendesk --cov-report=term-missing

data: ## Download and cache the market data panel
	$(PY) -m gendesk.cli data build

features: ## Build point-in-time features and regime labels
	$(PY) -m gendesk.cli features build

corpus: ## Build the tokenized page corpus
	$(PY) -m gendesk.cli corpus build

train: ## Stage 1 pretraining + Stage 2 WBC post-training
	$(PY) -m gendesk.cli train pretrain
	$(PY) -m gendesk.cli train wbc

rl: ## Stage 3 Dr. GRPO page-level reinforcement learning
	$(PY) -m gendesk.cli train rl

backtest: ## Walk-forward out-of-sample backtest against all baselines
	$(PY) -m gendesk.cli eval backtest

ablations: ## Run the ablation grid
	$(PY) -m gendesk.cli eval ablations

report: ## Render the results report from run artifacts
	$(PY) -m gendesk.cli eval report

pipeline: data features corpus train rl backtest ablations report ## End-to-end reproduction

app: ## Launch the Streamlit research desk
	.venv/bin/streamlit run app/main.py

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
