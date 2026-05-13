.PHONY: help install test lint format typecheck check fix clean

help:
	@echo "Available commands:"
	@echo "  make install   - Install all dependencies (incl. dev/api/app)"
	@echo "  make test      - Run pytest with coverage"
	@echo "  make lint      - Run ruff linter (no auto-fix)"
	@echo "  make format    - Auto-format code with ruff"
	@echo "  make typecheck - Run mypy"
	@echo "  make check     - Run all checks WITHOUT modifying files (used by CI)"
	@echo "  make fix       - Auto-format AND run all checks (recommended for dev)"
	@echo "  make clean     - Remove caches and build artifacts"

install:
	uv sync --all-extras

test:
	pytest -v

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy src

check: lint typecheck test

fix: format typecheck test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
