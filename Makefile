.PHONY: setup dev test lint typecheck

setup:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e .

dev:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[dev]"
	python3 -m pip install -r requirements-dev.txt

test:
	python3 -m pytest

lint:
	python3 -m flake8 .

typecheck:
	python3 -m mypy src/iints
