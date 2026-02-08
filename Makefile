.PHONY: help setup dev test lint typecheck clean build publish publish-test tag push-tag github-release release

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
BUILD ?= $(PYTHON) -m build
TWINE ?= $(PYTHON) -m twine
GH ?= gh
VERSION ?= $(shell $(PYTHON) -c "import re, pathlib; print(re.search(r'^version = \"(.*)\"$$', pathlib.Path('pyproject.toml').read_text(), re.M).group(1))")
TAG ?= v$(VERSION)
RELEASE_NOTES ?= docs/releases/v$(VERSION).md

help:
	@echo "Targets:"
	@echo "  setup        Install runtime deps"
	@echo "  dev          Install dev deps"
	@echo "  test         Run pytest"
	@echo "  lint         Run flake8"
	@echo "  typecheck    Run mypy"
	@echo "  clean        Remove build artifacts"
	@echo "  build        Build sdist/wheel"
	@echo "  publish      Upload to PyPI (needs TWINE creds)"
	@echo "  publish-test Upload to TestPyPI"
	@echo "  tag          Create git tag from pyproject version"
	@echo "  push-tag     Push the tag to origin"
	@echo "  github-release  Create a GitHub release (requires gh auth)"
	@echo "  release      test + build + publish + tag + push-tag"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -e .

dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m flake8 .

typecheck:
	$(PYTHON) -m mypy src/iints

clean:
	rm -rf dist build src/*.egg-info .pytest_cache

build: clean
	$(BUILD) --no-isolation

publish: build
	$(TWINE) upload dist/*

publish-test: build
	$(TWINE) upload --repository testpypi dist/*

tag:
	git tag -a $(TAG) -m "Release $(TAG)"

push-tag:
	git push origin $(TAG)

github-release:
	$(GH) release create $(TAG) -F $(RELEASE_NOTES) --title "$(TAG)"

release: test lint typecheck publish tag push-tag
	@echo "Release $(TAG) complete."
