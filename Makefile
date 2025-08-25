.PHONY: help install install-dev test lint format clean build docs run-example

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
SRC_DIR := src
TEST_DIR := tests
PACKAGE_NAME := jina_embeddings

# Default target
help:
	@echo "Jina Embeddings v4 - Development Commands"
	@echo "========================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install       Install package in production mode"
	@echo "  make install-dev   Install package in development mode with all extras"
	@echo "  make setup         Create virtual environment and install dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run all tests"
	@echo "  make test-unit     Run unit tests only"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make lint          Run code linters (flake8, mypy)"
	@echo "  make format        Format code with black and isort"
	@echo "  make check         Run all checks (lint + test)"
	@echo ""
	@echo "Building & Distribution:"
	@echo "  make build         Build distribution packages"
	@echo "  make clean         Clean build artifacts and cache"
	@echo "  make docs          Generate documentation"
	@echo ""
	@echo "Examples & Demo:"
	@echo "  make demo          Run hello world demo"
	@echo "  make example-text  Run text similarity example"
	@echo "  make example-image Run multimodal search example"
	@echo ""

# Setup virtual environment
setup:
	$(PYTHON) -m venv venv
	./venv/bin/pip install --upgrade pip setuptools wheel
	./venv/bin/pip install -e ".[all]"
	@echo "✅ Setup complete! Activate with: source venv/bin/activate"

# Installation targets
install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install .

install-dev:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[all]"
	pre-commit install

# Testing targets
test:
	pytest $(TEST_DIR) -v

test-unit:
	pytest $(TEST_DIR)/unit -v -m "unit"

test-integration:
	pytest $(TEST_DIR)/integration -v -m "integration"

test-cov:
	pytest $(TEST_DIR) --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term

# Code quality targets
lint:
	flake8 $(SRC_DIR)
	mypy $(SRC_DIR)

format:
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

check: lint test

# Building targets
build: clean
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Documentation
docs:
	@echo "📚 Generating documentation..."
	# Add sphinx or mkdocs commands here when documentation is set up

# Example runners
demo:
	$(PYTHON) examples/hello_world.py

example-text:
	$(PYTHON) examples/text_similarity.py

example-image:
	$(PYTHON) examples/multimodal_search.py

# Development server (if API is implemented)
serve:
	uvicorn jina_embeddings.api.main:app --reload --port 8000

# Docker targets (if Docker is used)
docker-build:
	docker build -t jina-embeddings-v4:latest .

docker-run:
	docker run -it --rm jina-embeddings-v4:latest

# Utility targets
shell:
	$(PYTHON)

requirements:
	$(PIP) freeze > requirements-freeze.txt

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files