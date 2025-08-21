# Ethiopian Telegram NER System Makefile

.PHONY: help install install-dev test test-unit test-integration lint format type-check clean setup-data run-pipeline dashboard

# Default target
help:
	@echo "Ethiopian Telegram NER System"
	@echo "Available commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black"
	@echo "  type-check       Run type checking with mypy"
	@echo "  clean            Clean up generated files"
	@echo "  setup-data       Create data directories"
	@echo "  run-pipeline     Run the full pipeline"
	@echo "  dashboard        Start Streamlit dashboard"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy pre-commit

# Testing
test: test-unit test-integration

test-unit:
	pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term

test-integration:
	pytest tests/integration/ -v -m "not slow"

test-slow:
	pytest tests/integration/ -v -m "slow"

# Code quality
lint:
	flake8 src tests --max-line-length=127
	black --check src tests

format:
	black src tests

type-check:
	mypy src --ignore-missing-imports

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Setup
setup-data:
	mkdir -p data/raw data/processed data/labeled
	mkdir -p models/checkpoints models/logs models/plots models/interpretability models/vendor_analytics
	mkdir -p logs

# Pipeline operations
run-pipeline:
	python scripts/main_pipeline.py --step full --limit 500

run-ingestion:
	python scripts/main_pipeline.py --step ingestion --limit 1000

run-preprocessing:
	python scripts/main_pipeline.py --step preprocessing --data-path data/raw/telegram_data.csv

run-training:
	python scripts/main_pipeline.py --step training --data-path data/labeled/dataset.txt

run-analytics:
	python scripts/main_pipeline.py --step analytics --data-path data/processed/processed_data.csv

# Dashboard
dashboard:
	streamlit run src/dashboard/streamlit_app.py

# Development setup
dev-setup: install-dev setup-data
	pre-commit install
	@echo "Development environment setup complete!"

# CI/CD simulation
ci-test: lint type-check test
	@echo "All CI checks passed!"

# Docker operations (if using Docker)
docker-build:
	docker build -t ethiopian-ner-system .

docker-run:
	docker run -p 8501:8501 ethiopian-ner-system

# Documentation
docs:
	@echo "Documentation generation placeholder"

# Release preparation
release-check: clean ci-test
	@echo "Release checks completed successfully!"