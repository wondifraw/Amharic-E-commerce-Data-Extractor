# Amharic E-commerce Data Extractor - Makefile

.PHONY: test benchmark clean install setup

# Install dependencies
install:
	pip install -r requirements.txt

# Setup project
setup: install
	python setup_env.py
	mkdir -p data/processed data/raw logs models reports
	python -m pytest tests/ --verbose

# Run unit tests
test:
	python -m pytest tests/ --verbose --cov=src

# Run performance benchmarks
benchmark:
	jupyter nbconvert --execute notebooks/Performance_Benchmarking.ipynb --to html
	@echo "Benchmark results saved to notebooks/Performance_Benchmarking.html"

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf notebooks/.ipynb_checkpoints

# Run full pipeline
pipeline:
	jupyter nbconvert --execute notebooks/Data_Collection.ipynb --to html
	jupyter nbconvert --execute notebooks/Preprocessing.ipynb --to html
	jupyter nbconvert --execute notebooks/Fine_tune_xlm_roberta.ipynb --to html
	jupyter nbconvert --execute notebooks/vendor_scorecard_Engine.ipynb --to html

# Development setup
dev: install
	pip install pytest pytest-cov jupyter black flake8
	pre-commit install