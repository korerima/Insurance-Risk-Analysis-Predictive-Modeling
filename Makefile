.PHONY: help install install-dev test test-cov lint format type-check security clean build docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  type-check   - Run type checking with mypy"
	@echo "  security     - Run security checks"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docs         - Build documentation"
	@echo "  pre-commit   - Run pre-commit hooks"
	@echo "  validate     - Validate data"
	@echo "  preprocess   - Preprocess data"
	@echo "  train        - Train model"
	@echo "  predict      - Make predictions"
	@echo "  dashboard    - Launch Streamlit dashboard"
	@echo "  dashboard-dev - Launch dashboard in development mode"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black src/ tests/ --line-length=127
	isort src/ tests/ --profile=black --line-length=127

type-check:
	mypy src/ --ignore-missing-imports

security:
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

# Build and documentation
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

build:
	python -m build

docs:
	cd docs && make html

# Pre-commit
pre-commit:
	pre-commit run --all-files

# CLI commands
validate:
	python -m src.cli validate --data-file Datasets/MachineLearningRating_v4.csv

preprocess:
	python -m src.cli preprocess --input-file Datasets/MachineLearningRating_v4.csv --output-file processed_data.csv

train:
	python -m src.cli train --data-file processed_data.csv --model-type random_forest

predict:
	python -m src.cli predict --data-file new_data.csv --model-file risk_model_random_forest.pkl

# Dashboard commands
dashboard:
	streamlit run src/dashboard/streamlit_app.py --server.port 8501 --server.address localhost

dashboard-dev:
	streamlit run src/dashboard/streamlit_app.py --server.port 8501 --server.address localhost --server.runOnSave true

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete!"

check-all: lint type-check security test
	@echo "All checks completed!"

# CI/CD helpers
ci-test:
	pytest tests/ -v --cov=src --cov-report=xml --cov-report=html

ci-lint:
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	black --check src/ tests/
	isort --check-only src/ tests/

ci-type:
	mypy src/ --ignore-missing-imports

ci-security:
	bandit -r src/ -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true

# Data management
dvc-pull:
	dvc pull

dvc-push:
	dvc push

# Environment management
venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

# Quick start
quick-start: install-dev dvc-pull validate preprocess train
	@echo "Quick start completed! Model trained and ready for predictions."

# Dashboard quick start
dashboard-start: install-dev dvc-pull dashboard
	@echo "Dashboard started! Open http://localhost:8501 in your browser."
