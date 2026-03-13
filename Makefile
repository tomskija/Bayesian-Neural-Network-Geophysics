.PHONY: help setup test format lint clean run run-tf example example-tf compare test-gpu

help:
	@echo "Bakken BNN - Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Install all dependencies"
	@echo "  make setup-dev      - Install with development tools"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-numpy     - Test NumPy implementation only"
	@echo "  make test-tf        - Test TensorFlow implementation only"
	@echo "  make test-gpu       - Test GPU functionality"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format         - Format code with black and isort"
	@echo "  make lint           - Run linting checks"
	@echo "  make type-check     - Run type checking with mypy"
	@echo ""
	@echo "Execution:"
	@echo "  make run            - Run main workflow (NumPy)"
	@echo "  make run-tf         - Run main workflow (TensorFlow)"
	@echo "  make example        - Run simple example (NumPy)"
	@echo "  make example-tf     - Run simple example (TensorFlow)"
	@echo "  make compare        - Compare implementations"
	@echo "  make workflow-tf    - Run Bakken workflow (TensorFlow)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove generated files"
	@echo "  make clean-all      - Remove all generated files and models"

setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

setup-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev,jupyter]"

test:
	pytest tests/ -v

test-numpy:
	pytest tests/test_bnn.py -v

test-tf:
	pytest tests/test_bnn_tf.py -v

test-gpu:
	pytest tests/test_bnn_tf.py::test_gpu_usage -v

format:
	black rockPropCalculator/ examples/ tests/
	isort rockPropCalculator/ examples/ tests/

lint:
	flake8 rockPropCalculator/ examples/
	pylint rockPropCalculator/

type-check:
	mypy rockPropCalculator/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/

clean-all: clean
	rm -rf logs/tensorboard/*
	rm -rf checkpoints/*
	rm -rf FiguresAndResults/*.png
	rm -rf FiguresAndResults/*.pkl

run:
	cd rockPropCalculator && python main.py

run-tf:
	cd rockPropCalculator && python main_tf.py

example:
	cd rockPropCalculator && python run_example.py

example-tf:
	cd rockPropCalculator && python run_example_tf.py

compare:
	cd examples && python compare_implementations.py

workflow-tf:
	cd examples && python bakken_workflow_tf.py