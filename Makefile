.PHONY: help install ingest run test clean format

help:
	@echo "Available commands:"
	@echo "  make install  - Install Python dependencies"
	@echo "  make ingest   - Ingest PDF document into vector database"
	@echo "  make run      - Launch FastAPI dev server with auto-reload"
	@echo "  make test     - Run pytest suite"
	@echo "  make clean    - Remove cache and temporary files"

install:
	pip install -r requirements.txt

ingest:
	python -m app.ingest

run:
	uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

clean:
	rm -rf __pycache__ .pytest_cache .chroma htmlcov *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
