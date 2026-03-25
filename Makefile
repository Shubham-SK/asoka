.PHONY: run lint test

run:
	uvicorn app.main:app --reload --port 8000

lint:
	ruff check app

test:
	pytest -q
