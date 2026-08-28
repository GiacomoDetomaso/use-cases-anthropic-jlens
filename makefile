.PHONY: typecheck lint lint-check format test

typecheck:
	uv run mypy src
lint:
	uv run ruff check . --fix
lint-check:
	uv run ruff check .
format:
	uv run ruff format .
test:
	uv run pytest