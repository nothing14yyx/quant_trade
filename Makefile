.RECIPEPREFIX := >

.PHONY: lint test test-quick

lint:
>ruff check --fix .
>mypy .

test: lint
>pytest

test-quick:
>pytest tests/signal/test_golden_single.py tests/signal/test_golden_batch.py -q
