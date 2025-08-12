.PHONY: test test-quick
test:
	pytest
test-quick:
	pytest tests/signal/test_golden_single.py tests/signal/test_golden_batch.py -q
