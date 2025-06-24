import sys
import logging


def test_import_does_not_configure_logging(tmp_path, monkeypatch):
    # Remove the module if already imported
    sys.modules.pop("quant_trade.run_scheduler", None)

    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    import quant_trade.run_scheduler  # noqa: F401

    assert list(root.handlers) == original_handlers
    assert root.level == original_level

