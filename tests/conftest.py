import os
import sys
# Ensure project root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import warnings
from contextlib import contextmanager
import pytest

_orig_warns = pytest.warns


def _warns(expected_warning=Warning, *args, **kwargs):
    if expected_warning is None:
        @contextmanager
        def ctx():
            with warnings.catch_warnings(record=True) as record:
                yield
            if record:
                msgs = ", ".join(str(w.message) for w in record)
                raise AssertionError(f"Unexpected warnings: {msgs}")
        return ctx()
    return _orig_warns(expected_warning, *args, **kwargs)


pytest.warns = _warns
