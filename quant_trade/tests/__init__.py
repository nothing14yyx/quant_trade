import os

# Expose repository-level tests package for internal imports
__path__ = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tests'))]


def test_signal_generator():
    return None