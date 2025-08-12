import logging
import os

_LOG_LEVEL_NAME = os.getenv('QT_LOG_LEVEL', 'INFO').upper()
try:
    _LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME)
except AttributeError:
    _LOG_LEVEL = logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Return a logger with level defined by ``QT_LOG_LEVEL`` env variable.

    The logger uses existing handlers and formatting without modifications.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)
    return logger

