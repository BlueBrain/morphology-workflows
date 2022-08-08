"""Util functions."""
import logging
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from morphio.mut import Morphology
from tqdm import tqdm

tqdm.pandas()

EXTS = {".asc", ".h5", ".swc"}  # allowed extensions


def is_morphology(filename):
    """Returns True if the extension is supported."""
    try:
        Morphology(filename)
        ext = Path(filename).suffix.lower()
        return ext in EXTS, ext
    except Exception:  # pylint: disable=broad-except
        return False, None


def silent_logger(log_name):
    """A decorator to silent a logger during the function execution."""

    def _silent_logger(function):
        @wraps(function)
        def decorated_func(*args, **kwargs):
            func_logger = logging.getLogger(log_name)
            func_logger.disabled = True
            try:
                return function(*args, **kwargs)
            finally:
                func_logger.disabled = False

        return decorated_func

    return _silent_logger


@contextmanager
def disable_loggers(*logger_names):
    """A context manager to silent loggers during the body execution.

    Args:
        *logger_names (str): The names of the loggers to be disabled.
    """
    if not logger_names:
        loggers = [logging.root]
    else:
        loggers = [logging.getLogger(i) for i in logger_names]

    disabled_loggers = [(i, i.disabled) for i in loggers]

    try:
        for i, _ in disabled_loggers:
            i.disabled = True
        yield
    finally:
        for i, j in disabled_loggers:
            i.disabled = j
