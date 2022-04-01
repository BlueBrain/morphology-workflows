"""Util functions."""
import logging
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
