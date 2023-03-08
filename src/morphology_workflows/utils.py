"""Util functions."""
import logging
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import pandas as pd
from luigi_tools.util import luigi_config_to_dict
from morphio.mut import Morphology
from tqdm import tqdm

tqdm.pandas()

EXTS = {".asc", ".h5", ".swc"}  # allowed extensions
EXAMPLE_PATH = Path(__file__).parent.parent.parent / "examples"


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


def create_dataset_from_dir(dir_path, output_path):
    """Generate a dataset from a directory."""
    dir_path = Path(dir_path)
    morph_files = []

    L = logging.getLogger(".".join(__name__.split(".")[:-1]))
    for i in dir_path.iterdir():
        if is_morphology(i)[0]:
            morph_files.append((i.with_suffix("").name, str(i)))
        else:
            L.info(f"The file is not a valid morphology and is thus discarded")
    df = pd.DataFrame(morph_files, columns=["morph_name", "morph_path"])
    df.to_csv(output_path, index=False)


def create_inputs(
    source_db=None,
    input_dir=None,
    output_dir="",
    dataset_filename="dataset.csv",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = luigi_config_to_dict(EXAMPLE_PATH / "luigi.cfg")
    luigi_cfg = luigi.configuration.cfg_parser.LuigiConfigParser()
    luigi_cfg.read_dict(cfg)
    luigi_cfg.write(output_dir / "luigi.cfg")
