"""Util functions."""
import logging
import shutil
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import luigi
import pandas as pd
from luigi_tools.util import configparser_to_dict
from morphio.mut import Morphology
from pkg_resources import resource_filename
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
    output_path = Path(output_path)
    morph_files = []

    L = logging.getLogger(".".join(__name__.split(".")[:-1]))
    for i in dir_path.iterdir():
        if is_morphology(i)[0]:
            morph_files.append((i.relative_to(output_path.parent).with_suffix("").name, str(i)))
        else:
            L.info(f"The file '{i}' is not a valid morphology and is thus discarded")
    df = pd.DataFrame(morph_files, columns=["morph_name", "morph_path"])
    df.to_csv(output_path, index=False)


def create_inputs(
    source_db=None,
    input_dir=None,
    output_dir=None,
    dataset_filename="dataset.csv",
):
    """Create inputs for the workflows."""
    if output_dir is None:
        output_dir = Path()
    else:
        output_dir = Path(output_dir)
        try:
            output_dir.mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(
                f"The directory {output_dir} already exists, please use another name"
            )

    template_dir = Path(
        resource_filename(
            "morphology_workflows",
            "_templates",
        )
    )

    shutil.copyfile(template_dir / "logging.conf", output_dir / "logging.conf")

    fetch_config_file = None
    if source_db == "Allen":
        fetch_config_file = "allen_config.json"
    elif source_db == "NeuroMorpho":
        fetch_config_file = "neuromorpho_config.json"
    elif source_db == "MouseLight":
        fetch_config_file = "mouselight_config.json"
    elif source_db is not None:
        raise ValueError(f"The value '{source_db}' is not valid for the 'source_db' parameter")

    luigi_cfg = luigi.configuration.cfg_parser.LuigiConfigParser()
    luigi_cfg.read(template_dir / "luigi.cfg")

    cfg = configparser_to_dict(luigi_cfg)

    if fetch_config_file is not None:
        shutil.copyfile(template_dir / fetch_config_file, output_dir / fetch_config_file)
        cfg["Fetch"]["source"] = source_db

    cfg["Fetch"]["config_file"] = fetch_config_file
    cfg["Curate"]["dataset_df"] = dataset_filename

    luigi_cfg.read_dict(cfg)
    with (output_dir / "luigi.cfg").open("w") as f:
        luigi_cfg.write(f)
