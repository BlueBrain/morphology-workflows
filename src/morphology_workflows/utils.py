"""Util functions."""

import logging
import hashlib
import re
import shutil
import warnings
from itertools import chain
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import luigi
import pandas as pd
from luigi_tools.util import configparser_to_dict
from morph_tool.converter import convert
from morphio import Option
from morphio.mut import Morphology
from pkg_resources import resource_filename
from tqdm import tqdm

_TEMPLATES = Path(
    resource_filename(
        "morphology_workflows",
        "_templates",
    )
)

tqdm.pandas()

EXTS = {".asc", ".h5", ".swc"}  # allowed extensions
L = logging.getLogger(__name__)


class StrIndexMixin:
    """Mixin to ensure the dataset index is casted to str."""

    def transform_index(self, df):
        """The index is always casted to str."""
        df.index = df.index.astype(str)
        return df


def is_morphology(filename):
    """Returns True if the extension is supported."""
    try:
        Morphology(filename)
        ext = Path(filename).suffix.lower()
    except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
        L.warning("Error when loading the morphology from %s:\n%s", filename, exc)
        return False, None
    return ext in EXTS, ext


@contextmanager
def silent_warnings(*warning_classes):
    """A context manager to silent warnings during the body execution.

    Args:
        *warning_classes (class): The warning classes to be filtered.
    """
    if not warning_classes:
        warning_classes = [Warning]
    with warnings.catch_warnings():
        for warning in warning_classes:
            warnings.simplefilter("ignore", warning)
        yield


@contextmanager
def silent_loggers(*logger_names):
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

    for i in dir_path.iterdir():
        if i.suffix.lower() in EXTS:
            morph_files.append((i.with_suffix("").name, str(i.resolve())))
        else:
            L.info("The file '%s' is not detected as a morphology file and is thus discarded", i)
    df = pd.DataFrame(morph_files, columns=["morph_name", "morph_path"])

    # Deduplicate names
    for idx, name in df.loc[df["morph_name"].duplicated(), "morph_name"].items():
        i = 2
        new_name = name + f"_{i}"
        while (df["morph_name"] == new_name).any():
            i += 1
            new_name = name + f"_{i}"
        df.loc[idx, "morph_name"] = new_name

    df.sort_values("morph_name", inplace=True)
    df.to_csv(output_path, index=False)
    L.info("Created dataset in %s", output_path)


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
            raise FileExistsError(  # noqa: TRY003
                f"The directory {output_dir} already exists, please use another name"
            ) from None

    shutil.copyfile(_TEMPLATES / "logging.conf", output_dir / "logging.conf")

    fetch_config_file = None
    if source_db == "Allen":
        fetch_config_file = "allen_config.json"
    elif source_db == "NeuroMorpho":
        fetch_config_file = "neuromorpho_config.json"
    elif source_db == "MouseLight":
        fetch_config_file = "mouselight_config.json"
    elif source_db is not None:
        raise ValueError(  # noqa: TRY003
            f"The value '{source_db}' is not valid for the 'source_db' parameter"
        )

    luigi_cfg = luigi.configuration.cfg_parser.LuigiConfigParser()
    luigi_cfg.read(_TEMPLATES / "luigi.cfg")

    cfg = configparser_to_dict(luigi_cfg)

    if fetch_config_file is not None:
        shutil.copyfile(_TEMPLATES / fetch_config_file, output_dir / fetch_config_file)
        cfg["Fetch"]["source"] = source_db
        cfg["Fetch"]["config_file"] = fetch_config_file
    cfg["Curate"]["dataset_df"] = dataset_filename

    luigi_cfg.read_dict(cfg)
    with (output_dir / "luigi.cfg").open("w") as f:
        luigi_cfg.write(f)

    # Fix end-of-file because of ConfigParser.write()
    with (output_dir / "luigi.cfg").open("r") as f:
        data = f.read()
    data = re.sub(r"[\n]{2,}", "\n\n", data + "\n")[:-1]
    with (output_dir / "luigi.cfg").open("w") as f:
        f.write(data)

    L.info("Created inputs in %s", output_dir)

    if input_dir is not None:
        create_dataset_from_dir(input_dir, output_dir / dataset_filename)


def placeholders_to_nested_dict(df: pd.DataFrame) -> dict:
    """Convert a DataFrame containing placeholders into a nested dict."""
    first_cols = [("Metadata", "Region"), ("Metadata", "Mtype")]
    if ("property", "name") in df.columns:
        first_cols.append(("property", "name"))
    values = df.set_index(first_cols).reorder_levels([1, 0], axis=1).stack().stack()
    d = values.to_dict()
    result = {}
    for key, value in d.items():
        target = result
        for k in key[:-1]:  # traverse all keys except the last one
            target = target.setdefault(k, {})
        target[key[-1]] = value
    return result


def import_morph(morph_path, new_morph_path, annotation_path, new_annotation_path):
    """Copy a morphology and an annotation."""
    if Path(morph_path).suffix != Path(new_morph_path).suffix:
        convert(morph_path, new_morph_path)
    else:
        shutil.copy(morph_path, new_morph_path)
    shutil.copy(annotation_path, new_annotation_path)
    return Morphology(new_morph_path, options=Option.nrn_order)


def seed_from_name(name):
    """Build a seed from the name hash."""
    return int(hashlib.md5(name.encode("ascii")).hexdigest(), 16) % (2**32 - 1)


def rng_from_name(name):
    """Build a random number generator from the name hash."""
    return np.random.default_rng(seed_from_name(name))


def write_neuron(neuron: Morphology, filename):
    """Write a NEURON ordered version of the morphology."""
    Morphology(neuron, options=Option.nrn_order).write(filename)


def compare_lists(l0, l1):
    """Compare two sets of lists, returns sets of overlap, only in l0, and only in l1."""
    l0_set = set(l0)
    l1_set = set(l1)

    overlap = l0_set & l1_set
    only_in_l0 = l0_set - l1_set
    only_in_l1 = l1_set - l0_set

    return overlap, only_in_l0, only_in_l1


def get_points(input_file):
    """Get all points (soma, then neurites) from morphology."""
    m = Morphology(input_file)
    points = chain.from_iterable([[m.soma.points], [section.points for section in m.iter()]])
    return np.vstack(list(points))
