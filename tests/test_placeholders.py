"""Test download tasks."""
# pylint: disable=redefined-outer-name
import json
import shutil
from copy import deepcopy
from pathlib import Path

import dictdiffer
import luigi
import numpy as np
import pandas as pd
import pytest
from pkg_resources import resource_filename

from morphology_workflows.placeholders import DEFAULT_CONFIG
from morphology_workflows.tasks import _TEMPLATES
from morphology_workflows.tasks.placeholders import Placeholders
from morphology_workflows.utils import placeholders_to_nested_dict


def build_metadata(input_dir):
    """Build dummy metadata for the input directory containing morphologies."""
    morphs = sorted(
        [str(i) for i in (input_dir).iterdir() if i.suffix.lower() in [".asc", ".h5", ".swc"]]
    )
    df = pd.DataFrame(
        {
            "species": ["any species"] * len(morphs),
            "brain_region": ["targeted region"] * 5 + ["other regions"] * (len(morphs) - 5),
            "cell_type": ["targeted mtype"] * 3 + ["other mtypes"] * (len(morphs) - 3),
            "nb_morphologies": [3] * 3 + [2] * 2 + [(len(morphs) - 5)] * (len(morphs) - 5),
            "morphology": morphs,
        }
    )
    df.to_csv(str(input_dir / "metadata.csv"))


@pytest.fixture()
def prepare_dir(tmp_working_dir, examples_test_dir):
    """Setup the working directory."""
    shutil.copyfile(_TEMPLATES / "luigi.cfg", tmp_working_dir / "luigi.cfg")
    shutil.copyfile(_TEMPLATES / "logging.conf", tmp_working_dir / "logging.conf")
    shutil.copytree(examples_test_dir / "morphologies", tmp_working_dir / "morphologies")

    build_metadata(tmp_working_dir / "morphologies")

    # Set current config in luigi
    luigi_config = luigi.configuration.get_config()
    luigi_config.read(str(tmp_working_dir / "luigi.cfg"))

    yield tmp_working_dir

    # Reset luigi config
    luigi_config.clear()


@pytest.fixture()
def default_config():
    """Setup the default config."""
    return [
        {
            "populations": [{"region": "targeted region", "mtype": "targeted mtype"}],
            "config": DEFAULT_CONFIG,
        },
    ]


@pytest.fixture()
def config_path(prepare_dir):
    """The path to the configuration file used in test."""
    return prepare_dir / "config.json"


def test_placeholders(
    prepare_dir, data_dir, default_config, config_path, WorkflowTask_exception_event  # noqa: ARG001
):  # pylint: disable=unused-argument
    """Test placeholders computation."""
    del default_config[0]["config"]
    with config_path.open("w") as f:
        json.dump(default_config, f)

    expected = pd.read_csv(data_dir / "placeholders.csv", header=[0, 1])

    # Test with 1 job
    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        config=config_path,
    )
    assert luigi.build([task], local_scheduler=True)

    result = pd.read_csv(task.output().path, header=[0, 1])

    pd.testing.assert_frame_equal(result, expected)

    # Test with 2 jobs
    result_path = Path("placeholders_2_jobs.csv")
    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        config=config_path,
        result_path=result_path,
        nb_jobs=2,
    )
    assert luigi.build([task], local_scheduler=True)

    result = pd.read_csv(task.output().path, header=[0, 1])

    pd.testing.assert_frame_equal(result, expected)

    # Test JSON export
    result_path = Path("placeholders.json")
    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        config=config_path,
        result_path=result_path,
    )
    assert luigi.build([task], local_scheduler=True)

    with result_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    assert not list(dictdiffer.diff(result, placeholders_to_nested_dict(expected), tolerance=1e-6))


def test_placeholders_bad_extension(tmp_working_dir, WorkflowTask_exception_event):
    """Test wrong extension in result path."""
    task = Placeholders(
        input_dir=tmp_working_dir,  # This path just has to exist in this test
        result_path="placeholders.WRONG_EXTENSION",
    )
    assert not luigi.build([task], local_scheduler=True)

    failed_task, exceptions = WorkflowTask_exception_event
    assert len(failed_task) == 1
    assert failed_task[0].startswith("Placeholders(")
    assert exceptions == [
        "The 'result_path' parameter should have a '.csv' or '.json' extension, not "
        "'.WRONG_EXTENSION'."
    ]


def test_placeholders_with_config(prepare_dir, data_dir, default_config, config_path):
    """Test placeholders computation."""
    with config_path.open("w") as f:
        json.dump(default_config, f)

    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        config=config_path,
    )
    assert luigi.build([task], local_scheduler=True)

    result = pd.read_csv(task.output().path, header=[0, 1])

    expected = pd.read_csv(data_dir / "placeholders.csv", header=[0, 1])

    pd.testing.assert_frame_equal(result, expected)


def test_placeholders_no_metadata(prepare_dir, data_dir, default_config, config_path):
    """Test placeholders computation with no metadata.csv file."""
    for i in (prepare_dir / "morphologies").iterdir():
        if not i.stem.startswith("C") or i.suffix != ".asc":
            i.unlink()

    del default_config[0]["config"]
    with config_path.open("w") as f:
        json.dump(default_config, f)

    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        config=config_path,
    )
    with pytest.warns(
        UserWarning,
        match="No metadata.csv file found in the input directory, loading all morphologies",
    ):
        assert luigi.build([task], local_scheduler=True)

    result = pd.read_csv(task.output().path, header=[0, 1])

    expected = pd.read_csv(data_dir / "placeholders.csv", header=[0, 1])

    pd.testing.assert_frame_equal(
        result.sort_values(("property", "name")).reset_index(drop=True), expected.loc[[0, 1]]
    )


def test_placeholders_optional_params(prepare_dir, data_dir, default_config, config_path):
    """Test placeholders computation with no metadata.csv file."""
    # Prepare data
    morph_dir = prepare_dir / "morphologies"
    for i in morph_dir.iterdir():
        if i.stem.startswith("C") and i.suffix == ".asc":
            continue
        i.unlink()
    morphs = sorted([str(i) for i in morph_dir.iterdir() if i.suffix.lower() == ".asc"])
    df = pd.DataFrame(
        {
            "species": ["any species"] * len(morphs),
            "brain_region": ["targeted region"] * len(morphs),
            "cell_type": ["targeted mtype"] * len(morphs),
            "nb_morphologies": [1] * len(morphs),
            "morphology": morphs,
        }
    )
    df.to_csv(str(morph_dir / "metadata.csv"))

    del default_config[0]["config"]

    # No mtype
    tmp_config = deepcopy(default_config)
    del tmp_config[0]["populations"][0]["mtype"]
    with config_path.open("w") as f:
        json.dump(tmp_config, f)
    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        config=config_path,
        result_path="no_mtype.csv",
    )
    assert luigi.build([task], local_scheduler=True)

    result = pd.read_csv(task.output().path, header=[0, 1])

    expected = pd.read_csv(data_dir / "placeholders.csv", header=[0, 1])
    expected[("Metadata", "Mtype")] = np.nan

    pd.testing.assert_frame_equal(
        result.sort_values(("property", "name")).reset_index(drop=True), expected.loc[[0, 1]]
    )

    # No region
    tmp_config = deepcopy(default_config)
    del tmp_config[0]["populations"][0]["region"]
    with config_path.open("w") as f:
        json.dump(tmp_config, f)
    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        config=config_path,
        result_path="no_region.csv",
    )
    assert luigi.build([task], local_scheduler=True)

    result = pd.read_csv(task.output().path, header=[0, 1])

    expected = pd.read_csv(data_dir / "placeholders.csv", header=[0, 1])
    expected[("Metadata", "Region")] = np.nan

    pd.testing.assert_frame_equal(
        result.sort_values(("property", "name")).reset_index(drop=True), expected.loc[[0, 1]]
    )

    # No region and no mtype
    tmp_config = deepcopy(default_config)
    del tmp_config[0]["populations"]
    with config_path.open("w") as f:
        json.dump(tmp_config, f)
    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        config=config_path,
        result_path="no_region-no_mtype.csv",
    )
    assert luigi.build([task], local_scheduler=True)

    result = pd.read_csv(task.output().path, header=[0, 1])

    expected = pd.read_csv(data_dir / "placeholders.csv", header=[0, 1])
    expected[("Metadata", "Region")] = np.nan
    expected[("Metadata", "Mtype")] = np.nan

    pd.testing.assert_frame_equal(
        result.sort_values(("property", "name")).reset_index(drop=True), expected.loc[[0, 1]]
    )


def test_placeholders_empty_population(prepare_dir, default_config, config_path):
    """Test that the default placeholders are used when the population is empty."""
    region = "unknown region"
    mtype = "targeted mtype"

    default_config[0]["populations"][0]["region"] = region
    with config_path.open("w") as f:
        json.dump(default_config, f)

    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        config=config_path,
    )
    assert luigi.build([task], local_scheduler=True)

    result = pd.read_csv(task.output().path, header=[0, 1])

    expected = pd.read_csv(
        resource_filename(
            "morphology_workflows",
            "_data/default_placeholders.csv",
        ),
        header=[0, 1],
    )
    expected[("Metadata", "Region")] = region
    expected[("Metadata", "Mtype")] = mtype

    assert sorted(result) == sorted(expected)
    expected = expected[result.columns].sort_values(("property", "name")).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_placeholders_aggregation_mode(prepare_dir, data_dir, default_config, config_path):
    """Test placeholders computation with different aggregation modes."""
    default_config[0]["populations"].append(deepcopy(default_config[0]["populations"][0]))
    default_config[0]["populations"][1]["mode"] = "morphology"
    default_config[0]["populations"].append(deepcopy(default_config[0]["populations"][0]))
    default_config[0]["populations"][2]["mode"] = "population"
    with config_path.open("w") as f:
        json.dump(default_config, f)

    # Update expected for aggregated population
    expected = pd.read_csv(data_dir / "placeholders.csv", header=[0, 1])
    min_cols = [col for col in expected.columns if col[1].startswith("min_")]
    max_cols = [col for col in expected.columns if col[1].startswith("max_")]
    sum_cols = [col for col in expected.columns if col[1].startswith("sum_")]
    tmp = expected.mean()
    tmp.loc[expected.isnull().any()] = np.nan  # Fix columns with NaN values
    tmp[min_cols] = expected[min_cols].min()
    tmp[max_cols] = expected[max_cols].max()
    tmp[sum_cols] = expected[sum_cols].sum()
    expected.loc[4] = tmp
    expected.loc[4, ("property", "name")] = "__aggregated_population__"
    expected.loc[4, ("Metadata", "Region")] = expected.loc[0, ("Metadata", "Region")]
    expected.loc[4, ("Metadata", "Mtype")] = expected.loc[0, ("Metadata", "Mtype")]
    expected = expected.sort_values(("property", "name")).reset_index(drop=True)

    # Compute placeholders with aggregation and export in CSV format
    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        config=config_path,
    )
    assert luigi.build([task], local_scheduler=True)

    result = pd.read_csv(task.output().path, header=[0, 1])
    result = result.sort_values(("property", "name")).reset_index(drop=True)

    # Check the results
    assert result[("property", "name")].tolist() == expected[("property", "name")].tolist()
    pd.testing.assert_frame_equal(result, expected)

    # Compute placeholders with aggregation and export in JSON format
    result_path = Path("placeholders.json")
    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        config=config_path,
        result_path=result_path,
    )
    assert luigi.build([task], local_scheduler=True)

    with result_path.open("r", encoding="utf-8") as f:
        result_json = json.load(f)

    # Check the results
    assert not list(
        dictdiffer.diff(result_json, placeholders_to_nested_dict(expected), tolerance=1e-6)
    )
