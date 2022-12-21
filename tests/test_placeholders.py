"""Test download tasks."""
# pylint: disable=redefined-outer-name
import shutil

import luigi
import pandas as pd
import pytest
from pkg_resources import resource_filename

from morphology_workflows.tasks.placeholders import Placeholders


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
def prepare_dir(tmp_working_dir, examples_dir):
    """Setup the working directory."""
    shutil.copyfile(examples_dir / "luigi.cfg", tmp_working_dir / "luigi.cfg")
    shutil.copyfile(examples_dir / "logging.conf", tmp_working_dir / "logging.conf")
    shutil.copytree(examples_dir / "morphologies", tmp_working_dir / "morphologies")

    build_metadata(tmp_working_dir / "morphologies")

    # Set current config in luigi
    luigi_config = luigi.configuration.get_config()
    luigi_config.read(str(tmp_working_dir / "luigi.cfg"))

    yield tmp_working_dir

    # Reset luigi config
    luigi_config.clear()


def test_placeholders(prepare_dir, data_dir):
    """Test placeholders computation."""
    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        region="targeted region",
        mtype="targeted mtype",
    )
    assert luigi.build([task], local_scheduler=True)

    result = pd.read_csv(task.output().path, header=[0, 1])

    expected = pd.read_csv(data_dir / "placeholders.csv", header=[0, 1])

    pd.testing.assert_frame_equal(result, expected)


def test_placeholders_no_metadata(prepare_dir, data_dir):
    """Test placeholders computation with no metadata.csv file."""
    for i in (prepare_dir / "morphologies").iterdir():
        if not i.stem.startswith("C") or i.suffix != ".asc":
            i.unlink()
    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        region="targeted region",
        mtype="targeted mtype",
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


def test_placeholders_empty_population(prepare_dir):
    """Test that the default placeholders are used when the population is empty."""
    region = "unknown region"
    mtype = "targeted mtype"

    task = Placeholders(
        input_dir=prepare_dir / "morphologies",
        region=region,
        mtype=mtype,
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

    pd.testing.assert_frame_equal(result, expected)
