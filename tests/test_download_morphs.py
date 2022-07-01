"""Test download tasks."""
# pylint: disable=redefined-outer-name
import configparser
import sys

import luigi
import mock
import numpy as np
import pandas as pd
import pytest

from morphology_workflows.tasks.fetch import Fetch


@pytest.fixture()
def prepare_dir(tmp_working_dir, examples_dir):
    """Setup the working directory."""
    config = configparser.ConfigParser()
    config.read(examples_dir / "logging.conf")
    config["logger_root"]["level"] = "DEBUG"
    config["logger_luigi"]["level"] = "DEBUG"
    with (tmp_working_dir / "logging.conf").open("w", encoding="utf-8") as f:
        config.write(f)

    luigi_config = configparser.ConfigParser()
    luigi_config.add_section("core")
    luigi_config["core"]["logging_conf_file"] = "logging.conf"
    with (tmp_working_dir / "luigi.cfg").open("w", encoding="utf-8") as f:
        luigi_config.write(f)

    # Set current config in luigi
    luigi_config = luigi.configuration.get_config()
    luigi_config.read(str(tmp_working_dir / "luigi.cfg"))

    yield tmp_working_dir

    # Reset luigi config
    luigi_config.clear()


def test_neuromorpho(prepare_dir, data_dir):
    """Download from NeuroMorpho."""
    task = Fetch(
        source="NeuroMorpho",
        config_file=data_dir / "neuromorpho_config_download.json",
        result_path=prepare_dir / "morphologies",
    )
    assert luigi.build([task], local_scheduler=True)

    metadata = pd.read_csv(task.output()["metadata"].path)

    # Check metadata
    expected = pd.DataFrame(
        {
            "species": ["mouse"] * 10,
            "brain_region": ["neocortex"] * 2 + ["tectum"] * 8,
            "cell_type": ["interneuron"] * 2 + ["pyramidal"] * 8,
            "nb_morphologies": [2] * 2 + [np.nan] * 4 + [9999] * 4,
            "morphology": [
                "102367.swc",
                "102369.swc",
                "121518.swc",
                "121694.swc",
                "121695.swc",
                "121696.swc",
                "121518.swc",
                "121694.swc",
                "121695.swc",
                "121696.swc",
            ],
        }
    )

    pd.testing.assert_frame_equal(metadata, expected)

    # Check the files exist
    output_path = task.output()["morphologies"].pathlib_path
    assert metadata["morphology"].apply(lambda x: (output_path / x).exists()).all()


def test_mouselight(prepare_dir, data_dir):
    """Download from NeuroMorpho."""
    task = Fetch(
        source="MouseLight",
        config_file=data_dir / "mouselight_config_download.json",
        result_path=prepare_dir / "morphologies",
    )
    assert luigi.build([task], local_scheduler=True)

    metadata = pd.read_csv(task.output()["metadata"].path)

    # Check metadata
    expected = pd.DataFrame(
        {
            "brain_region": ["AHN"] * 10,
            "nb_morphologies": [2] * 2 + [np.nan] * 4 + [9999] * 4,
            "seed": [0] * 10,
            "morphology": [
                "85214.swc",
                "84993.swc",
                "84993.swc",
                "85214.swc",
                "122053.swc",
                "85215.swc",
                "84993.swc",
                "85214.swc",
                "122053.swc",
                "85215.swc",
            ],
        }
    )

    pd.testing.assert_frame_equal(metadata, expected)

    # Check the files exist
    output_path = task.output()["morphologies"].pathlib_path
    assert metadata["morphology"].apply(lambda x: (output_path / x).exists()).all()


def test_allen(prepare_dir, data_dir):
    """Download from NeuroMorpho."""
    task = Fetch(
        source="Allen",
        config_file=data_dir / "allen_config_download.json",
        result_path=prepare_dir / "morphologies",
    )
    assert luigi.build([task], local_scheduler=True)

    metadata = pd.read_csv(task.output()["metadata"].path)

    # Check metadata
    expected = pd.DataFrame(
        {
            "species": ["Mus musculus"] * 10,
            "brain_region": ["VISli"] * 10,
            "cell_type": ["interneuron"] * 2 + ["pyramidal"] * 8,
            "nb_morphologies": [2] * 2 + [np.nan] * 4 + [9999] * 4,
            "seed": [0] * 10,
            "morphology": [
                "555019563.swc",
                "603402458.swc",
                "555019563.swc",
                "603402458.swc",
                "526573598.swc",
                "555241040.swc",
                "555019563.swc",
                "603402458.swc",
                "526573598.swc",
                "555241040.swc",
            ],
        }
    )

    pd.testing.assert_frame_equal(metadata, expected)

    # Check the files exist
    output_path = task.output()["morphologies"].pathlib_path
    assert metadata["morphology"].apply(lambda x: (output_path / x).exists()).all()


class TestMissingImports:
    """Test the errors raised when optional dependencies are not installed."""

    @mock.patch.dict(sys.modules, {"morphapi.api.allenmorphology": None})
    def test_allen(self, prepare_dir, data_dir):
        """Test Allen Brain case."""

        class FetchTmp(Fetch):
            """Dummy task class to attach the event handler."""

        task = FetchTmp(
            source="Allen",
            config_file=data_dir / "allen_config_download.json",
            result_path=prepare_dir / "morphologies",
        )

        failed_tasks = []
        exceptions = []

        @FetchTmp.event_handler(luigi.Event.FAILURE)
        def check_exception(failed_task, exception):
            failed_tasks.append(str(failed_task))
            exceptions.append(str(exception))

        assert not luigi.build([task], local_scheduler=True)

        assert failed_tasks == [
            "FetchTmp(tag_output=False, source=Allen, config_file="
            f"{data_dir / 'allen_config_download.json'}, "
            f"result_path={prepare_dir / 'morphologies'})"
        ]
        assert exceptions == [
            'You need to install the "allensdk" package to fetch morphologies from the Allen Brain '
            'database: "pip install allensdk"'
        ]

    @mock.patch.dict(sys.modules, {"bg_atlasapi": None})
    def test_mouselight(self, prepare_dir, data_dir):
        """Test MouseLight case."""

        class FetchTmp(Fetch):
            """Dummy task class to attach the event handler."""

        task = FetchTmp(
            source="MouseLight",
            config_file=data_dir / "mouselight_config_download.json",
            result_path=prepare_dir / "morphologies",
        )

        failed_tasks = []
        exceptions = []

        @FetchTmp.event_handler(luigi.Event.FAILURE)
        def check_exception(failed_task, exception):
            failed_tasks.append(str(failed_task))
            exceptions.append(str(exception))

        assert not luigi.build([task], local_scheduler=True)

        assert failed_tasks == [
            "FetchTmp(tag_output=False, source=MouseLight, "
            f"config_file={data_dir / 'mouselight_config_download.json'}, "
            f"result_path={prepare_dir / 'morphologies'})"
        ]
        assert exceptions == [
            'You need to install the "bg_atlasapi" package to fetch morphologies from the '
            'MouseLight database: "pip install bg_atlasapi"'
        ]
