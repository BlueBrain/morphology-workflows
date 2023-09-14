"""Configuration for the pytest test suite."""
from pathlib import Path

import dir_content_diff
import dir_content_diff.pandas
import pytest

DATA = Path(__file__).parent / "data"
EXAMPLES = Path(__file__).parent.parent / "examples"
EXAMPLES_TEST = Path(__file__).parent / "examples_test"

dir_content_diff.pandas.register()


@pytest.fixture()
def tmp_working_dir(tmp_path, monkeypatch):
    """Change working directory before a test and change it back when the test is finished."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture()
def data_dir():
    """The data directory."""
    return DATA


@pytest.fixture()
def examples_dir():
    """The examples directory."""
    return EXAMPLES


@pytest.fixture()
def examples_test_dir():
    """The examples directory."""
    return EXAMPLES_TEST
