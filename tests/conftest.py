"""Configuration for the pytest test suite."""
import os
from pathlib import Path

import dir_content_diff
import dir_content_diff.pandas
import pytest

DATA = Path(__file__).parent / "data"
EXAMPLES = Path(__file__).parent / "examples_test"

dir_content_diff.pandas.register()


@pytest.fixture()
def tmp_working_dir(tmp_path):
    """Change working directory before a test and change it back when the test is finished."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)


@pytest.fixture()
def data_dir():
    """The data directory."""
    return DATA


@pytest.fixture()
def examples_dir():
    """The examples directory."""
    return EXAMPLES
