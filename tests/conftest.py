"""Configuration for the pytest test suite."""
from copy import deepcopy
from pathlib import Path

import dir_content_diff
import dir_content_diff.pandas
import luigi
import pytest
from luigi_tools.task import WorkflowTask

DATA = Path(__file__).parent / "data"
EXAMPLES = Path(__file__).parent.parent / "examples"
EXAMPLES_TEST = Path(__file__).parent / "examples_test"

dir_content_diff.pandas.register()


@pytest.fixture()
def tmp_working_dir(tmp_path, monkeypatch):
    """Change working directory before a test and change it back when the test is finished."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


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


@pytest.fixture()
def WorkflowTask_exception_event():
    """Fixture to catch exception from tasks deriving from WorkflowTask.

    The events of the tasks are reset afterwards.
    """
    # pylint: disable=protected-access
    current_callbacks = deepcopy(luigi.Task._event_callbacks)  # noqa: SLF001

    failed_task = []
    exceptions = []

    @WorkflowTask.event_handler(luigi.Event.FAILURE)
    def check_exception(task, exception):
        failed_task.append(str(task))
        exceptions.append(str(exception))

    yield failed_task, exceptions

    luigi.Task._event_callbacks = current_callbacks  # noqa: SLF001
