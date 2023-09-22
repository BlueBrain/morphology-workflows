"""Placeholders tasks."""
import json
import logging

from data_validation_framework.target import TaggedOutputLocalTarget
from data_validation_framework.task import TagResultOutputMixin
from luigi.parameter import NumericalParameter
from luigi.parameter import OptionalPathParameter
from luigi.parameter import PathParameter
from luigi_tools.task import WorkflowTask

from morphology_workflows.placeholders import compute_placeholders
from morphology_workflows.tasks.fetch import Fetch
from morphology_workflows.utils import placeholders_to_nested_dict

logger = logging.getLogger(__name__)


class Placeholders(TagResultOutputMixin, WorkflowTask):
    """Compute the placeholders for a given region and mtype set from a given config."""

    config = OptionalPathParameter(
        description=":str: The path to the JSON config file.",
        exists=True,
        default=None,
    )
    input_dir = OptionalPathParameter(
        description=":str: The directory containing the morphologies and their metadata file.",
        exists=True,
        default=None,
    )
    result_path = PathParameter(
        default="placeholders.csv",
        description=":str: Path to the output file (can be CSV or JSON file).",
    )
    nb_jobs = NumericalParameter(
        var_type=int,
        min_value=1,
        max_value=10**10,
        default=1,
        description=":int: Number of jobs used by parallel tasks.",
    )

    def requires(self):
        if self.input_dir is None:  # pragma: no cover
            return Fetch()
        return None

    def run(self):
        if self.output().pathlib_path.suffix not in [".csv", ".json"]:
            raise ValueError(  # noqa: TRY003
                "The 'result_path' parameter should have a '.csv' or '.json' extension, not "
                f"'{self.output().pathlib_path.suffix}'."
            )
        input_dir = self.input_dir or self.input()["morphologies"]
        if self.config is not None:
            with self.config.open() as f:  # pylint: disable=no-member
                config = json.load(f)
        else:
            config = None
        df = compute_placeholders(input_dir, config, self.nb_jobs)

        if self.output().pathlib_path.suffix == ".csv":
            df.to_csv(self.output().path, index=False)
        elif self.output().pathlib_path.suffix == ".json":
            with self.output().pathlib_path.open("w") as f:
                json.dump(placeholders_to_nested_dict(df), f, indent=4, sort_keys=True)

    def output(self):
        return TaggedOutputLocalTarget(
            self.result_path,
            create_parent=True,
        )
