"""Placeholders tasks."""
import logging

from data_validation_framework.target import TaggedOutputLocalTarget
from data_validation_framework.task import TagResultOutputMixin
from luigi.parameter import OptionalPathParameter
from luigi.parameter import Parameter
from luigi.parameter import PathParameter
from luigi_tools.task import WorkflowTask

from morphology_workflows.placeholders import compute_placeholders
from morphology_workflows.tasks.fetch import Fetch

logger = logging.getLogger(__name__)


class Placeholders(TagResultOutputMixin, WorkflowTask):
    """Compute the place holders for a given region and mtype set."""

    region = Parameter(
        description=":str: The region to consider.",
    )
    mtype = Parameter(
        description=":str: The mtype to consider.",
    )
    config = OptionalPathParameter(
        description=":str: The path to the config file.",
        exists=True,
        default=None,
    )
    input_dir = OptionalPathParameter(
        description=":str: The directory containing the morphologies and their metadata file.",
        exists=True,
        default=None,
    )
    result_path = PathParameter(
        default="placeholders.csv", description=":str: Path to the output file."
    )

    def requires(self):
        if self.input_dir is None:
            return Fetch()
        return None

    def run(self):
        input_dir = self.input_dir or self.input()["morphologies"]
        df = compute_placeholders(input_dir, self.region, self.mtype, self.config)
        df.to_csv(self.output().path, index=False)

    def output(self):
        return TaggedOutputLocalTarget(
            self.result_path,
            create_parent=True,
        )
