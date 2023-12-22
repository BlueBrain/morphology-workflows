"""CLI for validation workflows."""
import argparse
import inspect
import logging
import os
import re
import sys
import textwrap
from pathlib import Path

import luigi
from luigi_tools.util import get_dependency_graph
from luigi_tools.util import graphviz_dependency_graph
from luigi_tools.util import render_dependency_graph

import morphology_workflows
from morphology_workflows.tasks import workflows
from morphology_workflows.tasks.fetch import Fetch
from morphology_workflows.tasks.placeholders import Placeholders
from morphology_workflows.utils import _TEMPLATES
from morphology_workflows.utils import create_inputs

L = logging.getLogger(__name__)

WORKFLOW_TASKS = {
    "Fetch": Fetch,
    "Placeholders": Placeholders,
    "Curate": workflows.Curate,
    "Annotate": workflows.Annotate,
    "Repair": workflows.Repair,
}

LOGGING_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

LUIGI_PARAMETERS = ["workers", "log_level"]


_PARAM_NO_VALUE = [
    luigi.parameter._no_value,  # pylint: disable=protected-access  # noqa: SLF001
    None,
]


def _process_param(param):
    desc = param.description
    choices = None
    interval = None
    optional = False
    if isinstance(param, luigi.OptionalParameter):
        optional = True
    if isinstance(param, luigi.ChoiceParameter):
        desc, choices = desc.rsplit("Choices: ", 1)
    if isinstance(param, luigi.NumericalParameter):
        desc, interval = desc.rsplit("permitted values: ", 1)
    try:
        param_type, param_doc = re.match("(:.*?:)? *(.*)", desc).groups()
    except AttributeError:
        param_type = None
        param_doc = desc
    return param_doc, param_type, choices, interval, optional


def format_description(
    param,
    default_str="{doc} Default value: {default}.",
    optional_str="(optional) {doc}",
    type_str="({type}) {doc}",
    choices_str="{doc} Choices: {choices}.",
    interval_str="{doc} Permitted values: {interval}.",
    param_no_value=None,
):
    """Format the description of a parameter."""
    if param_no_value is None:
        param_no_value = _PARAM_NO_VALUE

    try:
        param_doc, param_type, choices, interval, optional = _process_param(param)
        if optional:
            param_doc = optional_str.format(doc=param_doc)
        if param_type is not None:
            param_doc = type_str.format(doc=param_doc, type=param_type.replace(":", ""))
        if choices is not None:
            param_doc = choices_str.format(doc=param_doc, choices=choices)
        if interval is not None:
            param_doc = interval_str.format(doc=param_doc, interval=interval)
        # pylint: disable=protected-access
        if hasattr(param, "_default") and param._default not in param_no_value:  # noqa: SLF001
            param_doc = default_str.format(doc=param_doc, default=param._default)  # noqa: SLF001
    except AttributeError:
        param_doc = param.description
    return param_doc


class ArgParser:
    """Class to build parser and parse arguments."""

    def __init__(self):
        self.parsers = self._get_parsers()

    @property
    def parser(self):
        """Return the root parser."""
        return self.parsers["root"]

    def _get_parsers(self):
        """Return the main argument parser."""
        parser = argparse.ArgumentParser(
            description="Run the workflow",
        )
        parser.add_argument(
            "--version",
            action="version",
            version=f"%(prog)s, version {morphology_workflows.__version__}",
        )

        parser.add_argument("-c", "--config-path", help="Path to the Luigi config file.")

        parser.add_argument(
            "-m",
            "--master-scheduler",
            default=False,
            action="store_true",
            help="Use Luigi's master scheduler instead of local scheduler.",
        )

        parser.add_argument(
            "-ll",
            "--log-level",
            default=None,
            choices=LOGGING_LEVELS,
            help="Logger level (this will ignore the luigi 'logging_conf_file' argument).",
        )

        parser.add_argument("-lf", "--log-file", help="Logger file.")

        parser.add_argument(
            "-w",
            "--workers",
            type=int,
            default=1,
            help="Number of workers that luigi can summon.",
        )

        parser.add_argument(
            "-dg",
            "--create-dependency-graph",
            help=(
                "Create the dependency graph of a workflow instead of running it. "
                "Pass a path to render it as an image (depending on the extension of the given "
                "path)."
            ),
        )

        parser.add_argument(
            "-dgdpi",
            "--dependency-graph-dpi",
            help="The DPI used for the dependency graph export.",
        )

        return self._get_workflow_parsers(parser)

    @staticmethod
    def _get_workflow_parsers(parser=None):
        """Return the workflow argument parser.

        If parser is None, a new parser is created with the workflows as subparsers,
        otherwise if it is supplied, the parsers are added as subparsers.

        For each task listed in WORKFLOW_TASKS, a subparser is created as if it was
        created by luigi.
        """
        if not parser:
            parser = argparse.ArgumentParser()

        parsers = {"root": parser}

        workflow_parser = parser.add_subparsers(help="Possible workflows", dest="workflow")

        init_subparser = workflow_parser.add_parser(
            "Initialize",
            help="Create default inputs for curation workflows.",
            description=textwrap.dedent(
                """
                Create default inputs for a given workflow that users can then update according to
                their needs.
                Usually, the initialization consists in one of these two types:

                * the morphologies are fetched from an online database.
                * the morphologies are provided by the user.

                In the first case, the '--input-dir' argument should usually not be used.
                In the second case, the '--source-database' argument should usually not be used.
                """
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        init_subparser.add_argument(
            "--source-database",
            help="The database from which the morphologies will be fetched.",
            choices=Fetch.source._choices,  # pylint: disable=protected-access  # noqa: SLF001
        )
        init_subparser.add_argument(
            "--input-dir",
            help=(
                "The directory containing the input morphologies if they are not fetched from a "
                "database."
            ),
        )
        init_subparser.add_argument(
            "--output-dir", help="The directory in which the project inputs will be exported."
        )
        init_subparser.add_argument(
            "--dataset-filename",
            default="dataset.csv",
            help="The name of the CSV file to which the dataset will be exported.",
        )

        for workflow_name, task in WORKFLOW_TASKS.items():
            try:
                task_name = task.__name__
                doc = task.__doc__
                if ".. graphviz::" in doc:
                    doc = re.sub(
                        (
                            "The complete phase has the following dependency graph:"
                            r"\n\n    .. graphviz:: .*\.dot"
                        ),
                        "",
                        doc,
                        flags=re.DOTALL,
                    ).strip()
                subparser = workflow_parser.add_parser(workflow_name, help=doc, description=doc)
                for param, param_obj in task.get_params():
                    param_name = "--" + param.replace("_", "-")
                    subparser.add_argument(
                        param_name,
                        help=format_description(param_obj),
                        # pylint: disable=protected-access
                        **param_obj._parser_kwargs(param, task_name),  # noqa: SLF001
                    )
                parsers[workflow_name] = subparser
            except (AttributeError, TypeError):  # noqa: PERF203
                pass

        return parsers

    def parse_args(self, argv):
        """Parse the arguments, and return a argparse.Namespace object."""
        return self.parser.parse_args(argv)


def _build_parser():
    """Build the parser."""
    return ArgParser().parser


def export_dependency_graph(task, output_file, dpi=None):
    """Export the dependency graph of the given task."""
    g = get_dependency_graph(task, allow_orphans=True)

    # Create URLs
    base_f = Path(inspect.getfile(morphology_workflows)).parent
    node_kwargs = {}
    for _, child in g:
        if child is None:
            continue
        url = (
            Path(inspect.getfile(child.__class__)).relative_to(base_f).with_suffix("")
            / "index.html"
        )
        anchor = "#" + ".".join(child.__module__.split(".")[1:] + [child.__class__.__name__])
        node_kwargs[child] = {"URL": "../../" + url.as_posix() + anchor}

    graph_attrs = {}
    if dpi is not None:
        graph_attrs["dpi"] = dpi
    dot = graphviz_dependency_graph(g, node_kwargs=node_kwargs, graph_attrs=graph_attrs)
    render_dependency_graph(dot, output_file)


def main(arguments=None):
    """Main function."""
    # Setup logging
    luigi_config = luigi.configuration.get_config()
    logging_conf = luigi_config.get("core", "logging_conf_file", None)
    if logging_conf is not None and not Path(logging_conf).exists():
        L.warning(
            "The core->logging_conf_file entry is not a valid path so the default logging "
            "configuration is taken."
        )
        logging_conf = None
    if logging_conf is None:
        logging_conf = str(_TEMPLATES / "logging.conf")
        luigi_config.set("core", "logging_conf_file", logging_conf)
    logging.config.fileConfig(str(logging_conf), disable_existing_loggers=False)

    # Parse arguments
    if arguments is None:
        arguments = sys.argv[1:]
    parser = ArgParser()
    args = parser.parse_args(arguments)

    if args.log_level is not None:
        logging.config.dictConfig(
            {
                "version": 1,
                "incremental": True,
                "loggers": {
                    "": {  # root logger
                        "level": args.log_level,
                    },
                    "luigi": {"propagate": False},
                    "luigi_interface": {"propagate": False},
                },
            },
        )

    logging.getLogger("luigi").propagate = False
    logging.getLogger("luigi-interface").propagate = False

    logger = logging.getLogger(__name__)

    logger.debug("Args: %s", args)

    # Check that one workflow is in arguments
    if args is None or args.workflow is None:
        logger.critical("Arguments must contain one workflow. Check help with -h/--help argument.")
        parser.parser.print_help()
        return

    if args.workflow == "Initialize":
        create_inputs(
            source_db=args.source_database,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dataset_filename=args.dataset_filename,
        )
        return

    # Set luigi.cfg path
    if args.config_path is not None:
        os.environ["LUIGI_CONFIG_PATH"] = args.config_path

    # Get arguments to configure luigi
    env_params = {k: v for k, v in vars(args).items() if k in LUIGI_PARAMETERS}
    env_params["local_scheduler"] = not args.master_scheduler
    if args.log_level is not None:
        env_params["logging_conf_file"] = ""
        luigi_config.set("core", "no_configure_logging", "true")
    else:
        env_params.pop("log_level")

    # Prepare workflow task and arguments
    task_cls = WORKFLOW_TASKS[args.workflow]
    args_dict = {k.split(task_cls.get_task_family() + "_")[-1]: v for k, v in vars(args).items()}
    task_params = [i for i, j in task_cls.get_params()]
    args_dict = {k: v for k, v in args_dict.items() if v is not None and k in task_params}
    task = WORKFLOW_TASKS[args.workflow](**args_dict)

    # Export the dependency graph of the workflow instead of running it
    if args.create_dependency_graph is not None:
        export_dependency_graph(task, args.create_dependency_graph, dpi=args.dependency_graph_dpi)
        return

    # Run the luigi task
    logger.debug("Running the workflow using the following luigi config: %s", env_params)
    luigi.build([task], **env_params)


if __name__ == "__main__":
    main()
