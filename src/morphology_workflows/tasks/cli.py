"""CLI for validation workflows."""
import argparse
import inspect
import logging
import os
import re
import sys
from pathlib import Path

import luigi
from luigi_tools.util import get_dependency_graph
from luigi_tools.util import graphviz_dependency_graph
from luigi_tools.util import render_dependency_graph

import morphology_workflows
from morphology_workflows.tasks import workflows
from morphology_workflows.tasks.fetch import Fetch
from morphology_workflows.tasks.placeholders import Placeholders

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


_PARAM_NO_VALUE = [luigi.parameter._no_value, None]  # pylint: disable=protected-access


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
            param_doc = optional_str + param_doc
        if param_type is not None:
            param_doc = type_str.format(doc=param_doc, type=param_type.replace(":", ""))
        if choices is not None:
            param_doc = choices_str.format(doc=param_doc, choices=choices)
        if interval is not None:
            param_doc = interval_str.format(doc=param_doc, interval=interval)
        # pylint: disable=protected-access
        if hasattr(param, "_default") and param._default not in param_no_value:
            param_doc = default_str.format(doc=param_doc, default=param._default)
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

        parser.add_argument("-c", "--config-path", help="Path to the Luigi config file")

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
            default="INFO",
            choices=LOGGING_LEVELS,
            help="Logger level.",
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
                subparser = workflow_parser.add_parser(workflow_name, help=doc)
                for param, param_obj in task.get_params():
                    param_name = "--" + param.replace("_", "-")
                    subparser.add_argument(
                        param_name,
                        help=format_description(param_obj),
                        # pylint: disable=protected-access
                        **param_obj._parser_kwargs(param, task_name),
                    )
                parsers[workflow_name] = subparser
            except (AttributeError, TypeError):
                pass

        return parsers

    def parse_args(self, argv):
        """Parse the arguments, and return a argparse.Namespace object."""
        args = self.parser.parse_args(argv)

        return args


def _build_parser():
    """Build the parser."""
    tmp = ArgParser().parser
    return tmp


def main(arguments=None):
    """Main function."""
    if arguments is None:
        arguments = sys.argv[1:]

    # Parse arguments
    parser = ArgParser()
    args = parser.parse_args(arguments)

    L.debug("Args: %s", args)

    # Check that one workflow is in arguments
    if args is None or args.workflow is None:
        L.critical("Arguments must contain one workflow. Check help with -h/--help argument.")
        parser.parser.print_help()
        sys.exit()

    # Set luigi.cfg path
    if args.config_path is not None:
        os.environ["LUIGI_CONFIG_PATH"] = args.config_path

    # Get arguments to configure luigi
    luigi_config = {k: v for k, v in vars(args).items() if k in LUIGI_PARAMETERS}
    luigi_config["local_scheduler"] = not args.master_scheduler

    # Prepare workflow task and arguments
    task_cls = WORKFLOW_TASKS[args.workflow]
    args_dict = {k.split(task_cls.get_task_family() + "_")[-1]: v for k, v in vars(args).items()}
    task_params = [i for i, j in task_cls.get_params()]
    args_dict = {k: v for k, v in args_dict.items() if v is not None and k in task_params}
    task = WORKFLOW_TASKS[args.workflow](**args_dict)

    # Export the dependency graph of the workflow instead of running it
    if args.create_dependency_graph is not None:
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

        dot = graphviz_dependency_graph(g, node_kwargs=node_kwargs)
        render_dependency_graph(dot, args.create_dependency_graph)
        return

    # Run the luigi task
    luigi.build([task], **luigi_config)


if __name__ == "__main__":
    main()
