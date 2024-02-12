# noqa: INP001
"""Configuration file for the Sphinx documentation builder."""

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import importlib
import os
import re
import subprocess
from importlib import metadata
from pathlib import Path

import luigi
from luigi_tools.util import luigi_config_to_dict
from luigi_tools.util import set_luigi_config
from sphinx.util import logging

import morphology_workflows
import morphology_workflows.tasks
from morphology_workflows.tasks import cli

logger = logging.getLogger(__name__)

# -- Project information -----------------------------------------------------

project = "morphology-workflows"

# The short X.Y version
version = None
if os.environ.get("READTHEDOCS_VERSION", "") == "stable":
    logger.info("Stable ReadTheDocs environment found.")
    os.environ["SPHINX_BLUE_BRAIN_THEME_CHECK_VERSIONS"] = "False"
    with subprocess.Popen(
        ["git", "describe", "--abbrev=0", "--tags"],
        stdout=subprocess.PIPE,
    ) as proc:
        version = proc.communicate()[0].decode("utf-8").strip()
    if not version:
        logger.info(
            "Could not find version from the 'git describe' command ."
            "The 'importlib.metadata.version' function will be used."
        )

if not version:
    version = metadata.version("morphology-workflows")

logger.info("Version found: %s", version)

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinxarg.ext",
    "sphinx-jsonschema",
    "m2r2",
]

autoapi_dirs = [
    "../../src/morphology_workflows",
    "../../src/morphology_workflows/tasks",
]
autoapi_ignore = [
    "*_templates/**",
]
autoapi_python_use_implicit_namespaces = True
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_options = [
    "imported-members",
    "members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "undoc-members",
]
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx-bluebrain-theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_theme_options = {
    "metadata_distribution": "morphology-workflows",
}

html_title = "Morphology Workflows"

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# autosummary settings
autosummary_generate = True

# autodoc settings
autodoc_typehints = "signature"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {
    # TODO: Uncomment this once it is published on RTD
    # "data-validation-framework": (
    #     "https://data-validation-framework.readthedocs.io/en/stable",
    #     None,
    # ),
    "data-validation-framework": (
        "https://data-validation-framework.readthedocs.io/en/stable",
        None,
    ),
    "diameter-synthesis": ("https://diameter-synthesis.readthedocs.io/en/stable", None),
    "luigi": ("https://luigi.readthedocs.io/en/stable", None),
    "luigi-tools": ("https://luigi-tools.readthedocs.io/en/stable", None),
    "morph-tool": ("https://morph-tool.readthedocs.io/en/stable", None),
    "neurom": ("https://neurom.readthedocs.io/en/stable", None),
    "neuror": ("https://neuror.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "python": ("https://docs.python.org/3", None),
}

SKIP = [
    r".*\.L$",
    r".*tasks\..*\.__specifications__$",
    r".*tasks\..*\.args$",
    r".*tasks\..*\.input_index_col$",
    r".*tasks\..*\.inputs$",
    r".*tasks\..*\.kwargs$",
    r".*tasks\..*\.output$",
    r".*tasks\..*\.output_columns$",
    r".*tasks\..*\.requires$",
    r".*tasks\..*\.run$",
    r".*tasks\..*\.validation_function$",
]

IMPORT_MAPPING = {
    "morphology_workflows": morphology_workflows,
    "tasks": morphology_workflows.tasks,
}


# pylint: disable=unused-argument
def maybe_skip_member(app, what, name, obj, skip, options):  # noqa: ARG001
    """Skip useless members and format documentation of others."""
    skip = None
    for pattern in SKIP:
        if re.match(pattern, name) is not None:
            skip = True
            break

    if not skip:
        if name.count(".") < 3:
            return skip
        package, module, *path = name.split(".")
        root_package = IMPORT_MAPPING[package]
        actual_module = importlib.import_module(root_package.__name__ + "." + module)

        try:
            task = getattr(actual_module, path[-2])
            actual_obj = getattr(task, path[-1])
        except AttributeError:
            return skip

        if (
            isinstance(actual_obj, luigi.Parameter)
            and hasattr(actual_obj, "description")
            and actual_obj.description
        ):
            obj.docstring = cli.format_description(
                actual_obj,
                default_str="{doc}\n\n:default value: {default}",
                optional_str="(Optional) {doc}",
                type_str="{doc}\n\n:type: {type}",
                choices_str="{doc}\n\n:choices: {choices}",
                interval_str="{doc}\n\n:permitted values: {interval}",
            )

    return skip


def generate_images(*args, **kwargs):  # noqa: ARG001
    """Generate images of the workflows."""
    input_dir = Path(*Path(__file__).parts[:-3]) / "src/morphology_workflows/_templates"

    # Import luigi configuration
    dict_config = luigi_config_to_dict(input_dir / "luigi.cfg")

    # Update dataset_df values just to point to existing files
    dict_config["Curate"]["dataset_df"] = str(input_dir / "luigi.cfg")
    dict_config["Annotate"]["dataset_df"] = str(input_dir / "luigi.cfg")
    dict_config["Repair"]["dataset_df"] = str(input_dir / "luigi.cfg")

    # Export dependency graphs
    cur_cwd = Path(__file__).parent
    with set_luigi_config(dict_config):
        cli.main(
            ["-dg", str(cur_cwd / "autoapi/tasks/workflows/Curate.dot"), "-dgdpi", "100", "Curate"]
        )
        cli.main(
            ["-dg", str(cur_cwd / "autoapi/tasks/workflows/Curate.png"), "-dgdpi", "100", "Curate"]
        )
        cli.main(
            [
                "-dg",
                str(cur_cwd / "autoapi/tasks/workflows/Annotate.dot"),
                "-dgdpi",
                "100",
                "Annotate",
            ]
        )
        cli.main(
            [
                "-dg",
                str(cur_cwd / "autoapi/tasks/workflows/Annotate.png"),
                "-dgdpi",
                "100",
                "Annotate",
            ]
        )
        cli.main(
            ["-dg", str(cur_cwd / "autoapi/tasks/workflows/Repair.dot"), "-dgdpi", "100", "Repair"]
        )
        cli.main(
            ["-dg", str(cur_cwd / "autoapi/tasks/workflows/Repair.png"), "-dgdpi", "100", "Repair"]
        )


def setup(app):
    """Setup Sphinx by connecting functions to events."""
    app.connect("builder-inited", generate_images)
    app.connect("autoapi-skip-member", maybe_skip_member)
