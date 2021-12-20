"""Configuration file for the Sphinx documentation builder."""
#
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
from pathlib import Path

import luigi
from pkg_resources import get_distribution

import morphology_workflows
import morphology_workflows.tasks
from morphology_workflows.tasks import cli

# -- Project information -----------------------------------------------------

project = "morphology-workflows"

# The short X.Y version
version = get_distribution(project).version

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
    "m2r2",
]

autoapi_dirs = [
    "../../src/morphology_workflows",
    "../../src/morphology_workflows/tasks",
]
autoapi_ignore = [
    "*version.py",
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
    "metadata_distribution": project,
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
    "diameter-synthesis": ("https://diameter-synthesis.readthedocs.io/en/stable", None),
    "luigi": ("https://luigi.readthedocs.io/en/stable", None),
    "luigi-tools": ("https://luigi-tools.readthedocs.io/en/stable", None),
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


# pylint: disable=unused-argument
def maybe_skip_member(app, what, name, obj, skip, options):
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

        if isinstance(actual_obj, luigi.Parameter):
            if hasattr(actual_obj, "description") and actual_obj.description:
                obj.docstring = cli.format_description(
                    actual_obj,
                    default_str="{doc}\n\n:default value: {default}",
                    optional_str="(optional) {doc}",
                    type_str="{doc}\n\n:type: {type}",
                    choices_str="{doc}\n\n:choices: {choices}",
                    interval_str="{doc}\n\n:permitted values: {interval}",
                )

    return skip


def generate_images(*args, **kwargs):
    """Generate images of the workflows."""
    old_cwd = os.getcwd()
    try:
        os.environ["LUIGI_CONFIG_PATH"] = str(
            Path(*Path(__file__).parts[:-3]) / "tests/data/test_example_1/luigi.cfg"
        )
        cur_cwd = Path(__file__).parent
        cli.main(["-dg", str(cur_cwd / "autoapi/tasks/workflows/Curate.dot"), "Curate"])
        cli.main(["-dg", str(cur_cwd / "autoapi/tasks/workflows/Curate.png"), "Curate"])
        cli.main(["-dg", str(cur_cwd / "autoapi/tasks/workflows/Annotate.dot"), "Annotate"])
        cli.main(["-dg", str(cur_cwd / "autoapi/tasks/workflows/Annotate.png"), "Annotate"])
        cli.main(["-dg", str(cur_cwd / "autoapi/tasks/workflows/Repair.dot"), "Repair"])
        cli.main(["-dg", str(cur_cwd / "autoapi/tasks/workflows/Repair.png"), "Repair"])
    finally:
        os.chdir(old_cwd)


def setup(app):
    """Setup Sphinx by connecting functions to events."""
    app.connect("builder-inited", generate_images)
    app.connect("autoapi-skip-member", maybe_skip_member)
