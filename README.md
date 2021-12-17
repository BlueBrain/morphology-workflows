[![Version](https://img.shields.io/pypi/v/morphology-workflows)](https://github.com/BlueBrain/morphology-workflows/releases)
[![Build status](https://github.com/BlueBrain/morphology-workflows/actions/workflows/run-tox.yml/badge.svg?branch=main)](https://github.com/BlueBrain/morphology-workflows/actions)
[![Coverage](https://codecov.io/github/BlueBrain/morphology-workflows/coverage.svg?branch=main)](https://codecov.io/github/BlueBrain/morphology-workflows?branch=main)
[![License](https://img.shields.io/badge/License-Apache%202-blue)](https://github.com/BlueBrain/morphology-workflows/blob/main/LICENSE.txt)
[![Documentation status](https://readthedocs.org/projects/morphology-workflows/badge/?version=latest)](https://morphology-workflows.readthedocs.io/)


# Morphology Workflows

This project contains several workflows for processing morphologies:
- **Curate:** from raw morphologies, ensures that morphologies can be used with the rest of
  BBP codes
  [![](autoapi/tasks/workflows/Curate.png)](autoapi/tasks/workflows/index.html#tasks.workflows.Curate)
- **Annotate:** create various annotations on morphologies needed by specific BBP codes
  [![](autoapi/tasks/workflows/Annotate.png)](autoapi/tasks/workflows/index.html#tasks.workflows.Annotate)
- **Repair:** process morphologies to correct for artifacts of in-vitro reconstruction
  [![](autoapi/tasks/workflows/Repair.png)](autoapi/tasks/workflows/index.html#tasks.workflows.Repair)


In a nutshell, the user provides a list of morphologies in a ``.csv`` file, with their names and
paths and a ``luigi.cfg`` configuration file. Each workflow is run independently and creates an
output folder, with one subfolder per task. In each, there will be a ``report.csv`` and a ``data``
folder containing the output files of the task if any. In the ``report.csv`` file, columns contain
paths to these files, additional information, error messages if the task failed on that
morphologies, as well as a flag ``is_valid``, used in subsequent tasks to filter valid morphologies.
At the end of each workflow, another ``report.csv`` file is created, with the main output columns of
each tasks, and a ``report.pdf`` containing a human readable summary of the result of the workflow.


## Installation

This should be installed using pip:

```bash
pip install morphology-workflows
```


## Usage

This workflow is based on the ``luigi`` library but can be run via the command line interface:

```bash
morphology_workflows --local-scheduler Curate
```

> **NOTE** This command must be executed from a directory containing a ``luigi.cfg`` file.
> An example of such file is given in the ``examples`` directory.

More details can be found in the command line interface section of the documentation.


## Examples

The `examples` folder contains a simple example that will process a set of morphologies.
A ``dataset.csv`` file is provided which is taken as input for the workflows. A ``luigi.cfg`` file
is also provided to give a default configuration for the workflows.
This example can simply be run using the following command:

```bash
./run_curation.sh
```

This script will create a new directory ``out_curated`` which will contain the report and all the
results.


## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research
center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board
of the Swiss Federal Institutes of Technology.

For license and authors, see `LICENSE.txt` and `AUTHORS.md` respectively.

Copyright © 2021 Blue Brain Project/EPFL
