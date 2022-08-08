#!/bin/bash

rm -rf out_* *_release-*

python get_dataset.py

morphology_workflows Curate
morphology_workflows Annotate
morphology_workflows Repair
