#!/bin/bash

rm -rf out_*

python get_dataset.py

morphology_workflows --local-scheduler Curate
morphology_workflows --local-scheduler Annotate
morphology_workflows --local-scheduler Repair
