#!/bin/bash

rm -rf out_* *_release-*

morphology_workflows Initialize --source-database NeuroMorpho
morphology_workflows Fetch
morphology_workflows Curate
morphology_workflows Annotate
morphology_workflows Repair
