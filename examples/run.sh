#!/bin/bash

rm -rf out_* *_release-*

morphology-workflows Initialize --source-database NeuroMorpho
morphology-workflows Fetch
morphology-workflows Curate
morphology-workflows Annotate
morphology-workflows Repair
