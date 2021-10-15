# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	# Generate dependency graphs
	cd ../tests/data/test_example_1; morphology_workflows -dg ../../../doc/source/autoapi/tasks/workflows/Curate.dot Curate ; cd ../../../doc
	cd ../tests/data/test_example_1; morphology_workflows -dg ../../../doc/source/autoapi/tasks/workflows/Annotate.dot Annotate; cd ../../../doc
	cd ../tests/data/test_example_1; morphology_workflows -dg ../../../doc/source/autoapi/tasks/workflows/Repair.dot Repair; cd ../../../doc

	cd ../tests/data/test_example_1; morphology_workflows -dg ../../../doc/source/autoapi/tasks/workflows/Curate.png Curate ; cd ../../../doc
	cd ../tests/data/test_example_1; morphology_workflows -dg ../../../doc/source/autoapi/tasks/workflows/Annotate.png Annotate; cd ../../../doc
	cd ../tests/data/test_example_1; morphology_workflows -dg ../../../doc/source/autoapi/tasks/workflows/Repair.png Repair; cd ../../../doc

	# Generate the documentation
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
