Fetch configuration file
========================

The ``Fetch`` workflow aims at downloading morphology files from different source databases:

* `NeuroMorpho <https://neuromorpho.org>`_
* `MouseLight <https://ml-neuronbrowser.janelia.org>`_
* `Allen Brain <https://celltypes.brain-map.org>`_

At the end of the ``Fetch`` workflow, the result folder will contain all the morphology files and
a ``metadata.csv`` file which contains all the metadata of each morphology file.

Each database needs a different configuration file. In each case, the configuration file is a JSON
file containing a list of objects, each object being a configuration element that will be executed
independently. Because of this, two identical configuration elements will fetch the same
morphologies. The result folder will contain only one copy of each morphology file but the
``metadata.csv`` file will contain duplicated entries, so it is possible to understand why the
result folder does not contain the expected number of files and which filters should be updated.

Each configuration element can contain the following entries:

* ``nb_morphologies``: the number of morphologies to fetch using the filters of the current object.
* ``seed``: the random seed used to choose which elements are fetched among the ones available using
  the filters of the current object.
* all other entries should be valid for the requested source API.

This page shows examples of each type of configuration file.

NeuroMorpho
-----------

The filter entries should any valid filter entry for the
`NeuroMorpho API <https://neuromorpho.org/apiReference.html>`_.

.. literalinclude:: ../../src/morphology_workflows/_templates/neuromorpho_config.json
   :language: json

MouseLight
----------

The filter entries support only the following entries:

* ``brain_region``: the name of the brain region to filter.

.. literalinclude:: ../../src/morphology_workflows/_templates/mouselight_config.json
   :language: json

Allen Brain
-----------

The filter entries support only the following entries:

* ``species``: the name of the species to filter.
* ``brain_region``: the name of the brain region to filter.
* any other valid filter entry for the
  `AllenSDK API <http://alleninstitute.github.io/AllenSDK/cell_types.html>`_ (the filter keys
  should be chosen from the ones used to create
  `these entries <https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/api/queries/cell_types_api.py#L260>`_.

.. literalinclude:: ../../src/morphology_workflows/_templates/allen_config.json
   :language: json
