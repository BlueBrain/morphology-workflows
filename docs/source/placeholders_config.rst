Placeholders config file
========================

Schema
~~~~~~

The config passed to the ``Placeholder`` workflow should follow the following schema:

.. jsonschema:: ../../src/morphology_workflows/_templates/placeholder_config_schema.json
    :lift_definitions:
    :auto_reference:
    :auto_target:

Here is an example of a simple configuration file:


Example
~~~~~~~

.. code-block:: JSON
   :linenos:

    [
        {
            "populations": [
                {
                    "region": "Region example",
                    "mtype": "Mtype example",
                    "name": "Population example"
                }
            ],
            "config": {
                "neurite": {
                    "aspect_ratio": {
                        "kwargs": [
                            {"projection_plane": "xy"},
                        ],
                        "modes": ["mean"],
                    },
                    "length_fraction_above_soma": {
                        "kwargs": [{"up": "X"}, {"up": "Y"}, {"up": "Z"}],
                        "modes": ["mean"],
                    },
                    "total_height": {"modes": ["max"]},
                    "total_length": {"modes": ["sum"]},
                },
                "morphology": {
                    "soma_radius": {"modes": ["mean"]},
                    "soma_surface_area": {"modes": ["mean"]}
                },
                "neurite_type": ["AXON", "APICAL_DENDRITE", "BASAL_DENDRITE"]
            }
        }
    ]
