"""Module to generate exemplars from a population of morphologies."""
import numpy as np

from morphio.mut import Morphology
import neurom as nm
from morph_tool.morphdb import MorphDB


def single_compartment_exemplar(morph_df, neurite_fraction=0.2):
    """We create a single compartment morphology representative of the perisomatic surface area.

    The effective radius is computed as a weighted sum of the radiuses of the different sections.

    Adapted from /gpfs/bbp.cscs.ch/project/proj136/placeholder_singlecell/optimisation_local_luigi_OLD
    """

    db = MorphDB()
    db.df = morph_df
    config = {
        "neurite": {"total_area": {"modes": ["sum"]}},
        "morphology": {"soma_surface_area": {"modes": ["mean"]}},
        "neurite_type": ["AXON", "APICAL_DENDRITE", "BASAL_DENDRITE"],
    }

    df_features = db.features(config)
    radius = np.sqrt(
        (
            df_features["morphology"]["mean_soma_surface_area"]
            + (
                df_features["basal_dendrite"]["sum_total_area"]
                + df_features["apical_dendrite"]["sum_total_area"]
                + df_features["axon"]["sum_total_area"]
            )
            * neurite_fraction
        )
        / (4.0 * np.pi)
    ).mean()

    m = Morphology()
    m.soma.points = [[-radius, 0, 0], [0, 0, 0], [radius, 0, 0]]
    m.soma.diameters = [2 * radius, 2 * radius, 2 * radius]
    return m


def full_exemplar():
    """We create a full exemplar from the population.

    TODO
    """
    pass
