"""Placeholders functions."""
import logging
import warnings
from pathlib import Path
from typing import Optional

import neurom
import numpy as np
import pandas as pd
from neurom.apps import morph_stats
from pkg_resources import resource_filename

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "neurite": {
        "aspect_ratio": {
            "kwargs": [
                {"projection_plane": "xy"},
                {"projection_plane": "xz"},
                {"projection_plane": "yz"},
            ],
            "modes": ["mean"],
        },
        "circularity": {
            "kwargs": [
                {"projection_plane": "xy"},
                {"projection_plane": "xz"},
                {"projection_plane": "yz"},
            ],
            "modes": ["mean"],
        },
        "shape_factor": {
            "kwargs": [
                {"projection_plane": "xy"},
                {"projection_plane": "xz"},
                {"projection_plane": "yz"},
            ],
            "modes": ["mean"],
        },
        "length_fraction_above_soma": {
            "kwargs": [{"up": "X"}, {"up": "Y"}, {"up": "Z"}],
            "modes": ["mean"],
        },
        "volume_density": {"modes": ["mean"]},
        "principal_direction_extents": {"modes": ["max"]},
        "total_height": {"modes": ["max"]},
        "total_width": {"modes": ["max"]},
        "total_depth": {"modes": ["max"]},
        "total_length": {"modes": ["sum"]},
        "total_area": {"modes": ["sum"]},
    },
    "morphology": {"soma_radius": {"modes": ["mean"]}, "soma_surface_area": {"modes": ["mean"]}},
    "neurite_type": ["AXON", "APICAL_DENDRITE", "BASAL_DENDRITE"],
}


def select_population(
    input_morphologies: str, region: Optional[str], mtype: Optional[str]
) -> neurom.core.Population:
    """Compute the placeholder values for a given region - mtype couple."""
    logger.debug("Get population for %s in %s from %s", mtype, region, input_morphologies)

    # Load the morphologies for a selected region - mtype couple
    input_dir = Path(input_morphologies)
    metadata_path = input_dir / "metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if region is not None:
            region_mask = metadata["brain_region"] == region
        else:
            region_mask = np.ones(len(metadata), dtype=bool)
        if mtype is not None:
            mtype_mask = metadata["cell_type"] == mtype
        else:
            mtype_mask = np.ones(len(metadata), dtype=bool)
        metadata = metadata.loc[region_mask & mtype_mask]
        population = neurom.load_morphologies(input_dir / metadata["morphology"])
    else:
        warnings.warn("No metadata.csv file found in the input directory, loading all morphologies")
        population = neurom.load_morphologies(input_dir)

    return population


def compute_placeholders(
    input_morphologies: str,
    region: str = None,
    mtype: str = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute the placeholder values for a given region - mtype couple."""
    # Select morphologies
    population = select_population(input_morphologies, region, mtype)

    # Extract dataframe
    if config is None:
        config = DEFAULT_CONFIG

    if len(population) == 0:
        logger.debug(
            (
                "The population for the %s region and %s mtype is empty so the default "
                "placeholders are used."
            ),
            region,
            mtype,
        )
        df_placeholder = pd.read_csv(
            resource_filename(
                "morphology_workflows",
                "_data/default_placeholders.csv",
            ),
            header=[0, 1],
        )
    else:
        logger.debug("Compute placeholders with the following config: %s", config)
        df_placeholder = morph_stats.extract_dataframe(population, config)

    # Add region and mtype to the dataframe
    df_placeholder[("Metadata", "Region")] = region
    df_placeholder[("Metadata", "Mtype")] = mtype

    return df_placeholder
