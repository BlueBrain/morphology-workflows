"""Tasks to fetch morphologies."""
import copy
import json
import logging

import luigi
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from data_validation_framework.task import TagResultOutputMixin
from luigi.parameter import PathParameter
from luigi_tools.task import WorkflowTask
from morphapi.api.mouselight import MouseLightAPI
from morphapi.api.neuromorphorg import NeuroMorpOrgAPI

from morphology_workflows.utils import silent_logger

logger = logging.getLogger(__name__)


class Fetch(TagResultOutputMixin, WorkflowTask):
    """Fetch morphologies from the given source.

    The JSON configuration file should contain a list of objects where each object is a config set::

        [
            {
                "brain_region": "AHN",
                "nb_morphologies": 2,
                "seed": 0
            },
            {
                "brain_region": "VISp",
                "nb_morphologies": 2,
                "seed": 0
            }
        ]
    """

    source = luigi.ChoiceParameter(
        description=":str: The source used to download the morphologies",
        choices=["Allen", "BBP-Nexus", "NeuroMorpho", "MouseLight"],
    )
    config_file = PathParameter(
        description=":str: Path to the JSON config file",
        exists=True,
    )
    result_path = PathParameter(
        default="morphologies", description=":str: Path to the output directory."
    )

    @staticmethod
    def _neuron_paths(neurons, root_path):
        return [i.data_file.relative_to(root_path).as_posix() for i in neurons]

    def neuromorpho_download(self, config):
        """Download morphologies from the NeuromMorpho.org database."""
        api = NeuroMorpOrgAPI()
        api.neuromorphorg_cache = self.output()["morphologies"].pathlib_path
        api.neuromorphorg_cache.mkdir(parents=True, exist_ok=True)

        for conf_element in config:
            downloaded_neurons = []
            page = 0
            remaining = conf_element.get("nb_morphologies", float("inf"))

            while remaining > 0:
                size = min(500, remaining)  # Can get the metadata for up to 500 neurons at a time
                remaining = remaining - size
                logger.debug("Downloading page %s for: %s", page, conf_element)
                try:
                    metadata, total = api.get_neurons_metadata(
                        size=size,
                        page=page,
                        species=conf_element.get("species", None),
                        cell_type=conf_element.get("cell_type", None),
                        brain_region=conf_element.get("brain_region", None),
                    )

                    logger.debug(
                        "Found %s morphologies to download for %s", len(metadata), conf_element
                    )

                    # Download these neurons
                    downloaded_neurons.extend(
                        self._neuron_paths(
                            api.download_neurons(metadata, load_neurons=False),
                            api.neuromorphorg_cache,
                        )
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error(
                        "Could not download the morphologies for %s for the following reason: %s",
                        conf_element,
                        str(exc),
                    )
                    break

                page += 1
                if page >= total["totalPages"]:
                    break

            conf_element["morphologies"] = downloaded_neurons

    def mouselight_download(self, config):
        """Download morphologies from the MouseLight database."""
        try:
            # pylint: disable=import-outside-toplevel
            from bg_atlasapi import BrainGlobeAtlas
        except ImportError as exc:
            raise ImportError(
                'You need to install the "bg_atlasapi" package to fetch morphologies from the '
                'MouseLight database: "pip install bg_atlasapi"'
            ) from exc
        api = MouseLightAPI()
        api.mouselight_cache = self.output()["morphologies"].pathlib_path
        api.mouselight_cache.mkdir(parents=True, exist_ok=True)
        full_metadata = api.fetch_neurons_metadata()
        atlas = BrainGlobeAtlas("allen_mouse_25um")

        for conf_element in config:
            rng = np.random.default_rng(conf_element.get("seed", None))
            brain_region = conf_element.get("brain_region", None)
            if brain_region is not None:
                brain_region = [brain_region]
            try:
                neurons_metadata = api.filter_neurons_metadata(
                    full_metadata, filterby="soma", filter_regions=brain_region, atlas=atlas
                )
                sampled_metadata = rng.choice(
                    neurons_metadata,
                    size=min(
                        len(neurons_metadata), conf_element.get("nb_morphologies", float("inf"))
                    ),
                    replace=False,
                ).tolist()
                downloaded_neurons = api.download_neurons(sampled_metadata)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Could not download the morphologies for %s for the following reason: %s",
                    conf_element,
                    str(exc),
                )
                downloaded_neurons = []

            conf_element["morphologies"] = self._neuron_paths(
                downloaded_neurons, api.mouselight_cache
            )

    @silent_logger("allensdk.api.api")
    @silent_logger("allensdk.api.api.retrieve_file_over_http")
    def allen_download(self, config):
        """Download morphologies from the Allen database."""
        try:
            # pylint: disable=import-outside-toplevel
            from morphapi.api.allenmorphology import AllenMorphology
        except ImportError as exc:
            raise ImportError(
                'You need to install the "allensdk" package to fetch morphologies from the '
                'Allen Brain database: "pip install allensdk"'
            ) from exc
        api = AllenMorphology()
        api.allen_morphology_cache = self.output()["morphologies"].pathlib_path
        api.allen_morphology_cache.mkdir(parents=True, exist_ok=True)

        for conf_element in config:
            size = conf_element.get("nb_morphologies", float("inf"))
            species = conf_element.get("species", None)
            brain_region = conf_element.get("brain_region", None)
            # cell_type = conf_element.get("cell_type", None)

            mask = np.full(len(api.neurons), True, dtype=bool)
            if species is not None:
                mask = mask & (api.neurons.species == species)
            if brain_region is not None:
                mask = mask & (api.neurons.structure_area_abbrev == brain_region)
            # if cell_type is not None:
            #     mask = mask & (api.neurons.structure_area_abbrev == region)

            neurons = api.neurons.loc[mask]

            logger.debug("Found %s morphologies to download for %s", len(neurons), conf_element)

            if len(neurons) > 0:
                # Download some neurons
                downloaded_neurons = self._neuron_paths(
                    api.download_neurons(
                        neurons.sample(
                            min(len(neurons), size), random_state=conf_element.get("seed", None)
                        ).id.values,
                        load_neurons=False,
                    ),
                    api.allen_morphology_cache,
                )
            else:
                logger.warning("No data available with the given constraints: %s", conf_element)
                downloaded_neurons = []

            conf_element["morphologies"] = downloaded_neurons

    def run(self):
        with self.config_file.open() as f:  # pylint: disable=no-member
            config = json.load(f)

        if self.source == "Allen":
            self.allen_download(config)
        elif self.source == "BBP-Nexus":
            raise NotImplementedError("Fetching morphologies from Nexus is not implemented yet.")
        elif self.source == "MouseLight":
            self.mouselight_download(config)
        elif self.source == "NeuroMorpho":
            self.neuromorpho_download(config)

        formatted_result = []
        for element in config:
            params = {k: v for k, v in element.items() if k != "morphologies"}
            for morph_name in element.get("morphologies", []):
                entry = copy.copy(params)
                entry["morphology"] = morph_name
                formatted_result.append(entry)

        df = pd.DataFrame(formatted_result)
        df.to_csv(self.output()["metadata"].path, index=False)

    def output(self):
        return {
            "morphologies": TaggedOutputLocalTarget(
                self.result_path,
                create_parent=True,
            ),
            "metadata": TaggedOutputLocalTarget(
                self.result_path / "metadata.csv",
                create_parent=True,
            ),
        }
