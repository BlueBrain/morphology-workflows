"""Optional phase 4: Clones."""

import json
import logging
import math
import pprint
from collections import namedtuple
from collections import defaultdict
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import luigi
import pandas as pd
from bluepyparallel import init_parallel_factory
from data_validation_framework.result import ValidationResult
from data_validation_framework.task import ElementValidationTask
from data_validation_framework.task import SetValidationTask
from morph_tool.converter import convert
from morph_tool.exceptions import MorphToolException
from morph_tool.morphdb import MorphDB
from morph_tool.morphdb import MorphInfo
from morphology_workflows.curation import collect
from neurom.core import Morphology
from tqdm import tqdm

from morphology_workflows.clone import PlacementAnnotation
from morphology_workflows.clone import apply_scaling
from morphology_workflows.clone import collect_annotations
from morphology_workflows.clone import expand_mtype_to_layers
from morphology_workflows.clone import filter_missing_perimeter_morphs
from morphology_workflows.clone import get_category_overlap
from morphology_workflows.clone import get_cellcounts_from_recipe
from morphology_workflows.clone import graft_axons
from morphology_workflows.clone import make_clones
from morphology_workflows.clone import parse_morphdb_transform_rules
from morphology_workflows.clone import read_placement_rules
from morphology_workflows.utils import import_morph

logger = logging.getLogger(__name__)
Category = namedtuple("Category", "mtype layer")
NEURONDB_XML = Path("neuronDB.xml")


class CollectRepaired(ElementValidationTask):
    """Collect annotated dataset to work with on this phase."""

    output_columns = {
        "morph_path": None,
        "has_apical": None,
        "has_axon": None,
        "has_basal": None,
        "cut_leaves_path": None,
        "apical_point_path": None,
    }
    input_index_col = luigi.Parameter(default="morph_name")
    morph_path_col = luigi.Parameter(default="repair_release_morph_path_h5")

    def args(self):
        return [self.morph_path_col]

    validation_function = collect


class CollectAnnotations(ElementValidationTask):
    """Collect annotations to work with on this phase."""

    output_columns = {"annotation_path": None}
    input_index_col = luigi.Parameter(default="morph_name")
    annotation_path = luigi.OptionalParameter(
        description=":str: Path to the directory containing the annotations.",
        default=None,
    )

    def args(self):
        return [self.annotation_path]

    @staticmethod
    def validation_function(row, data_dir, annotation_path):  # pylint: disable=arguments-differ
        """Collect annotations."""
        path = Path(annotation_path or "", row.name + ".xml")
        if annotation_path is None or not path.exists():
            annotation = PlacementAnnotation(name=row.name)

            morph = Morphology(row.morph_path)
            annotation.calculate_axon_hard_limit(morph, "L1_axon_hard_limit")
            annotation.calculate_dendrite_hard_limit(morph, "L1_hard_limit")
            ret_code = 2
            comment = f"The annotation was not found for {row.name}. A new annotation is created."
        else:
            annotation = PlacementAnnotation.load(path)
            ret_code = 0
            comment = None

        annotation.save(data_dir)

        return ValidationResult(
            is_valid=True,
            annotation_path=annotation.path(data_dir),
            ret_code=ret_code,
            comment=comment,
        )


def get_candidates_from_neurondb(neurondb):
    """Returns a dict of 'Category' -> list of names."""
    candidates = defaultdict(list)
    for morph in neurondb:
        candidates[Category(morph.mtype, morph.layer)].append(morph.name)

    return dict(candidates)


class CloneMorphologies(SetValidationTask):
    """Create clones of morphologies.

    This task is divided in 4 steps:

    1. Graft axons
        Graft the axons from some morphologies to other morphologies based on the neurondb and
        the ``cross_mtypes`` parameter.

        At the end of the phase, discard morphologies who have the ``<use_axon>`` or
        ``<use_dendrites>`` flags set to False.

    2. Scale the morphologies
        This phase produces new version of each morphology by scaling them along the
        Y axis with the scaling factors given in the `y_scales` parameter.

    3. Transform neuronDB according to the exclusion and substitution rules
        The rules are applied with the following logic:

        - First the 'exclusions' are applied
        - Then the 'substitutions'
        - Finally the 'mtype_copy_to_layer', so that the later applies to the former.

    4. Create jittered versions of each morphologies and their corresponding annotations.
        Two kinds of jitter are applied:

        - rotational jitter: the section bifurcation angles are jittered with an std of 10 degrees
        - section length jitter: each section is scaled individually with an
          an std of 20% (ie. each section is stretched or compressed by around 20%)

        Procedure:

        1. For each ``Category[mtype, layer]``, determine the number of clones to be generated
           for this category. It depends on the total number of morphs in the circuit and
           the proportion of the category comparatively to other categories present in the circuit.
           This information is stored in builder_recipe.

        2. Each morphology file can be used by multiple ``category[mtype, layer]``.
           The number of clones to be generated for a given morphology file is the sum of the number
           of required clones for each category it belongs to.

        3. Produce the clones and adds an entry in self.output_db for each of them.

        4. If a placement_rules file is provided, also create an annotation file for each clones,
           else this step is skipped.

        .. note::

            - when a ``category[mtype, layer]`` appears in neurondb but not in ``builder_recipe``,
              we can not compute the number of clones required by this category, so its number is
              set to 0.
            - when a ``category[mtype, layer]`` appears in the builder_recipe but not in neurondb,
              it means there is no morphologies available for this category. So the category is
              skipped, but that can lead to problems later on in the pipeline.
            - different sub-mtypes are considered as different mtypes: they will have
              distinct clones.


    This task produces one directory for each step with:

    - the morphology files.
    - the neuronDB.xml and neuronDB.dat files.
    - an annotations folder with one annotation for each new morphology.
    - a lineage.json file.
    """

    cross_mtypes = luigi.ListParameter(
        description=":list: The list of mtypes to be grafted.",
        default=[
            "L2_IPC",
            "L4_UPC",
            "L4_SSC",
            "L5_UPC",
            "L6_UPC",
            "L6_BPC",
            "L6_HPC",
            "L3_TPC:B",
            "L5_TPC:A",
            "L5_TPC:B",
            "L5_TPC:C",
            "L6_TPC:A",
            "L6_TPC:C",
        ],
    )
    placement_rules_path = luigi.Parameter(
        description=":str: Path to the XML file describing the placement rules.",
        default="placement_rules.xml",
    )
    axon_limit_name = luigi.Parameter(
        description=":str: Name used from axon limit.",
        default="L1_axon_hard_limit",
    )
    dendrite_limit_name = luigi.Parameter(
        description=":str: Name used for dendrite limit.",
        default="L1_hard_limit",
    )
    y_scales = luigi.ListParameter(
        description=":list: Y scales to apply.", default=[0.95, 0.975, 1.025, 1.05]
    )
    transform_rules_path = luigi.Parameter(
        description=":str: Path to the XML file containing the transform rules.",
        default="transform_rules.xml",
    )
    builder_recipe_path = luigi.Parameter(
        description=":str: Path to the recipe file.",
        default="builder_recipe.xml",
    )
    clone_multiplier = luigi.NumericalParameter(
        description=":float: The multiplication factor to the number of generated clones.",
        min_value=0,
        max_value=float("inf"),
        var_type=float,
        left_op=luigi.parameter.operator.lt,
        default=1,
    )
    std_angle = luigi.NumericalParameter(
        description=":float: The std of the angles used to generate the clones.",
        min_value=0,
        max_value=float("inf"),
        var_type=float,
        left_op=luigi.parameter.operator.lt,
        default=10.0,
    )
    std_scale = luigi.NumericalParameter(
        description=":float: The std of the scales used to generate the clones.",
        min_value=0,
        max_value=float("inf"),
        var_type=float,
        left_op=luigi.parameter.operator.lt,
        default=0.2,
    )
    parallel_factory_type = luigi.ChoiceParameter(
        description=":str: The type of parallel factory used for computation.",
        default="multiprocessing",
        choices=["serial", "multiprocessing", "ipyparallel", "dask", "dask_dataframe"],
    )
    morph_path_col = luigi.Parameter(default="repair_morph_db_path_h5")

    def inputs(self):
        return {
            CollectRepaired: {"morph_path": "morph_path"},
            CollectAnnotations: {"annotation_path": "annotation_path"},
        }

    def args(self):
        return (
            self.cross_mtypes,
            self.placement_rules_path,
            self.axon_limit_name,
            self.dendrite_limit_name,
            self.y_scales,
            self.transform_rules_path,
            self.builder_recipe_path,
            self.clone_multiplier,
            self.std_angle,
            self.std_scale,
            self.parallel_factory_type,
            self.nb_processes,
            self.morph_path_col,
        )

    @staticmethod
    def _write_db(db, output_path):
        """Write the neuronDB as XML and DAT files."""
        db.write_lineage(output_path)
        path = output_path / NEURONDB_XML
        db.write(path)
        db.write(path.with_suffix(".dat"))
        return db, path

    @staticmethod
    def _graft_step(
        df,
        data_dir,
        cross_mtypes,
        placement_rules_path,
        axon_limit_name,
        dendrite_limit_name,
        morph_path_col,
    ):
        """Graft axons to morphologies."""
        # Remove morphologies with perimeters
        filter_missing_perimeter_morphs(df)
        print(df)

        valid_morphs = df.loc[df["is_valid"]]

        # Graft axons
        logger.info("Grafting axons for the following mtypes: %s", cross_mtypes)

        graft_dir = data_dir / "grafting"
        graft_dir.mkdir(parents=True, exist_ok=True)
        placement_rules = None
        if Path(placement_rules_path).exists():
            with open(placement_rules_path, encoding="utf-8") as fd:
                placement_rules = read_placement_rules(fd.read())

        input_db = MorphDB.from_neurondb(valid_morphs.iloc[0][morph_path_col])

        graft_db = graft_axons(
            valid_morphs,
            graft_dir,
            input_db,
            cross_mtypes,
            placement_rules,
            axon_limit_name,
            dendrite_limit_name,
        )

        # Add initial morphologies which have use_axon and use_dendrites to True
        graft_annotations_path = Path(graft_dir) / "annotations"
        graft_annotations_path.mkdir(parents=True, exist_ok=True)
        for _, morph_info in input_db.df.iterrows():
            if morph_info["use_axon"] and morph_info["use_dendrites"]:
                morph_name = morph_info["name"]
                morph_path, morph_annotation_path = df.loc[
                    morph_name, ["morph_path", "annotation_path"]
                ]
                graft_db.df = pd.concat([graft_db.df, morph_info])
                import_morph(
                    morph_path,
                    graft_dir / (morph_name + ".h5"),
                    morph_annotation_path,
                    graft_annotations_path,
                )

        graft_db.df.sort_values(by="name", inplace=True)
        CloneMorphologies._write_db(graft_db, graft_dir)
        return graft_db, graft_dir, placement_rules

    @staticmethod
    def _scale_step(data_dir, graft_db, graft_dir, y_scales, parallel_mapper):
        """Scale morphologies."""
        logger.info("Scaling morphologies with the following scales: %s", y_scales)

        scale_dir = data_dir / "scaling"
        scale_dir.mkdir(parents=True, exist_ok=True)
        scale_db = graft_db

        parents = defaultdict(list)
        for morph in scale_db:
            parents[morph.name].append(morph)

        names = list(scale_db.groupby("name").keys())
        annotation_path = graft_dir / "annotations"
        scale_annotation_path = scale_dir / "annotations"
        scale_annotation_path.mkdir(parents=True, exist_ok=True)

        for output_infos in parallel_mapper(
            apply_scaling,
            names,
            graft_dir,
            scale_dir,
            y_scales,
            parents,
            annotation_path,
            scale_annotation_path,
        ):
            for output_info in output_infos:
                scale_db.add_morph(output_info)

        scale_db.sort()
        CloneMorphologies._write_db(scale_db, scale_dir)
        return scale_db, scale_dir, scale_annotation_path

    @staticmethod
    def _transform_rules_step(data_dir, scale_db, transform_rules_path):
        """Transform rules."""
        logger.info("Transform rules from the following file: %s", transform_rules_path)

        transform_rules_dir = data_dir / "transform_rules"
        transform_rules_dir.mkdir(parents=True, exist_ok=True)
        transform_rules_db = scale_db

        rules = parse_morphdb_transform_rules(transform_rules_path)

        transform_rules_db.remove_morphs(rules.exclusions)

        def layer_expansion(name, mtype, layer):
            """Add morph to DB, respecting any layer copies that may be needed."""
            logger.debug("Add %s of mtype %s into the layer %s", name, mtype, layer)
            transform_rules_db.add_morph(MorphInfo(name=name, mtype=mtype, layer=layer))

            mtype_layer = (mtype, layer)
            for new_layer in rules.layer_copies.get(mtype_layer, []):
                logger.debug("Add %s of mtype %s into the layer %s", name, mtype, new_layer)
                transform_rules_db.add_morph(MorphInfo(name=name, mtype=mtype, layer=new_layer))

        for morph in transform_rules_db:
            # Note: in morphology-repair-workflow, the iteration was over the not filtered
            # morphologies but it was not consistent with the comments so I changed it here.

            layer_expansion(morph.name, morph.mtype, morph.layer)

            for expansion in rules.expansions.get(morph.mtype, []):
                for layer in expand_mtype_to_layers(expansion):
                    layer_expansion(morph.name, expansion, layer)

        transform_rules_db.sort()
        CloneMorphologies._write_db(transform_rules_db, transform_rules_dir)
        return transform_rules_db

    @staticmethod
    def _clone_step(
        data_dir,
        transform_rules_db,
        scale_dir,
        scale_annotation_path,
        placement_rules,
        builder_recipe_path,
        clone_multiplier,
        axon_limit_name,
        dendrite_limit_name,
        std_angle,
        std_scale,
        parallel_mapper,
    ):  # pylint: disable=too-many-arguments
        """Clone morphologies."""
        logger.info("Clone morphologies with the following recipe: %s", builder_recipe_path)

        clone_dir = data_dir / "clone"
        clone_dir.mkdir(parents=True, exist_ok=True)
        clone_db = transform_rules_db
        clone_annotations_path = clone_dir / "annotations"
        clone_annotations_path.mkdir(parents=True, exist_ok=True)

        count_mapping = get_cellcounts_from_recipe(builder_recipe_path)
        logger.info("Expected Cell Counts:\n%s", pprint.pformat(count_mapping))
        candidates = get_candidates_from_neurondb(clone_db)

        clone_counts = {
            category: int(
                math.ceil(clone_multiplier * count_mapping[category] / len(candidates[category]))
            )  # noqa
            for category in get_category_overlap(count_mapping, candidates)
        }

        items = clone_db.groupby("name").items()
        for output_infos in parallel_mapper(
            make_clones,
            items,
            clone_counts,
            axon_limit_name,
            dendrite_limit_name,
            scale_dir,
            clone_dir,
            scale_annotation_path,
            clone_annotations_path,
            placement_rules,
            std_angle,
            std_scale,
        ):
            for output_info in output_infos:
                clone_db.add_morph(output_info)

        CloneMorphologies._write_db(clone_db, clone_dir)
        return clone_annotations_path

    @staticmethod
    def _compact_annotations_step(annotations_path):
        """Compact the annotations into one JSON file."""
        annotations = collect_annotations(annotations_path)
        compact_annotations_path = annotations_path / "clone_annotations.json"

        with open(compact_annotations_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, sort_keys=True)

    @staticmethod
    def validation_function(
        df,
        data_dir,
        cross_mtypes,
        placement_rules_path,
        axon_limit_name,
        dendrite_limit_name,
        y_scales,
        transform_rules_path,
        builder_recipe_path,
        clone_multiplier,
        std_angle,
        std_scale,
        parallel_factory_type,
        nb_processes,
        morph_path_col,
    ):  # pylint: disable=arguments-differ, too-many-arguments, too-many-locals
        logger.warning(
            "THIS STEP WAS IMPORTED FROM morphology-repair-workflow BUT HAS NOT BEEN MUCH TESTED "
            "YET, BE CAREFUL WHEN YOU USE IT."
        )
        parallel_factory_kwargs = (
            {"processes": nb_processes} if parallel_factory_type == "multiprocessing" else {}
        )
        parallel_factory = init_parallel_factory(
            parallel_factory_type if parallel_factory_type != "serial" else None,
            **parallel_factory_kwargs,
        )
        parallel_mapper = parallel_factory.get_mapper()

        graft_db, graft_dir, placement_rules = CloneMorphologies._graft_step(
            df,
            data_dir,
            cross_mtypes,
            placement_rules_path,
            axon_limit_name,
            dendrite_limit_name,
            morph_path_col,
        )

        scale_db, scale_dir, scale_annotation_path = CloneMorphologies._scale_step(
            data_dir,
            graft_db,
            graft_dir,
            y_scales,
            parallel_mapper,
        )
        if Path(transform_rules_path).exists():
            transform_rules_db = CloneMorphologies._transform_rules_step(
                data_dir, scale_db, transform_rules_path
            )
        else:
            transform_rules_db = scale_db

        clone_annotations_path = CloneMorphologies._clone_step(
            data_dir,
            transform_rules_db,
            scale_dir,
            scale_annotation_path,
            placement_rules,
            builder_recipe_path,
            clone_multiplier,
            axon_limit_name,
            dendrite_limit_name,
            std_angle,
            std_scale,
            parallel_mapper,
        )

        CloneMorphologies._compact_annotations_step(clone_annotations_path)


def _get_layer_mtype(data):
    """Helper to get layer from mtype, if mtype exists as a str."""
    if "layer" in data:
        return data["mtype"], data["layer"]

    layer = "no_layer"
    mtype = "no_mtype"

    if isinstance(data["mtype"], str):
        mtype = data["mtype"]
        if len(data["mtype"]) > 1:
            layer = data["mtype"][1]
    return mtype, layer


def _convert(input_file, output_file):
    """Handles crashes in conversion of writing of morphologies."""
    try:
        logger.debug("Converting %s into %s", input_file, output_file)
        convert(input_file, output_file, nrn_order=True, sanitize=True)
    except MorphToolException as exc:
        return (
            f"Could not convert the file '{input_file}' into '{output_file}' because of the "
            f"following exception:\n{exc}"
        )
    except RuntimeError:  # This can happen if duplicates are being written at the same time
        pass
    return output_file


def _create_db_row(_data, clone_path, extension):
    """Create a db row and convert morphology."""
    index, data = _data
    mtype, layer = _get_layer_mtype(data)
    m = MorphInfo(name=data["name"], mtype=mtype, layer=layer, use_dendrite=True, use_axon=True)

    clone_release_path = str(clone_path / Path(data["morph_path"]).stem) + extension

    data[f"clone_release_morph_path_{extension[1:]}"] = _convert(
        data["morph_path"], clone_release_path
    )
    return index, data, m


class MakeCloneRelease(SetValidationTask):
    """Make a morpology release of clones."""

    clone_path = luigi.OptionalParameter(
        default="clone_release",
        description=":str: Path to clone morphologies (not created if None)",
    )
    extensions = [".asc", ".h5", ".swc"]
    output_columns = {}
    for extension in extensions:
        ext = extension[1:]
        output_columns.update(
            {
                f"clone_morph_db_path_{ext}": None,
                f"clone_release_morph_path_{ext}": None,
            }
        )

    @staticmethod
    def validation_function(
        df, data_dir, clone_path, extensions, clone_data_path
    ):  # pylint: disable=arguments-differ, unused-argument
        """Make a clone release."""
        df = MorphDB.from_neurondb(
            clone_data_path / "clone" / "neuronDB.xml",
            morphology_folder=str(clone_data_path / "clone"),
        ).df.rename(columns={"path": "morph_path"})
        for extension in extensions:
            _clone_path = Path(f"{clone_path}/{extension[1:]}")
            _clone_path.mkdir(exist_ok=True, parents=True)
            __create_db_row = partial(
                _create_db_row,
                clone_path=_clone_path,
                extension=extension,
            )

            _m = []
            with Pool() as pool:
                for index, row, m in tqdm(pool.imap(__create_db_row, df.iterrows()), total=len(df)):
                    df.loc[index] = pd.Series(row)
                    _m.append(m)

            db = MorphDB(_m)

            db.write(_clone_path / "neurondb.xml")
            df[f"clone_morph_db_path_{extension[:1]}"] = _clone_path / "neurondb.xml"

    def kwargs(self):
        return {
            "clone_path": self.clone_path,
            "extensions": self.extensions,
            "clone_data_path": self.input()[0]["data"].pathlib_path,
        }

    def inputs(self):
        return {CloneMorphologies: {}}
