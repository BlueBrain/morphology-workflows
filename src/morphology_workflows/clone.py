"""Cloning functions."""

import copy
import functools
import itertools
import json
import logging
import math
import shutil
import xml.etree.ElementTree as ET  # noqa: N817
from collections import defaultdict
from collections import namedtuple
from collections.abc import MutableMapping
from copy import deepcopy
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import lxml.etree
import morphio
import neurom
import numpy as np
import pandas as pd
from morph_tool.exceptions import MorphToolException
from morph_tool.graft import graft_axon
from morph_tool.morphdb import MorphDB
from morph_tool.morphdb import MorphInfo
from morphio import SectionType
from neuroc.scale import RotationParameters
from neuroc.scale import ScaleParameters
from neuroc.scale import scale_morphology
from neuroc.scale import yield_clones
from neurom import geom
from neurom import iter_neurites
from neurom import load_morphology
from neurom.core.dataformat import COLS
from neurom.core.types import NeuriteType
from neurom.core.types import tree_type_checker
from tqdm import tqdm

from morphology_workflows.utils import _convert
from morphology_workflows.utils import compare_lists
from morphology_workflows.utils import get_points
from morphology_workflows.utils import import_morph
from morphology_workflows.utils import rng_from_name
from morphology_workflows.utils import seed_from_name
from morphology_workflows.utils import write_neuron

L = logging.getLogger(__name__)
Category = namedtuple("Category", "mtype layer")  # noqa: PYI024


# ########################################################################## #
# Imported from `morphology_repair_workflow.clone_morphologies==2.0.4.dev2`. #
# ########################################################################## #

COL_Y = 1


def _parse_recipe(recipe_filename):
    """Parse a BBP recipe and return the corresponding etree."""
    # HACK: The XML parser does not handle the entity tag "&connectivityRecipe"  # noqa: FIX004
    # Since we don't care about the tag (all we want from the recipe is the number of
    # morphologies for each mtype), let's just pretend the line does not exist...
    with open(recipe_filename, encoding="utf-8") as f:
        lines = filter(lambda line: "&connectivityRecipe" not in line, f)
        return ET.fromstringlist(lines)


def get_cellcounts_from_recipe(recipe_path):
    """Get the number of cells needed, by category from the BBP recipe."""
    root = _parse_recipe(recipe_path)
    neuron_types = root.findall(".//NeuronTypes")[0]

    category_to_cellcount = {}
    total = int(neuron_types.attrib["totalNeurons"])
    for layer_node in neuron_types.findall("./Layer"):
        num_in_layer = math.ceil(total * float(layer_node.attrib["percentage"]) / 100.0)
        for structural_type_node in layer_node.findall("./StructuralType"):
            percentage = float(structural_type_node.attrib["percentage"]) / 100.0
            category = Category(structural_type_node.attrib["id"], layer_node.attrib["id"])
            category_to_cellcount[category] = math.ceil(num_in_layer * percentage)
    return category_to_cellcount


def get_category_overlap(category_to_cellcount, candidates):
    """Finds the overlap of categories between the recipe and the neurondb."""
    overlap, only_in_l0, only_in_l1 = compare_lists(category_to_cellcount.keys(), candidates.keys())

    if only_in_l0:
        # TODO: decide if this needs to be an exception
        L.warning("Categories only in recipe, will not have enough morphologies: %s", only_in_l0)

    if only_in_l1:
        L.info("Categories only in neurondb: %s", only_in_l1)

    return overlap


def _clone_path(input_filename, output_path, clone_id):
    """Build a clone path from a parent name and the clone ID."""
    return Path(output_path, f"{input_filename.stem}_-_Clone_{clone_id}{input_filename.suffix}")


def make_clones(  # noqa: PLR0913
    task,
    clone_counts,
    axon_hardlimit_name,
    dendrite_hardlimit_name,
    input_path,
    output_path,
    input_annotation,
    output_annotation,
    placement_rules,
    std_angle=10.0,
    std_scale=0.2,
):
    """Generate the clones.

    It also creates the clone annotations if placement_rules is not None.
    """
    name, morphs = task
    filename = input_path / (name + ".h5")
    parent_morph = morphio.Morphology(filename)
    shutil.copy(filename, output_path / filename.name)

    clone_generator = yield_clones(
        parent_morph,
        rotation_params=RotationParameters(std_angle=std_angle),
        section_scaling=ScaleParameters(std=std_scale),
        seed=seed_from_name(name),
    )

    # Yields one MorphInfo object per clone to be generated
    iter_morph = (
        morph
        for _, morph in morphs.iterrows()
        for _ in range(clone_counts.get(Category(morph.mtype, morph.layer), 0))
    )

    parent_annotation = PlacementAnnotation.load(Path(input_annotation, name))
    parent_annotation.save(output_annotation)

    return [
        _make_clone(
            _clone_path(filename, output_path, i),
            clone_generator,
            placement_rules,
            parent_annotation,
            axon_hardlimit_name,
            dendrite_hardlimit_name,
            output_annotation=output_annotation,
            parent_morph=parent_morph,
            parent_info=morph,
        )
        for i, morph in enumerate(iter_morph)
    ]


def _make_clone(  # noqa: PLR0913
    clone_path,
    clone_generator,
    placement_rules,
    parent_annotation,
    axon_hardlimit_name,
    dendrite_hardlimit_name,
    output_annotation,
    parent_morph,
    parent_info,
):
    """Create a clone and its annotation."""
    write_neuron(next(clone_generator), clone_path)

    clone_info = MorphInfo(name=clone_path.stem, mtype=parent_info.mtype, layer=parent_info.layer)
    clone_info.dendrite_donor = parent_info.name
    clone_info.axon_donor = parent_info.name

    _make_clone_annotation(
        parent_morph,
        clone_path,
        placement_rules,
        parent_annotation,
        clone_info,
        axon_hardlimit_name,
        dendrite_hardlimit_name,
        output_annotation,
    )
    return clone_info


def _make_clone_annotation(
    parent_morph,
    clone_path,
    placement_rules,
    parent_annotation,
    clone_info,
    axon_hardlimit_name,
    dendrite_hardlimit_name,
    output_annotation,
):
    """Create an annotation for a given cloned morphology."""
    parent_name = clone_info.dendrite_donor
    rule_indices = _extract_segment_indices(
        parent_morph, parent_name, placement_rules.get(clone_info.mtype, {}), parent_annotation
    )

    child_annotation = _update_rules(clone_path, clone_info.name, parent_annotation, rule_indices)
    neuron = load_morphology(clone_path)

    child_annotation.calculate_axon_hard_limit(neuron, axon_hardlimit_name)
    child_annotation.calculate_dendrite_hard_limit(neuron, dendrite_hardlimit_name)
    child_annotation.save(output_annotation)


def _update_rules(clone_path, child_name, parent_annotation, rule_indices):
    """Recalculate the relevant annotation rules based on the new file."""
    child_annotation = copy.deepcopy(parent_annotation)
    child_annotation.name = child_name
    for rule in child_annotation:
        if rule in rule_indices:
            indices = rule_indices[rule]
            y_min, y_max = _extract_y_extent(clone_path, indices)

            child_annotation.update_y_extent(rule, y_min, y_max)

    return child_annotation


def _extract_y_indices(morph, segment_type, y_min, y_max):
    """Extract the Y indices."""
    assert segment_type in (
        "axon",
        "dendrite",
    ), "Segment type must be axon or dendrite"

    in_range = [False] * len(morph.soma.points)

    if segment_type == "axon":
        allowed = {SectionType.axon}
    else:
        allowed = {SectionType.apical_dendrite, SectionType.basal_dendrite}

    for section in morph.iter():
        if section.type in allowed:
            in_range.append(
                (y_min <= section.points[:, COL_Y]) & (section.points[:, COL_Y] <= y_max)
            )
        else:
            in_range.append([False] * len(section.points))

    return np.hstack(in_range)


def _extract_y_extent(clone_path, indices):
    """Find y_min, y_max for morph, within indices."""
    points = get_points(clone_path)
    y_points = points[indices, COL_Y]
    return np.min(y_points), np.max(y_points)


def _extract_segment_indices(parent_morph, parent, placement_rules, parent_annotation):
    """Return a dict of rule -> y-indices based on the segment type."""
    ret = {}
    for rule, properties in parent_annotation.items():
        if rule in placement_rules and "segment_type" in placement_rules[rule]:
            segment_type = placement_rules[rule]["segment_type"]
            y_min, y_max = properties["y_min"], properties["y_max"]
            indices = _extract_y_indices(parent_morph, segment_type, y_min, y_max)
            if np.count_nonzero(indices) == 0:
                L.warning(
                    "Parent: %s has rule (%s: %d - %d) that results in no extent",
                    parent,
                    rule,
                    y_min,
                    y_max,
                )
                continue
            ret[rule] = indices

    return ret


# ################################################################### #
# Imported from `morphology_repair_workflow.graft_axons==2.0.4.dev2`. #
# ################################################################### #


def _filter_graft_inputs(df, neurondb, cross_mtypes):
    """Filter morphologies according to their use_axon values.

    Only morphologies that are in the neurondb.xml are considered for grafting
    axon usage depends on use_axon being true in the neurondb.xml

    Returns:
        to_cross: dict mapping mtype -> list(MorphInfo) for the axon recipients
        src_axons: dict mapping mtype -> list(MorphInfo) for the axon donors
        morph_names_layers: dict morph_name -> list(layers)
    """
    morph_names_layers = defaultdict(set)

    morphs = []
    for _, morph in neurondb.df.iterrows():
        morph_names_layers[morph["name"]].add(morph["layer"])
        if len(morph_names_layers[morph["name"]]) > 1:
            continue
        morphs.append(morph["name"])

    cross_morphs = df.loc[(df.index.isin(morphs)) & (df["mtype"].isin(cross_mtypes))]

    def to_dict(series):
        tmp = series.to_frame().groupby("mtype").groups
        return {k: set(v.tolist()) for k, v in tmp.items()}

    to_cross = to_dict(
        cross_morphs.loc[(cross_morphs["has_basal"]) | (cross_morphs["has_apical"]), "mtype"]
    )

    src_axons = to_dict(cross_morphs.loc[cross_morphs["has_axon"], "mtype"])

    if len(src_axons) == 0:
        raise ValueError("No axons found to graft")  # noqa: TRY003
    return dict(to_cross), dict(src_axons), dict(morph_names_layers)


def _create_graft_annotation(  # noqa: PLR0913
    annotations_path,
    morph,
    morph_name,
    recipient,
    mtype,
    src_axon,
    placement_rules,
    axon_hardlimit_name,
    dendrite_hardlimit_name,
):
    """Create annotations for the frankenmorph."""
    new_annotation = PlacementAnnotation(morph_name)

    neuron = neurom.core.Morphology(morph)
    new_annotation.calculate_axon_hard_limit(neuron, axon_hardlimit_name)
    new_annotation.calculate_dendrite_hard_limit(neuron, dendrite_hardlimit_name)

    placement_rules = placement_rules.get(mtype, {})

    for name, segment_type in ((recipient, "dendrite"), (src_axon, "axon")):
        parent_annotations = PlacementAnnotation.load(Path(annotations_path, name))
        for rule_name, rules in parent_annotations.items():
            if (
                rule_name in placement_rules
                and segment_type == placement_rules[rule_name]["segment_type"]
            ):
                new_annotation.add_rule(rule_name, rules)

    new_annotation.save(annotations_path)


def _record_grafts(  # noqa: PLR0913
    morph,
    annotations_path,
    morph_name,
    recipient,
    mtype,
    src_axon,
    morph_names_layers,
    new_neurondb,
    placement_rules,
    axon_hardlimit_name,
    dendrite_hardlimit_name,
):
    """Save information about the created grafts to the neurondb and annotations."""
    for layer in morph_names_layers[recipient]:
        morph_info = MorphInfo(name=morph_name, mtype=mtype, layer=layer)
        morph_info.dendrite_donor = recipient
        morph_info.axon_donor = src_axon

        new_neurondb += MorphDB([morph_info])
        _create_graft_annotation(
            annotations_path,
            morph,
            morph_name,
            recipient,
            mtype,
            src_axon,
            placement_rules,
            axon_hardlimit_name,
            dendrite_hardlimit_name,
        )


def graft_axons(
    df,
    output_path,
    input_neurondb,
    cross_mtypes,
    placement_rules,
    axon_hardlimit_name,
    dendrite_hardlimit_name,
):
    """Graft axons from donors to recipient morphologies, but only ones of cross_mtypes.

    Args:
        df(DataFrame): input df
        input_path(str): path to input files
        output_path(str): path to output directory
        input_neurondb (MorphDB): the input (previous phase) morphdb
        cross_mtypes(list): names of mytpes to be used for crossing, ex:  ['L23_PC', 'L4_PC', ...]
        placement_rules(dict): mapping of placement_rules as returned by `read_placement_rules`
        axon_hardlimit_name(str): named used in xml annotation file for axon y-extent
        dendrite_hardlimit_name(str): named used in xml annotation file for dendrites y-extent

    Returns:
        The output neurondb
    """
    L.debug("Crossing mtypes: %s", cross_mtypes)
    to_cross, src_axons, morph_names_layers = _filter_graft_inputs(df, input_neurondb, cross_mtypes)

    missed_crossings = set()

    output_neurondb = MorphDB()

    new_annotations_path = Path(output_path) / "annotations"
    new_annotations_path.mkdir(parents=True, exist_ok=True)

    @functools.lru_cache(maxsize=None)
    def load_morph(name):
        """Copy the morphology and the annotation (only once thanks to the cache)."""
        morph_path, annotation_path = df.loc[name, ["morph_path", "annotation_path"]].values
        new_path = output_path / (name + ".h5")
        return import_morph(morph_path, new_path, annotation_path, new_annotations_path)

    for mtype in cross_mtypes:
        labels = to_cross.get(mtype, set()) | src_axons.get(mtype, set())

        if not labels:
            missed_crossings.add(mtype)
            continue

        L.debug("Crossing mtype %s", mtype)
        for recipient, src_axon in itertools.product(to_cross[mtype], src_axons[mtype]):
            if recipient == src_axon:
                continue

            morph = load_morph(recipient)
            donor_neuron = load_morph(src_axon)
            rng = rng_from_name(f"{recipient}_{src_axon}")

            try:
                graft_axon(morph, donor_neuron, rng)
            except MorphToolException:
                L.error("Error while grafting axon: %s on morphology: %s", src_axon, recipient)
                continue

            morph_name = f"dend-{recipient}_axon-{src_axon}"
            new_morph_path = Path(output_path, morph_name + ".h5")
            write_neuron(morph, new_morph_path)

            _record_grafts(
                morph,
                new_annotations_path,
                morph_name,
                recipient,
                mtype,
                src_axon,
                morph_names_layers,
                output_neurondb,
                placement_rules,
                axon_hardlimit_name,
                dendrite_hardlimit_name,
            )

    load_morph.cache_clear()

    if missed_crossings:
        L.info("Did not find crossings for %s", missed_crossings)

    return output_neurondb


def filter_missing_perimeter_morphs(df):
    """Remove morphologies with perimeters."""
    for name, path in df["morph_path"].items():
        morph = morphio.mut.Morphology(path)
        if len([v.perimeters for k, v in morph.sections.items() if len(v.perimeters) > 0]) > 0:
            df.loc[name, "is_valid"] = False
            df.loc[name, "ret_code"] = 1
            df.loc[
                name, "comment"
            ] = "The morphologies with perimeters can not be used for axon grafting"

    L.info(
        "Removed the following morphologies because they miss perimeters: %s",
        df.loc[~df["is_valid"]].index.tolist(),
    )


# ############################################################################# #
# Imported from `morphology_repair_workflow.placement_annotations==2.0.4.dev2`. #
# ############################################################################# #


SEGMENT_TO_NEURITE = {
    "axon": tree_type_checker(NeuriteType.axon),
    "dendrite": tree_type_checker(NeuriteType.basal_dendrite, NeuriteType.apical_dendrite),
}


def calculate_y_extent(morph, neurite_type):
    """Find min/max y value of morphology based on neurite_type."""
    total_min, total_max = float("inf"), float("-inf")
    for n in iter_neurites(morph, filt=neurite_type):
        min_, max_ = geom.bounding_box(n)
        total_min = min(min_[COLS.Y], total_min)
        total_max = max(max_[COLS.Y], total_max)
    return total_min, total_max


def convert_to_json_type(dict_):
    """Convert to json's corresponding type when possible.

    Note: original dict is modified in place
    """
    for k, v in dict_.items():
        try:
            json_value = json.loads(v)
            if isinstance(json_value, (int, float)):
                dict_[k] = float(json_value)
            elif isinstance(json_value, list):
                dict_[k] = json_value
        except ValueError:  # noqa: PERF203
            pass
    return dict_


def read_placement_rules(rule_contents):
    """Convert an placement_rules xml string to a dictionary of rules.

    Format of dict is:
        {'L1_HAC':
            {'L1_HAC, axon, Layer_1':
                {'old_id': 'L1_HAC_axon_target',
                'segment_type': 'axon',
                'type': 'region_target',
                'y_max_fraction': '1.00',
                'y_max_layer': '1',
                'y_min_fraction': '0.00',
                'y_min_layer': '1'}},
    """
    root = ET.fromstring(rule_contents)
    assert root.tag == "placement_rules"

    def get_rules(el):
        """Extract rules into a dictionary."""
        ret = {}
        for child in el:
            attribs = dict(child.attrib)
            name = attribs["id"]
            del attribs["id"]

            ret[name] = convert_to_json_type(attribs)
        return ret

    ret = {}
    for child in root:
        if child.tag == "global_rule_set":
            ret["global_rule_set"] = get_rules(child)
        elif child.tag == "mtype_rule_set":
            ret[child.attrib["mtype"]] = get_rules(child)

    return ret


class PlacementAnnotation(MutableMapping):
    """Dict like class to handle placement annotations."""

    def __init__(self, name, rules=None):
        # pylint: disable=super-init-not-called
        self.name = name
        self._rules = rules or {}
        assert isinstance(self._rules, dict)

    def __str__(self):
        """Str."""
        ret = [self.name + ":"]
        ret.extend(f'\t"{k}": {v}' % (k, v) for k, v in self._rules.items())
        return "\n".join(ret)

    def __delitem__(self, key):
        """Del."""
        del self._rules[key]

    def __getitem__(self, key):
        """Get."""
        return self._rules[key]

    def __iter__(self):
        """Iter."""
        return iter(self._rules)

    def __len__(self):
        """Len."""
        return len(self._rules)

    def __setitem__(self, key, value):
        """Set."""
        assert isinstance(value, dict)
        self._rules[key] = value

    def add_rule(self, name, properties):
        """Adds rule `name` w/ properties.

        Note: this overwrites an old rule of the same name
        """
        self[name] = properties

    def xml(self):
        """Return the xml representation of a placement annotation."""
        annotations = ET.Element("annotations")
        annotations.attrib["morphology"] = self.name
        for rule, attribs in self._rules.items():
            placement = ET.SubElement(annotations, "placement")
            placement.attrib["rule"] = rule
            for k, v in attribs.items():
                placement.attrib[k] = str(v)
        return annotations

    def add_scale_bias(self, percentage):
        """Add the scale bias rule."""
        self.add_rule(
            "ScaleBias",
            {
                "percentage": str(percentage),
            },
        )

    def update_y_extent(self, rule, y_min, y_max):
        """Given rule, update the y_min and y_max.

        if previous values exist, add the y_change value
        """
        properties = {"y_min": y_min, "y_max": y_max}
        if rule in self:
            old_y = self[rule]
            properties["y_change"] = (y_max - y_min) / (old_y["y_max"] - old_y["y_min"])
        self.add_rule(rule, properties)

    def calculate_axon_hard_limit(self, morph, name):
        """Given a neuron.fst morphology, update the axon y_extent."""
        min_, max_ = calculate_y_extent(morph, SEGMENT_TO_NEURITE["axon"])
        self.update_y_extent(name, min_, max_)

    def calculate_dendrite_hard_limit(self, morph, name):
        """Given a neuron.fst morphology, update the dendrites y_extent."""
        min_, max_ = calculate_y_extent(morph, SEGMENT_TO_NEURITE["dendrite"])
        self.update_y_extent(name, min_, max_)

    @classmethod
    def load(cls, file_path):
        """Load a set of annotations created for a particular morphology from an xml file.

        If the file is missing, an rule-less annotation is returned
        """
        file_path = str(file_path)
        if file_path.endswith(".xml"):
            file_path = file_path[:-4]
        with open(str(file_path) + ".xml", encoding="utf-8") as fd:
            return cls.from_xml_string(fd.read())

    @classmethod
    def from_xml_string(cls, annotations_string):
        """Load a set of annotations created for a particular morphology from a string."""
        annotations = ET.fromstring(annotations_string)
        name = annotations.attrib["morphology"]
        ret = cls(name)
        for rule in annotations:
            rule_name = rule.attrib["rule"]
            properties = {k: v for k, v in rule.attrib.items() if k != "rule"}
            convert_to_json_type(properties)
            ret.add_rule(rule_name, properties)
        return ret

    def path(self, output_path):
        """Build annotation path."""
        return Path(output_path, self.name + ".xml")

    def save(self, output_path):
        """Save the annotations XML to the output_path/name."""
        with self.path(output_path).open("w", encoding="utf-8") as fd:
            fd.write(ET.tostring(self.xml(), encoding="unicode"))


# ########################################################################## #
# Imported from `morphology_repair_workflow.scale_morphologies==2.0.4.dev2`. #
# ########################################################################## #


def _scaled_name(morph_name, y_scale):
    """By inspection, this is the output from the MorphScale command."""
    return f"{morph_name}_-_Scale_x{1:1.3f}_y{y_scale:1.3f}_z{1:1.3f}.h5"


def apply_scaling(
    name, input_path, output_path, y_scales, parents, annotation_path, output_annotation_path
):
    """Worker function for parallelized computations.

    Note: the 2nd argument is unused to respect Phase.parallel_map expected signature
    """
    rng = rng_from_name(name)

    path = input_path / (name + ".h5")
    parent_annotation = PlacementAnnotation.load(annotation_path / name)
    parent_annotation.save(output_annotation_path)

    original = morphio.Morphology(path)
    write_neuron(original, output_path / (name + ".h5"))
    output_info = []

    for y_scale in y_scales:
        outname = Path(output_path, _scaled_name(path.stem, y_scale))
        if outname.exists():
            raise OSError(f"The file {outname} already exists!")  # noqa: TRY003
        morphology = morphio.mut.Morphology(original)
        scale_morphology(morphology, section_scaling=ScaleParameters(mean=y_scale, axis=1), rng=rng)
        write_neuron(morphology, outname)

        _add_annotation(outname, parent_annotation, y_scale, output_annotation_path)

        # A scaled morpho might have more than 1 parent
        # It can be used in multiple (mtype, layer)
        for parent in parents[path.stem]:
            info = MorphInfo(name=outname.stem, mtype=parent.mtype, layer=parent.layer)
            info.dendrite_donor = parent.name
            info.axon_donor = parent.name
            output_info.append(info)
    return output_info


def _add_annotation(path, parent_annotation, y_scale, annotations_path):
    """Create annotations for newly scaled morphologies."""
    child = deepcopy(parent_annotation)
    child.name = path.stem
    child.add_scale_bias((y_scale - 1.0) * 100)

    # scale rules which have y_min/y_max: examining the code in
    # MorphScale:./apps/MorphScale.cpp each of the segment start and
    # endpoints are multiplied by the y_scale, so it should be safe to
    # do the same to the y_min and y_max
    for properties in child.values():
        if "y_min" in properties:
            properties["y_min"] = y_scale * properties["y_min"]
            properties["y_max"] = y_scale * properties["y_max"]

    child.save(annotations_path)


# ####################################################################### #
# Imported from `morphology_repair_workflow.transform_rules==2.0.4.dev2`. #
# ####################################################################### #

# exclusions is a set of morphology names to be excluded
# expansions is a dictionary of from_ (mtype) -> list of[to (mtype)]
# layer_copies: dictionary of tuples (mtype, layer) -> list of layers to expand
MorphDBTransformRules = namedtuple(  # noqa: PYI024
    "MorphDBTransformRules", "exclusions expansions layer_copies"
)


def parse_morphdb_transform_rules(rules_path):
    """Parse rules in XML file that describe morphdb transformations.

    Format:

    .. code-block:: xml

        <rules>
            <exclude name="Excluded0"/>  <!-- name of morphology to exclude -->
            ...
            <substitute from="L5_MC" to="L6_MC"/>
                <!-- mtype in 'from' are duplicated with and saved with mtype of 'to' -->
            ...
            <mtype_copy_to_layer mtype="L23_CHC" src_layer="2" dst_layer="3" />
                <!-- for all mtypes=L23_CHC in layer 2, copy them to layer 2 -->
        </rules>

    Note: this is a replication of the 'exclude_and_substitute.py' algorithm, so the XML format
    was re-used for continuity.
    """
    rules_xml = ET.parse(rules_path)
    exclusions = {rule.get("name") for rule in rules_xml.findall("exclude")}
    expansions = defaultdict(list)
    layer_copies = defaultdict(list)

    for rule in rules_xml.findall("substitute"):
        from_ = rule.get("from")
        to = rule.get("to")
        assert from_ and to, f"Need to have both from: '{from_}' and to: '{to}'"  # noqa: PT018
        expansions[from_].append(to)

    for rule in rules_xml.findall("mtype_copy_to_layer"):
        mtype = rule.get("mtype")
        src_layer = rule.get("src_layer")
        dst_layer = rule.get("dst_layer")
        assert (  # noqa: PT018
            src_layer and dst_layer and to and mtype
        ), f"Need mtype: '{mtype}' src_layer: '{src_layer}' and dst_layer '{dst_layer}'"
        layer_copies[(mtype, src_layer)].append(dst_layer)

    return MorphDBTransformRules(exclusions, dict(expansions), dict(layer_copies))


def expand_mtype_to_layers(mtype):
    """Return a list of layers encoded in the mtype."""
    assert 0 < int(mtype[1]) < 7, "mtype has incorrectly encoded layer: " + mtype
    ret = [
        mtype[1],
    ]
    if mtype[2] == "3":  # in the L23_* cases
        ret.append("3")
    return ret


# ################################################# #
# Imported from `placement_algorithm.files==2.1.3`. #
# ################################################# #


def parse_annotations(filepath):
    """Parse XML with morphology annotations."""
    etree = lxml.etree.parse(filepath)
    result = {}
    for elem in etree.findall("placement"):
        attr = dict(elem.attrib)
        rule_id = attr.pop("rule")
        if rule_id in result:
            raise ValueError(f"Duplicate annotation for rule '{rule_id}'")  # noqa: TRY003
        result[rule_id] = attr
    return result


# ############################################################### #
# Imported from `placement_algorithm.compact_annotations==2.1.3`. #
# ############################################################### #


def collect_annotations(annotation_dir):
    """Collect all annotations from a directory."""
    result = {}
    for filepath in tqdm(Path(annotation_dir).rglob("*.xml")):
        morph = Path(filepath).stem
        result[morph] = parse_annotations(filepath)
    return result


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


def clone_release(
    df, data_dir, clone_path, extensions, clone_data_path  # noqa: ARG001
):  # pylint: disable=unused-argument
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
