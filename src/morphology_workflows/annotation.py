"""Annotation functions."""
import json
import logging
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from data_validation_framework.result import ValidationResult
from morph_tool.apical_point import apical_point_position
from morph_tool.transform import rotate
from morphio import SectionType
from morphio.mut import Morphology
from neurom import COLS
from neurom import geom
from neurom import iter_neurites
from neurom import load_morphology
from neurom import view
from neurom.core.types import NeuriteType
from neurom.core.types import tree_type_checker
from neuror.cut_plane.cut_leaves import find_cut_leaves

from morphology_workflows.marker_helper import Marker
from morphology_workflows.marker_helper import MarkerSet

logger = logging.getLogger(__name__)
mpl.use("Agg")


def _get_bbox(neuron):
    """Get bbox for all dendrite except axon."""
    min_x, max_x, min_y, max_y = 1e10, -1e10, 1e10, -1e10
    for neurite in neuron.neurites:
        max_x = max(max_x, np.max(neurite.points[:, 0]))
        min_x = min(min_x, np.min(neurite.points[:, 0]))
        max_y = max(max_y, np.max(neurite.points[:, 1]))
        min_y = min(min_y, np.min(neurite.points[:, 1]))
    return min_x, max_x, min_y, max_y


def _get_bbox_no_axon(neuron):
    """Get bbox for all dendrite except axon."""
    min_x, max_x, min_y, max_y = 1e10, -1e10, 1e10, -1e10
    for neurite in neuron.neurites:
        if neurite.type != SectionType.axon:
            max_x = max(max_x, np.max(neurite.points[:, 0]))
            min_x = min(min_x, np.min(neurite.points[:, 0]))
            max_y = max(max_y, np.max(neurite.points[:, 1]))
            min_y = min(min_y, np.min(neurite.points[:, 1]))
    return min_x, max_x, min_y, max_y


def assign_mtypes(
    row,
    data_dir,  # noqa: ARG001
    mtype_regex=None,
):  # pylint: disable=unused-argument
    """Assign mtype to morphologies."""
    if not mtype_regex:
        return ValidationResult(is_valid=True, mtype=row.mtype, comment="No regex provided!")
    if not hasattr(row, "mtype") or not isinstance(row.mtype, str):
        return ValidationResult(is_valid=True, mtype=None, comment="No mtype data!")

    mtype_re = [re.compile(_re) for _re in mtype_regex]
    _matches = [_re.findall(row.mtype)[0] for _re in mtype_re if _re.match(row.mtype)]
    if len(_matches) > 1:
        logger.warning("Found several mtype names: %s, so we take the first", _matches)
    if len(_matches) == 0:
        return ValidationResult(is_valid=False, comment=f"provided mtype is not valid: {row.mtype}")
    return ValidationResult(is_valid=True, mtype=_matches[0])


def calculate_y_extent(morph, neurite_type):
    """Find min/max y value of morphology based on neurite_type.

    Taken from morphology_repair_workflow.placement_annotations.py
    """
    total_min = float("inf")
    total_max = float("-inf")
    for n in iter_neurites(morph, filt=neurite_type):
        min_, max_ = geom.bounding_box(n)
        total_min = min(min_[COLS.Y], total_min)
        total_max = max(max_[COLS.Y], total_max)
    return total_min, total_max


SEGMENT_TO_NEURITE = {
    "axon": tree_type_checker(NeuriteType.axon),
    "dendrite": tree_type_checker(NeuriteType.basal_dendrite, NeuriteType.apical_dendrite),
}


def compute_hard_limits(
    row,
    data_dir,
    skip_axon=False,
    dendrite_hard_limit="L1_hard_limit",
    axon_hard_limit="L1_axon_hard_limit",
):
    """Compute hard limits."""
    morph = load_morphology(row.morph_path)

    rules = {}
    y_min, y_max = calculate_y_extent(morph, SEGMENT_TO_NEURITE["dendrite"])
    rules[dendrite_hard_limit] = {"y_min": y_min, "y_max": y_max}
    if not skip_axon:
        y_min, y_max = calculate_y_extent(morph, SEGMENT_TO_NEURITE["axon"])
        rules[axon_hard_limit] = {"y_min": y_min, "y_max": y_max}

    _markers = [
        Marker(
            name + "_" + tpe,
            "axis",
            [[0.0, rule[tpe], 0.0], [1.0, rule[tpe], 0.0]],
            morph_name=row.name,
            morph_path=row.morph_path,
        )
        for name, rule in rules.items()
        for tpe in ["y_min", "y_max"]
    ]

    hard_limit_path = Path(data_dir, row.name + ".yaml")
    MarkerSet.from_markers(_markers).save(filename=hard_limit_path)
    return ValidationResult(is_valid=True, hard_limit_path=hard_limit_path)


def _labeled_line(ax, name, y_pos, color, pos="center"):
    """Draw a line to the axis, returning the line and the annotation."""
    x_pos = {"center": 0, "left": ax.dataLim.bounds[0]}[pos]
    line = ax.axhline(y_pos, c=color)
    anno = ax.annotate(name, xy=(0, y_pos), xytext=(x_pos, y_pos + 3))
    return line, anno


def plot_hard_limits(row, data_dir, with_plotly=True):
    """Plotting placement annotations on morphologies.

    TODO: update the plotting without plotly.
    """
    plot_path = None
    if not row.isnull()["hard_limit_path"]:
        if with_plotly:
            plot_path = (data_dir / row.name).with_suffix(".html")
            MarkerSet.from_file(row.hard_limit_path).plot(filename=plot_path)
        else:
            pass
        #    pa = PlacementAnnotation.load(Path(row.hard_limit_path).with_suffix(""))
        #    neuron = load_morphology(row.morph_path)
        #    plt.figure()
        #    ax = plt.gca()
        #    for rule, data in pa.items():
        #        _labeled_line(ax, rule + " min", data["y_min"], color="g")
        #        _labeled_line(ax, rule + " max", data["y_max"], color="g")
        #
        #    view.plot_morph(neuron, ax, realistic_diameters=True)
        #
        #    plt.axis(_get_bbox(neuron))
        #    plot_path = (data_dir / row.name).with_suffix(".pdf")
        #    plt.savefig(plot_path)
        #    plt.close()

    return ValidationResult(is_valid=True, plot_hard_limit_path=plot_path)


def find_apical_point(row, data_dir, tuft_percent=20):  # pylint: disable=unused-argument
    """Find apical point."""
    if row.has_apical:
        neuron = Morphology(row.morph_path)
        apical_point = apical_point_position(neuron, tuft_percent=tuft_percent)
        if apical_point is not None:
            marker = Marker(
                "apical point",
                "points",
                apical_point,
                morph_name=row.name,
                morph_path=row.morph_path,
                plot_style={"color": "purple"},
            )
            apical_point_path = (data_dir / Path(row.morph_path).name).with_suffix(".yaml")
            MarkerSet.from_markers([marker]).save(filename=apical_point_path)
            return ValidationResult(is_valid=True, apical_point_path=apical_point_path)
    return ValidationResult(is_valid=True, apical_point_path=None)


def plot_apical_point(row, data_dir, with_plotly=True):
    """Plotting apical points on morphologies."""
    plot_path = None
    if row.has_apical and not row.isnull()["apical_point_path"]:
        if with_plotly:
            plot_path = str(data_dir / row.name) + ".html"
            MarkerSet.from_file(row.apical_point_path).plot(filename=plot_path)
        else:  # pragma: no cover
            plot_path = str(data_dir / row.name) + ".pdf"
            neuron = load_morphology(row.morph_path)
            plt.figure()
            apical_point = MarkerSet.from_file(row.apical_point_path)
            view.plot_morph(neuron, plt.gca(), realistic_diameters=True, soma_outline=False)
            plt.scatter(*apical_point.markers[0].data[:2], c="g", s=20)
            plt.axis(_get_bbox_no_axon(neuron))
            plt.axis("equal")
            plt.savefig(plot_path)
            plt.close()
    return ValidationResult(is_valid=True, plot_apical_point_path=plot_path)


def detect_cut_leaves(row, data_dir, bin_width=15, percentile_threshold=75):
    """Detect cut leaves and save status in df."""
    morph = Morphology(row.morph_path)
    rotation_matrix = np.eye(3)
    if not row.isnull()["rotation_matrix"]:
        rotation_matrix = np.array(json.loads(row.rotation_matrix)).T
        rotate(morph, rotation_matrix)
    morph = morph.as_immutable()

    if not morph.root_sections:
        raise ValueError(  # noqa: TRY003
            "Can not search for cut leaves for a neuron with no neurites."
        )

    leaves, qualities = find_cut_leaves(
        morph,
        bin_width=bin_width,
        searched_axes=("Z",),
        searched_half_spaces=(-1, 1),
        percentile_threshold=percentile_threshold,
    )

    if len(leaves) > 0:
        leaves = leaves.dot(rotation_matrix)
        marker = Marker(
            "cut leaves",
            "points",
            leaves,
            morph_name=row.name,
            morph_path=row.morph_path,
            plot_style={"color": "green"},
        )
        plane_path = (data_dir / Path(row.morph_path).name).with_suffix(".yaml")
        MarkerSet.from_markers([marker]).save(filename=plane_path)
        return ValidationResult(
            is_valid=True,
            cut_leaves_path=plane_path,
            cut_qualities=json.dumps(qualities),
        )
    return ValidationResult(
        is_valid=True,
        cut_leaves_path=None,
        cut_qualities=None,
        comment="No cut leaves found",
    )


def plot_cut_leaves(row, data_dir, with_plotly=True):
    """Plotting cut leaves on morphologies."""
    markers = None
    if not row.isnull()["cut_leaves_path"]:
        markers = MarkerSet.from_file(row.cut_leaves_path)

    plot_path = None
    if markers:
        if with_plotly:
            plot_path = (data_dir / row.name).with_suffix(".html")
            markers.plot(filename=plot_path)
        else:  # pragma: no cover
            plt.figure()
            plot_path = (data_dir / row.name).with_suffix(".pdf")
            neuron = load_morphology(row.morph_path)
            view.plot_morph(
                neuron, plt.gca(), realistic_diameters=False, plane="xz", soma_outline=False
            )

            points = markers.markers[0].data.T[[0, 2]]
            plt.scatter(*points, color="g", s=5)

            plt.gca().axis("equal")
            plt.title(f"{row.name}")
            plt.suptitle(f"probability = {row.cut_qualities}", fontsize=8)
            plt.savefig(plot_path)
            plt.close()

    return ValidationResult(is_valid=True, plot_cut_leaves_path=plot_path)
