"""Annotation tasks."""
import logging

import luigi
from data_validation_framework.task import ElementValidationTask
from data_validation_framework.task import SkippableMixin
from luigi_tools.parameter import BoolParameter

from morphology_workflows.annotation import assign_mtypes
from morphology_workflows.annotation import compute_hard_limits
from morphology_workflows.annotation import detect_cut_leaves
from morphology_workflows.annotation import find_apical_point
from morphology_workflows.annotation import plot_apical_point
from morphology_workflows.annotation import plot_cut_leaves
from morphology_workflows.annotation import plot_hard_limits
from morphology_workflows.curation import collect

logger = logging.getLogger(__name__)


class CollectCurated(ElementValidationTask):
    """Collect curated dataset to work with on this phase."""

    output_columns = {
        "morph_path": None,
        "has_apical": None,
        "mtype": None,
        "rotation_matrix": None,
    }

    validation_function = collect


class MType(ElementValidationTask):
    """Assign mtypes to morphologies.

    Currently, we only check if the mtype corresponds to the provided list of regex.
    If the list of regex is empty, we let all the morphologies pass (equivalent to skip).

    If they do not, we invalidate the morphologies.
    """

    output_columns = {"mtype": None}
    validation_function = assign_mtypes
    mtype_regex = luigi.OptionalListParameter(default=None)

    def kwargs(self):
        """ """
        return {"mtype_regex": self.mtype_regex}

    def inputs(self):
        """ """
        return {CollectCurated: {"mtype": "mtype"}}


class HardLimit(ElementValidationTask):
    """Compute hard limits.

    Hard limits are markers used to place morphologies in a circuit. They can be automatically
    computed as min and max extent of axon and dendrites.

    .. todo:: There can be more such markers, manually placed with ``nse-tools/placement-annotator``
        internal tool.
    """

    output_columns = {"hard_limit_path": None}
    validation_function = compute_hard_limits

    skip_axon = BoolParameter(
        default=False, description=":bool: Skip the computation for axon hard limit"
    )
    dendrite_hard_limit = luigi.Parameter(
        default="L1_hard_limit", description=":str: Name of the dendrite hard limit annotation"
    )
    axon_hard_limit = luigi.Parameter(
        default="L1_axon_hard_limit", description=":str: Name of the axon hard limit annotation"
    )

    def kwargs(self):
        """ """
        return {
            "skip_axon": self.skip_axon,
            "dendrite_hard_limit": self.dendrite_hard_limit,
            "axon_hard_limit": self.axon_hard_limit,
        }

    def inputs(self):
        """ """
        return {CollectCurated: {"morph_path": "morph_path"}}


class PlotHardLimit(SkippableMixin(), ElementValidationTask):
    """Plot the hard limits.

    Plot hard limits as lines on morphologies with plotly.
    """

    output_columns = {"plot_hard_limit_path": None}
    validation_function = plot_hard_limits

    with_plotly = BoolParameter(default=True, description=":bool: Use Plotly for plotting")

    def kwargs(self):
        """ """
        return {"with_plotly": self.with_plotly}

    def inputs(self):
        """ """
        return {
            CollectCurated: {"morph_path": "morph_path"},
            HardLimit: {"hard_limit_path": "hard_limit_path"},
        }


class ApicalPoint(SkippableMixin(), ElementValidationTask):
    """Detect apical point.

    For morphologies with apical dendrites we estimate the location of the apical point with
    the automatic tool :func:`morph_tool.apical_point.apical_point_position`.
    """

    output_columns = {"apical_point_path": None}
    validation_function = find_apical_point

    tuft_percent = luigi.FloatParameter(
        default=20, description=":float: Fraction of tuft for morph-tool algorithm"
    )

    def kwargs(self):
        """ """
        return {"tuft_percent": self.tuft_percent}

    def inputs(self):
        """ """
        return {CollectCurated: {"morph_path": "morph_path", "has_apical": "has_apical"}}


class PlotApicalPoint(SkippableMixin(), ElementValidationTask):
    """Plot apical point.

    Plot apical point as a single scatter point on a morphology with plotly.
    """

    output_columns = {"plot_apical_point_path": None}
    validation_function = plot_apical_point

    with_plotly = BoolParameter(default=True, description=":bool: Use Plotly for plotting")

    def kwargs(self):
        """ """
        return {"with_plotly": self.with_plotly}

    def inputs(self):
        """ """
        return {
            CollectCurated: {"morph_path": "morph_path", "has_apical": "has_apical"},
            ApicalPoint: {"apical_point_path": "apical_point_path"},
        }


class CutLeaves(ElementValidationTask):
    """Detect the cut leaves.

    Cut leaves are considered to represent boundaries of slices of in-vitro reconstructions.
    See :func:`neuror.cut_plane.cut_leaves` for more details on the algorithm and its parameters.
    """

    output_columns = {"cut_leaves_path": None, "cut_qualities": None}
    validation_function = detect_cut_leaves

    bin_width = luigi.FloatParameter(
        default=15, description=":float: Thickness of cut plane, (used if legacy_detection=False)"
    )
    percentile_threshold = luigi.FloatParameter(
        default=70, description=":float: Threshold percenetile for finding a cut plane"
    )

    def kwargs(self):
        """ """
        return {"bin_width": self.bin_width, "percentile_threshold": self.percentile_threshold}

    def inputs(self):
        """ """
        return {CollectCurated: {"morph_path": "morph_path", "rotation_matrix": "rotation_matrix"}}


class PlotCutLeaves(SkippableMixin(), ElementValidationTask):
    """Plot the cut leaves.

    We plot the cut-leaves as scatter markers, as the terminal points on dendrites intersecting
    the cut plane.
    """

    output_columns = {"plot_cut_leaves_path": None}
    validation_function = plot_cut_leaves

    with_plotly = BoolParameter(default=True, description=":bool: Use Plotly for plotting")

    def kwargs(self):
        """ """
        return {"with_plotly": self.with_plotly}

    def inputs(self):
        """ """
        return {
            CutLeaves: {"cut_leaves_path": "cut_leaves_path", "cut_qualities": "cut_qualities"},
            CollectCurated: {"morph_path": "morph_path"},
        }
