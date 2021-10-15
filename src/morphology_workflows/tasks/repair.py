"""Repair tasks."""
import logging

import luigi
from data_validation_framework.task import ElementValidationTask
from data_validation_framework.task import SetValidationTask
from luigi_tools.parameter import BoolParameter

from morphology_workflows.curation import collect
from morphology_workflows.repair import fix_zero_diameters
from morphology_workflows.repair import make_collage
from morphology_workflows.repair import make_release
from morphology_workflows.repair import plot_repair
from morphology_workflows.repair import plot_smooth_diameters
from morphology_workflows.repair import repair
from morphology_workflows.repair import smooth_diameters
from morphology_workflows.repair import unravel
from morphology_workflows.tasks.utils import SkippableMixin

logger = logging.getLogger(__name__)


class CollectAnnotated(ElementValidationTask):
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

    validation_function = collect


class FixZeroDiameters(ElementValidationTask):
    """Fix zero diameters.

    This task applies a fix on zero diameters on dendrites, by calling
    :func:`neuror.zero_diameter_fixer.fix_zero_diameters`.
    """

    output_columns = {"morph_path": None}

    validation_function = fix_zero_diameters

    def inputs(self):
        """ """
        return {CollectAnnotated(): {"morph_path": "morph_path"}}


class Unravel(ElementValidationTask):
    """Unravel morphologies.

    In-vitro morphologies produce recostruction with too much tortuosity, which is corrected for
    here, using :func:`neuror.unravel.unravel`.
    As this changes the location of the points, the cut leaves have to be updated, as well as the
    apical points, if any.
    """

    window_half_length = luigi.FloatParameter(
        default=10, description=":float: Size in path length of unravel window"
    )
    output_columns = {
        "morph_path": None,
        "unravelled_cut_leaves_path": "",
        "unravelled_apical_point_path": "",
    }

    validation_function = unravel

    def kwargs(self):
        """ """
        return {"window_half_length": self.window_half_length}

    def inputs(self):
        """ """
        return {
            FixZeroDiameters(): {"morph_path": "morph_path"},
            CollectAnnotated(): {
                "cut_leaves_path": "cut_leaves_path",
                "apical_point_path": "apical_point_path",
            },
        }


class RepairNeurites(ElementValidationTask):
    """RepairNeurites morphologies.

    Using the cut leaves, we recreate missing branches using neuror.main.repair.

    .. todo::

        Currently, axons are not repaired, as they need other axons.

        Improve repair of dendrite by using all dendrites of same type.
    """

    output_columns = {"morph_path": None}
    validation_function = repair
    with_plot = BoolParameter(
        default=False, description=":bool: Save plots with highlighted repaired branches"
    )
    repair_params = luigi.DictParameter(
        default=None, description=":dict: Repair internal parameters"
    )

    def kwargs(self):
        """ """
        return {"with_plot": self.with_plot, "repair_params": self.repair_params}

    def inputs(self):
        """ """
        return {
            CollectAnnotated(): {
                "has_axon": "has_axon",
                "has_basal": "has_basal",
                "has_apical": "has_apical",
            },
            Unravel(): {
                "unravelled_apical_point_path": "unravelled_apical_point_path",
                "morph_path": "morph_path",
                "unravelled_cut_leaves_path": "unravelled_cut_leaves_path",
            },
        }


class MakeCollage(SetValidationTask):
    """Make collage plot of morphologies."""

    collage_path = luigi.Parameter(default="collage.pdf", description=":str: Path to collage plot")
    separation = luigi.FloatParameter(default=1500)
    layer_thickness = luigi.ListParameter(default=[700.0, 525.0, 190.0, 353.0, 149.0, 165.0])
    dpi = luigi.IntParameter(default=1000)
    n_morph_per_page = luigi.IntParameter(default=10)

    def kwargs(self):
        """ """
        return {
            "collage_path": self.collage_path,
            "separation": self.separation,
            "layer_thickness": self.layer_thickness,
            "dpi": self.dpi,
            "n_morph_per_page": self.n_morph_per_page,
        }

    validation_function = make_collage

    def inputs(self):
        """ """
        return {SmoothDiameters(): {"morph_path": "morph_path"}}


class MakeRelease(SetValidationTask):
    """Make a morpology release, with three possible folders: zero-diameter, unravel or repair."""

    zero_diameter_path = luigi.Parameter(
        default=None, description=":str: Path to zero diameter morphologies (not created if None)"
    )
    unravel_path = luigi.Parameter(
        default=None, description=":str: Path to unravel morphologies (not created if None)"
    )
    repair_path = luigi.Parameter(
        default="repaired_release",
        description=":str: Path to repaired morphologies (not created if None)",
    )

    output_columns = {
        "zero_diameter_morph_db_path": None,
        "unravel_morph_db_path": None,
        "repair_morph_db_path": None,
        "zero_diameter_release_morph_path": None,
        "unravel_release_morph_path": None,
        "repair_release_morph_path": None,
    }

    validation_function = make_release

    def kwargs(self):
        """ """
        return {
            "zero_diameter_path": self.zero_diameter_path,
            "unravel_path": self.unravel_path,
            "repair_path": self.repair_path,
        }

    def inputs(self):
        """ """
        return {
            FixZeroDiameters(): {"morph_path": "zero_diameter_morph_path"},
            Unravel(): {"morph_path": "unravel_morph_path"},
            RepairNeurites(): {"morph_path": "repair_morph_path"},
        }


class PlotRepair(SkippableMixin(), ElementValidationTask):
    """Plot the cut leaves on repaired cells."""

    output_columns = {"plot_repair_path": None}
    validation_function = plot_repair

    with_plotly = BoolParameter(default=False, description=":bool: Use Plotly for plotting")

    def kwargs(self):
        """ """
        return {"with_plotly": self.with_plotly, "skip": self.skip}

    def inputs(self):
        """ """
        return {
            Unravel(): {"unravelled_cut_leaves_path": "cut_leaves_path"},
            RepairNeurites(): {"morph_path": "morph_path"},
        }


class SmoothDiameters(SkippableMixin(True), ElementValidationTask):
    """Smooth diameters with :mod:`diameter_synthesis`.

    We use actual diameters to learn a diameter model used to diametrize the morphology.

    By default, this task is skipped.
    """

    output_columns = {"morph_path": None}
    validation_function = smooth_diameters

    def kwargs(self):
        """ """
        return {"skip": self.skip}

    def inputs(self):
        """ """
        return {
            RepairNeurites(): {"morph_path": "morph_path"},
            Unravel(): {"unravelled_apical_point_path": "apical_point_path"},
        }


class PlotSmoothDiameters(SkippableMixin(), ElementValidationTask):
    """Plot smoothed diameters versus originals."""

    output_columns = {"plot_smooth_path": None}
    validation_function = plot_smooth_diameters

    def kwargs(self):
        """ """
        return {"skip": self.skip}

    def inputs(self):
        """ """
        return {
            RepairNeurites(): {"morph_path": "morph_path"},
            SmoothDiameters(): {"morph_path": "smooth_morph_path"},
        }
