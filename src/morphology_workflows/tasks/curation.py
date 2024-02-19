"""Curation tasks."""
import logging

import luigi
from data_validation_framework.task import ElementValidationTask
from data_validation_framework.task import SetValidationTask
from data_validation_framework.task import SkippableMixin
from luigi.parameter import OptionalChoiceParameter
from luigi.parameter import OptionalNumericalParameter
from luigi_tools.parameter import BoolParameter
from neuror.sanitize import _ZERO_LENGTH

from morphology_workflows.curation import align
from morphology_workflows.curation import check_neurites
from morphology_workflows.curation import collect
from morphology_workflows.curation import detect_errors
from morphology_workflows.curation import extract_marker
from morphology_workflows.curation import fix_neurites_in_soma
from morphology_workflows.curation import make_error_report
from morphology_workflows.curation import orient
from morphology_workflows.curation import plot_errors
from morphology_workflows.curation import plot_markers
from morphology_workflows.curation import plot_morphology
from morphology_workflows.curation import recenter
from morphology_workflows.curation import resample
from morphology_workflows.curation import sanitize
from morphology_workflows.utils import StrIndexMixin

logger = logging.getLogger(__name__)


class Collect(StrIndexMixin, ElementValidationTask):
    """Collect external dataset from .csv file.

    Original dataset has to have a 'morph_name' and 'morph_path' columns, with the name and path to
    the corresponding morphologies. Any other valid columns will be gathered as well.
    In addition, it will only collect morphologies with allowed extension: ``.asc``, ``.h5``,
    ``.swc``.
    """

    output_columns = {"morph_path": None}
    validation_function = collect


class ExtractMarkers(StrIndexMixin, SkippableMixin(), ElementValidationTask):
    """Extract marker information from the original morphology files, if any.

    Markers are additional spatial information contained in some reconstructed morphologies.
    Some can be read by MorphIO (https://github.com/BlueBrain/MorphIO/pull/186).
    The markers are stored in a custom API: :mod:`morphology_workflows.marker_helper`.
    """

    output_columns = {"marker_path": None}
    validation_function = extract_marker

    def inputs(self):
        return {Collect: {"morph_path": "morph_path"}}


class PlotMarkers(StrIndexMixin, SkippableMixin(), ElementValidationTask):
    """Plot markers on morphologies.

    Plot the markers extracted from :class:`tasks.curation.ExtractMarkers` on the morphologies.
    """

    output_columns = {"plot_marker_path": None}
    validation_function = plot_markers

    with_plotly = BoolParameter(default=True, description=":bool: Use Plotly for plotting")

    def kwargs(self):
        return {"with_plotly": self.with_plotly}

    def inputs(self):
        return {
            ExtractMarkers: {"marker_path": "marker_path"},
        }


class CheckNeurites(StrIndexMixin, ElementValidationTask):
    """Detect which neurites are present in the morphology, and add soma if missing.

    This task adds three important boolean flags:
        - has_axon
        - has_basal
        - has_apical

    set to ``False`` if any are absent.

    For the axon, if it only has fewer sections than ``axon_n_section_min``, it will be considered
    as ``has_axon=False``.

    If soma is absent, a soma is added as a circle, with center the mean of the first points of root
    sections, and radius as standard deviation of these points around the center.

    If the input dataset has columns ``use_axon`` or ``use_dendrites``, our corresponding flags
    will match.

    .. todo::
        Set various levels, a one section neurite can be set to ``has_basal=False`` for
        example.
    """

    output_columns = {
        "morph_path": None,
        "has_axon": False,
        "has_basal": False,
        "has_apical": False,
    }
    validation_function = check_neurites

    mock_soma_type = OptionalChoiceParameter(
        description=":str: Add a soma if missing of the given type",
        choices=["spherical", "contour"],
        default="spherical",
    )
    axon_n_section_min = luigi.IntParameter(
        description=":int: Minimum number of sections in an axon to consider it valid",
        default=5,
    )
    ensure_stub_axon = luigi.BoolParameter(
        description=":bool: Add a stub axon if there is no axon on the morphology",
        default=False,
    )
    min_length_first_section = OptionalNumericalParameter(
        description=(
            ":float: Resize the first section to be at least of the given size (do nothing if None "
            "is given)"
        ),
        default=_ZERO_LENGTH,
        var_type=float,
        min_value=0,
        max_value=float("inf"),
        left_op=luigi.parameter.operator.lt,
    )

    def kwargs(self):
        return {
            "mock_soma_type": self.mock_soma_type,
            "axon_n_section_min": self.axon_n_section_min,
            "ensure_stub_axon": self.ensure_stub_axon,
            "min_length_first_section": self.min_length_first_section,
        }

    def inputs(self):
        return {Collect: {"morph_path": "morph_path"}}


class Sanitize(StrIndexMixin, ElementValidationTask):
    """Sanitize the morphologies.

    Sanitization is done with :func:`neuror.sanitize.sanitize` and does the following:

    - ensures it can be loaded with MorphIO
    - raises if the morphology has no soma or of invalid format
    - removes unifurcations
    - set negative diameters to zero
    - raises if the morphology has a neurite whose type changes along the way
    - removes segments with near zero lengths (shorter than 1e-4)

    Note that the :class:`tasks.curation.CheckNeurites` task adds a soma if missing, so a failure
    here means that the soma does not have a valid type.
    """

    ensure_roots_at_soma = luigi.BoolParameter(
        default=True, description=":bool: Reconnect root sections at soma if they are not"
    )
    output_columns = {"morph_path": None}

    validation_function = sanitize

    def kwargs(self):
        return {"ensure_roots_at_soma": self.ensure_roots_at_soma}

    def inputs(self):
        return {CheckNeurites: {"morph_path": "morph_path"}}


class Recenter(StrIndexMixin, SkippableMixin(), ElementValidationTask):
    """Recenter morphologies.

    Often, morphologies do not have a soma centered at ``[0, 0, 0]``, so we recenter and save
    the original location, in case it is important to know where the morphology is located in atlas.
    """

    output_columns = {"morph_path": None, "soma_location": ""}
    validation_function = recenter

    def inputs(self):
        return {Sanitize: {"morph_path": "morph_path"}}


class Orient(StrIndexMixin, ElementValidationTask):
    """Orient morphologies.

    Sometimes, morphologies are oriented along non-standard axis. At BBP, the standard axis is
        - ``y`` for pia direction (apical trunk, minus main axon, etc...)
        - ``z`` is the slice direction is invitro
        - ``x`` the other direction

    If ``y`` is provided (default value), no oriention will be applied.
    One can also use minus sign, as ``-y`` to flip y axis.
    """

    output_columns = {"morph_path": None}
    validation_function = orient

    pia_direction = luigi.Parameter(
        default="y", description=":str: Axis of pia direction, x, y or z"
    )

    def kwargs(self):
        return {"pia_direction": self.pia_direction}

    def inputs(self):
        return {Recenter: {"morph_path": "morph_path"}}


class Align(StrIndexMixin, SkippableMixin(True), ElementValidationTask):
    """Align morphologies.

    Sometimes, a morphology is not aligned with any consistent direction, so we can try here to
    align them using a various algorithms from :func:`morph_tool.transform.align_morphology`.

    ``method`` can be: ``whole``, ``trunk``, ``first_segment``, ``first_section``, ``custom``

    If ``method`` is ``custom``, one must provide external orientation direction in a json file
    via the parameter ``custom_orientation_json_path``.


    TODO: implement the axon_collateral method (for thalamus or example), following
    https://bbpgitlab.epfl.ch/cells/archive/iavarone/thalamus_pipeline/-/blob/master/notebooks/cell_alignement/Rt_TC_rotations_AxonCollaterals_Releases2019.ipynb

    By default, this task is skipped.
    """

    output_columns = {"morph_path": None, "rotation_matrix": ""}
    validation_function = align

    method = luigi.ChoiceParameter(
        default="whole",
        choices=["whole", "trunk", "first_segment", "first_section", "custom"],
        description=":str: Method to align morphology",
    )
    neurite_type = luigi.Parameter(
        default="apical", description=":str: Neurite to use to align morphology"
    )
    direction = luigi.OptionalListParameter(
        default=None,
        schema={"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
    )
    custom_orientation_json_path = luigi.OptionalStrParameter(
        default=None, description=":str: Path to json with custom orientations"
    )

    def kwargs(self):
        return {
            "method": self.method,
            "neurite_type": self.neurite_type,
            "direction": self.direction,
            "custom_orientation_json_path": self.custom_orientation_json_path,
        }

    def inputs(self):
        return {Orient: {"morph_path": "morph_path"}}


class EnsureNeuritesOutsideSoma(StrIndexMixin, SkippableMixin(True), ElementValidationTask):
    """Fix radius of the soma of each morphology and cut the root section of neurites if needed."""

    output_columns = {"morph_path": None}
    validation_function = fix_neurites_in_soma

    def inputs(self):
        return {Align: {"morph_path": "morph_path"}}


class DetectErrors(StrIndexMixin, SkippableMixin(), ElementValidationTask):
    """Detect errors in reconstructions.

    Reconstructions may contain errors, which are detected here.
    They are of the following type:

    - fat ends
    - z-jumps
    - narrow start
    - dangling branch
    - multifurcation
    - z-range (new error only present here, which check is the z thickness is larger than min_range)

    This task uses :func:`neuror.sanitize.annotate_neurolucida`.
    This task creates new ``.asc`` file with error annotated so it can be red by Neuroluscida,
    and a :class:`morphology_workflows.marker_helper.MarkerSet` container of the errors, for later
    plotting.
    """

    output_columns = {
        "error_marker_path": None,
        "error_annotated_path": None,
        "error_summary": None,
    }
    validation_function = detect_errors

    min_range = luigi.FloatParameter(
        default=50, description=":float: Minimum z-range to be an error"
    )

    def kwargs(self):
        return {"min_range": self.min_range}

    def inputs(self):
        return {CheckNeurites: {"morph_path": "morph_path"}}


class PlotErrors(StrIndexMixin, SkippableMixin(), ElementValidationTask):
    """Plot detected errors.

    From the detected errors in :class:`tasks.curation.DetectErrors`, plot them on the morphologies.
    """

    output_columns = {"plot_errors_path": ""}
    validation_function = plot_errors

    with_plotly = BoolParameter(default=True, description=":bool: Use Plotly for plotting")

    def kwargs(self):
        return {"with_plotly": self.with_plotly}

    def inputs(self):
        return {
            Recenter: {"morph_path": "morph_path"},
            DetectErrors: {"error_marker_path": "error_marker_path"},
        }


class ErrorsReport(StrIndexMixin, SkippableMixin(), SetValidationTask):
    """Save error report for all morphologies.

    From the detected errors in :class:`tasks.curation.DetectErrors`, create a CSV report.
    """

    error_report_path = luigi.Parameter(
        default="error_report.csv", description=":str: Path to error report file in .csv"
    )
    validation_function = make_error_report

    def kwargs(self):
        return {"error_report_path": self.error_report_path}

    def inputs(self):
        return {
            DetectErrors: {"error_marker_path": "error_marker_path"},
        }


class Resample(StrIndexMixin, SkippableMixin(), ElementValidationTask):
    """Resample morphologies.

    This tasks ensures a constant sampling rate of points along all branches
    using the :func:`morph_tool.resampling.resample_linear_density` function.
    """

    output_columns = {"morph_path": None}
    validation_function = resample

    linear_density = luigi.FloatParameter(
        default=1.0, description=":float: Density of points per micron"
    )

    def kwargs(self):
        return {"linear_density": self.linear_density}

    def inputs(self):
        return {EnsureNeuritesOutsideSoma: {"morph_path": "morph_path"}}


class PlotMorphologies(StrIndexMixin, SkippableMixin(), ElementValidationTask):
    """Plot curated morphologies.

    This tasks creates a single pdf with all morphologies to visualise them after curation.

    The PDF file is created using the :func:`neurom.view.matplotlib_impl.plot_morph` function.
    """

    output_columns = {"plot_path": None}
    validation_function = plot_morphology

    with_plotly = BoolParameter(default=False, description=":bool: Use Plotly for plotting")
    with_realistic_diameters = BoolParameter(
        default=True,
        description=":bool: Use realistic diameters for plotting (take more time to plot)",
    )

    def kwargs(self):
        return {
            "with_plotly": self.with_plotly,
            "realistic_diameters": self.with_realistic_diameters,
        }

    def inputs(self):
        return {
            Resample: {"morph_path": "morph_path"},
        }
