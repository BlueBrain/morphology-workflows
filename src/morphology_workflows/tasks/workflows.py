"""Workflow tasks."""
import luigi
from data_validation_framework.task import ValidationWorkflow

from morphology_workflows.tasks.annotation import ApicalPoint
from morphology_workflows.tasks.annotation import CollectCurated
from morphology_workflows.tasks.annotation import CutLeaves
from morphology_workflows.tasks.annotation import HardLimit
from morphology_workflows.tasks.annotation import MType
from morphology_workflows.tasks.annotation import PlotApicalPoint
from morphology_workflows.tasks.annotation import PlotCutLeaves
from morphology_workflows.tasks.annotation import PlotHardLimit
from morphology_workflows.tasks.curation import Align
from morphology_workflows.tasks.curation import CheckNeurites
from morphology_workflows.tasks.curation import Collect
from morphology_workflows.tasks.curation import DetectErrors
from morphology_workflows.tasks.curation import EnsureNeuritesOutsideSoma
from morphology_workflows.tasks.curation import ErrorsReport
from morphology_workflows.tasks.curation import ExtractMarkers
from morphology_workflows.tasks.curation import Orient
from morphology_workflows.tasks.curation import PlotErrors
from morphology_workflows.tasks.curation import PlotMarkers
from morphology_workflows.tasks.curation import PlotMorphologies
from morphology_workflows.tasks.curation import Recenter
from morphology_workflows.tasks.curation import Resample
from morphology_workflows.tasks.curation import Sanitize
from morphology_workflows.tasks.repair import CollectAnnotated
from morphology_workflows.tasks.repair import FixZeroDiameters
from morphology_workflows.tasks.repair import MakeCollage
from morphology_workflows.tasks.repair import MakeRelease
from morphology_workflows.tasks.repair import PlotRepair
from morphology_workflows.tasks.repair import PlotSmoothDiameters
from morphology_workflows.tasks.repair import RepairNeurites
from morphology_workflows.tasks.repair import SmoothDiameters
from morphology_workflows.tasks.repair import Unravel


def save_reduced_df(
    df, data_dir, df_path="reduced_df.csv", parents=None
):  # pylint: disable=unused-argument
    """Save reduced version of the report."""
    _to_remove = ["exception", "is_valid", "ret_code", "comment"]
    if parents:
        out_dir = list(data_dir.parents)[parents - 1]
    else:
        out_dir = data_dir
    df.loc[
        df.is_valid, [col for col in df.columns if isinstance(col, str) and col not in _to_remove]
    ].rename_axis(index="morph_name").reset_index().to_csv(out_dir / df_path, index=False)


class Curate(ValidationWorkflow):
    """Run Curation phase.

    The complete phase has the following dependency graph:

    .. graphviz:: Curate.dot
    """

    __specifications__ = "Run the Curation phase."

    input_index_col = luigi.Parameter(default="morph_name")
    args = ["curated_dataset.csv", 2]

    validation_function = staticmethod(save_reduced_df)

    def inputs(self):
        """ """
        return {
            Collect(): {},
            ExtractMarkers(): {"marker_path": "marker_path"},
            Sanitize(): {},
            CheckNeurites(): {
                "has_axon": "has_axon",
                "has_basal": "has_basal",
                "has_apical": "has_apical",
            },
            Recenter(): {"soma_location": "soma_location"},
            DetectErrors(): {
                "error_marker_path": "error_marker_path",
                "error_annotated_path": "error_annotated_path",
            },
            PlotMarkers(): {},
            PlotErrors(): {},
            ErrorsReport(): {},
            Align(): {"rotation_matrix": "rotation_matrix"},
            EnsureNeuritesOutsideSoma(): {},
            Orient(): {},
            Resample(): {"morph_path": "morph_path"},
            PlotMorphologies(): {},
        }


class Annotate(ValidationWorkflow):
    """Run Annotation phase.

    The complete phase has the following dependency graph:

    .. graphviz:: Annotate.dot
    """

    __specifications__ = "Run the Annotation phase."

    input_index_col = luigi.Parameter(default="morph_name")
    args = ["annotated_dataset.csv", 2]

    validation_function = staticmethod(save_reduced_df)

    def inputs(self):
        """ """
        return {
            CollectCurated(): {},
            MType(): {"mtype": "mtype"},
            HardLimit(): {"hard_limit_path": "hard_limit_path"},
            ApicalPoint(): {"apical_point_path": "apical_point_path"},
            CutLeaves(): {"cut_leaves_path": "cut_leaves_path"},
            PlotApicalPoint(): {},
            PlotCutLeaves(): {},
            PlotHardLimit(): {},
        }


class Repair(ValidationWorkflow):
    """Run Repair phase.

    The complete phase has the following dependency graph:

    .. graphviz:: Repair.dot
    """

    __specifications__ = "Run the Repair phase."

    input_index_col = luigi.Parameter(default="morph_name")
    make_release = luigi.BoolParameter(
        default=True, description=":bool: Set to True to make a morpology release with neurondb.xml"
    )
    args = ["repaired_dataset.csv", 2]

    validation_function = staticmethod(save_reduced_df)

    report_config = {
        "extensions": [
            "sphinx.ext.graphviz",
            "sphinx.ext.intersphinx",
            "sphinx.ext.napoleon",
            "sphinx.ext.todo",
            "rst2pdf.pdfbuilder",
        ]
    }

    def inputs(self):
        """ """
        inputs = {
            CollectAnnotated(): {
                "morph_path": "morph_path",
                "apical_point_path": "apical_point_path",
                "cut_leaves_path": "cut_leaves_path",
            },
            FixZeroDiameters(): {"morph_path": "zero_diameter_morph_path"},
            Unravel(): {"morph_path": "unravel_morph_path"},
            RepairNeurites(): {"morph_path": "repair_morph_path"},
            SmoothDiameters(): {"morph_path": "smooth_morph_path"},
            PlotSmoothDiameters(): {},
            PlotRepair(): {},
            MakeCollage(): {},
        }

        if self.make_release:
            inputs[MakeRelease()] = {
                "zero_diameter_morph_db_path": "zero_diameter_morph_db_path",
                "unravel_morph_db_path": "unravel_morph_db_path",
                "repair_morph_db_path": "repair_morph_db_path",
                "zero_diameter_release_morph_path": "zero_diameter_release_morph_path",
                "unravel_release_morph_path": "unravel_release_morph_path",
                "repair_release_morph_path": "repair_release_morph_path",
            }

        return inputs
