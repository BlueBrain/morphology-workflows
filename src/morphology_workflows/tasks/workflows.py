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
from morphology_workflows.utils import StrIndexMixin


def path_reduced_df(data_dir, df_path="reduced_df.csv", parents=None):
    """Build the path to the reduced version of the report."""
    if parents:
        out_dir = list(data_dir.parents)[parents - 1]
    else:
        out_dir = data_dir
    return out_dir / df_path


def save_reduced_df(df, data_dir, df_path="reduced_df.csv", parents=None):
    """Save reduced version of the report."""
    _to_remove = ["exception", "is_valid", "ret_code", "comment"]
    target_path = path_reduced_df(data_dir, df_path=df_path, parents=parents)
    df.loc[
        df.is_valid, [col for col in df.columns if isinstance(col, str) and col not in _to_remove]
    ].rename_axis(index="morph_name").reset_index().to_csv(target_path, index=False)


class Curate(StrIndexMixin, ValidationWorkflow):
    """Run Curation phase.

    The complete phase has the following dependency graph:

    .. graphviz:: Curate.dot

    (click on steps in the image for more details)
    """

    __specifications__ = "Run the Curation phase."

    input_index_col = luigi.Parameter(default="morph_name")
    args = ["curated_dataset.csv", 2]

    validation_function = save_reduced_df

    def inputs(self):
        return {
            Collect: {},
            ExtractMarkers: {"marker_path": "marker_path"},
            Sanitize: {},
            CheckNeurites: {
                "has_axon": "has_axon",
                "has_basal": "has_basal",
                "has_apical": "has_apical",
            },
            Recenter: {"soma_location": "soma_location"},
            DetectErrors: {
                "error_marker_path": "error_marker_path",
                "error_annotated_path": "error_annotated_path",
            },
            PlotMarkers: {},
            PlotErrors: {},
            ErrorsReport: {},
            Align: {"rotation_matrix": "rotation_matrix"},
            EnsureNeuritesOutsideSoma: {},
            Orient: {},
            Resample: {"morph_path": "morph_path"},
            PlotMorphologies: {},
        }


class Annotate(StrIndexMixin, ValidationWorkflow):
    """Run Annotation phase.

    The complete phase has the following dependency graph:

    .. graphviz:: Annotate.dot

    (click on steps in the image for more details)
    """

    __specifications__ = "Run the Annotation phase."

    input_index_col = luigi.Parameter(default="morph_name")
    args = ["annotated_dataset.csv", 2]

    validation_function = save_reduced_df

    def inputs(self):
        if self.dataset_df is None:
            # If no dataset is given, the Curate workflow must be executed before.
            curate_task = Curate()
            input_path = path_reduced_df(
                curate_task.output()["data"].pathlib_path, *curate_task.args
            )
            Annotate.dataset_df.exists = False
            self.dataset_df = input_path
            CollectCurated.extra_requires = lambda x: curate_task
            Annotate.extra_requires = lambda x: curate_task

        return {
            CollectCurated: {},
            MType: {"mtype": "mtype"},
            HardLimit: {"hard_limit_path": "hard_limit_path"},
            ApicalPoint: {"apical_point_path": "apical_point_path"},
            CutLeaves: {"cut_leaves_path": "cut_leaves_path"},
            PlotApicalPoint: {},
            PlotCutLeaves: {},
            PlotHardLimit: {},
        }


class Repair(StrIndexMixin, ValidationWorkflow):
    """Run Repair phase.

    The complete phase has the following dependency graph:

    .. graphviz:: Repair.dot

    (click on steps in the image for more details)
    """

    __specifications__ = "Run the Repair phase."

    input_index_col = luigi.Parameter(default="morph_name")
    make_release = luigi.BoolParameter(
        default=True, description=":bool: Set to True to make a morpology release with neurondb.xml"
    )
    args = ["repaired_dataset.csv", 2]

    validation_function = save_reduced_df

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
        if self.dataset_df is None:
            # If no dataset is given, the Annotate workflow must be executed before.
            annotate_task = Annotate()
            input_path = path_reduced_df(
                annotate_task.output()["data"].pathlib_path, *annotate_task.args
            )
            Repair.dataset_df.exists = False
            self.dataset_df = input_path
            CollectAnnotated.extra_requires = lambda x: annotate_task
            Repair.extra_requires = lambda x: annotate_task

        inputs = {
            CollectAnnotated: {
                "morph_path": "morph_path",
                "apical_point_path": "apical_point_path",
                "cut_leaves_path": "cut_leaves_path",
            },
            FixZeroDiameters: {"morph_path": "zero_diameter_morph_path"},
            Unravel: {"morph_path": "unravel_morph_path"},
            RepairNeurites: {"morph_path": "repair_morph_path"},
            SmoothDiameters: {"morph_path": "smooth_morph_path"},
            PlotSmoothDiameters: {},
            PlotRepair: {},
            MakeCollage: {},
        }

        if self.make_release:
            folders = [
                "zero_diameter_morph_db_path",
                "unravel_morph_db_path",
                "repair_morph_db_path",
                "zero_diameter_release_morph_path",
                "unravel_release_morph_path",
                "repair_release_morph_path",
            ]
            mapping = {}
            for extension in MakeRelease.extensions:
                ext = extension[1:]
                mapping.update({f"{folder}_{ext}": f"{folder}_{ext}" for folder in folders})
            inputs[MakeRelease] = mapping

        return inputs
