"""Transformation phase (Optional)."""
import logging

import luigi
from luigi_tools.parameter import BoolParameter
from morphology_workflows.curation import collect
from morphology_workflows.repair import make_release

from morphology_processing_workflow.tasks import ElementValidationTask
from morphology_processing_workflow.tasks import SetValidationTask
from morphology_processing_workflow.transform import apply_transformation
from morphology_processing_workflow.transform import compare_transformed
from morphology_processing_workflow.transform import learn_diameter_transform
from morphology_processing_workflow.transform import learn_morphology_transform
from morphology_processing_workflow.transform import learn_soma_transform

logger = logging.getLogger(__name__)


class CollectSourceDataset(ElementValidationTask):
    """Collect source dataset to transform."""

    output_columns = {
        "morph_path": None,
        "has_apical": None,
        "has_axon": None,
        "has_basal": None,
        "cut_leaves_path": None,
        "apical_point_path": None,
        "mtype": None,
    }

    validation_function = collect


class CollectTargetDataset(ElementValidationTask):
    """Collect target dataset to learn the transformation."""

    output_columns = {
        "morph_path": None,
        "has_apical": None,
        "has_axon": None,
        "has_basal": None,
        "cut_leaves_path": None,
        "apical_point_path": None,
        "mtype": None,
    }
    input_index_col = luigi.Parameter(default="morph_name")

    validation_function = collect


class LearnSomaTransform(SetValidationTask):
    """Learn how to transform soma from source to target datasets."""

    per_mtype = BoolParameter(default=False)
    method = luigi.ChoiceParameter(default="ratio_of_means", choices=["ratio_of_means"])
    plot = BoolParameter(default=True)

    def kwargs(self):
        return {
            "target_dataset": self.extra_input()["report"].path,
            "method": self.method,
            "per_mtype": self.per_mtype,
            "plot": self.plot,
        }

    def extra_requires(self):
        return CollectTargetDataset(result_path=self.result_path)

    def inputs(self):
        return {CollectSourceDataset: {"morph_path": "morph_path"}}

    output_columns = {"soma_transform": None}
    validation_function = learn_soma_transform


class LearnMorphologyTransform(SetValidationTask):
    """Learn how to transform morphologies."""

    per_mtype = BoolParameter(default=False)
    method = luigi.ChoiceParameter(default="cortical", choices=["cortical", "extents_no_axon"])
    extra_data = luigi.DictParameter(default={"scaling": 0.66})
    plot = BoolParameter(default=True)

    def inputs(self):
        return {CollectSourceDataset: {"morph_path": "morph_path"}}

    def extra_requires(self):
        return CollectTargetDataset(result_path=self.result_path)

    def kwargs(self):
        return {
            "target_dataset": self.extra_input()["report"].path,
            "method": self.method,
            "per_mtype": self.per_mtype,
            "extra_data": self.extra_data,
            "plot": self.plot,
        }

    output_columns = {"morphology_transform": None}
    validation_function = learn_morphology_transform


class LearnDiameterTransform(SetValidationTask):
    """Learn how to transform diameters."""

    per_mtype = BoolParameter(default=False)
    method = luigi.ChoiceParameter(default="branch_order", choices=["branch_order"])
    plot = BoolParameter(default=True)
    max_x = luigi.FloatParameter(default=30)

    def inputs(self):
        return {CollectSourceDataset: {"morph_path": "morph_path"}}

    def extra_requires(self):
        return CollectTargetDataset(result_path=self.result_path)

    def kwargs(self):
        return {
            "target_dataset": self.extra_input()["report"].path,
            "method": self.method,
            "per_mtype": self.per_mtype,
            "plot": self.plot,
            "max_x": self.max_x,
        }

    output_columns = {"diameter_transform": None}
    validation_function = learn_diameter_transform


class ApplyTransformation(ElementValidationTask):
    """Apply the transformation on source dataset."""

    def inputs(self):
        return {
            CollectSourceDataset: {"morph_path": "morph_path"},
            LearnSomaTransform: {"soma_transform": "soma_transform"},
            LearnMorphologyTransform: {"morphology_transform": "morphology_transform"},
            LearnDiameterTransform: {"diameter_transform": "diameter_transform"},
        }

    output_columns = {"transformed_morph_path": None}
    validation_function = apply_transformation


class CompareTransformed(SetValidationTask):
    """Compare the transformed morphologies by simple plots."""

    def extra_requires(self):
        return CollectTargetDataset(result_path=self.result_path)

    def kwargs(self):
        return {"target_dataset": self.extra_input()["report"].path}

    def inputs(self):
        return {
            CollectSourceDataset: {"morph_path": "morph_path"},
            ApplyTransformation: {"transformed_morph_path": "transformed_morph_path"},
        }

    validation_function = compare_transformed


class MakeTransformRelease(SetValidationTask):
    """Make a morpology release, with only repair."""

    release_path = luigi.OptionalParameter(
        default=None,
        description=":str: Path to the directory in which all the releases will be exported",
    )
    repair_path = luigi.Parameter(
        default="repaired_release",
        description=":str: Path to repaired morphologies (not created if None)",
    )
    extensions = [".asc", ".h5", ".swc"]
    output_columns = {"layer": None}
    for extension in extensions:
        ext = extension[1:]
        output_columns.update(
            {
                f"repair_morph_db_path_{ext}": None,
                f"repair_release_morph_path_{ext}": None,
            }
        )

    validation_function = make_release

    def kwargs(self):
        return {
            "release_path": self.release_path,
            "zero_diameter_path": None,
            "unravel_path": None,
            "repair_path": self.repair_path,
            "extensions": self.extensions,
        }

    def inputs(self):
        return {
            ApplyTransformation: {"transformed_morph_path": "repair_morph_path"},
            CollectSourceDataset: {
                "mtype": "mtype",
                "has_apical": "has_apical",
                "has_axon": "has_axon",
                "has_basal": "has_basal",
            },
        }
