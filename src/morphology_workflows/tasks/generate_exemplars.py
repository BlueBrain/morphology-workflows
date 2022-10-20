"""Workflow to create exemplar morphologies."""
import yaml
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from data_validation_framework.task import TagResultOutputMixin
from luigi.parameter import OptionalParameter
from luigi.parameter import PathParameter
from luigi.parameter import ChoiceParameter
from luigi_tools.task import WorkflowTask

from morphology_workflows.generate_exemplars import single_compartment_exemplar
from morphology_workflows.generate_exemplars import full_exemplar


class GenerateExemplar(TagResultOutputMixin, WorkflowTask):
    """Generate exemplar morphologies per mtypes."""

    input_morph_df_path = PathParameter(
        description=":str: Path to dataset csv file with morphologies", exists=True, default=None
    )
    exemplar_morphology = PathParameter(
        default="exemplar_morphologies", description=":str: Path to the exemplar morphology."
    )
    mtype = OptionalParameter(
        default=None,
        description=":str: The mtype to consider.",
    )
    mode = ChoiceParameter(
        default="single_compartment",
        choices=["single_compartment", "full"],
        description=":str: Mode to generate exemplar",
    )

    def run(self):
        morph_df = pd.read_csv(self.input_morph_df_path)
        self.output().pathlib_path.mkdir(exist_ok=True)
        mtype = "no_mtype"
        if "mtype" in morph_df.columns:
            morph_df = morph_df[morph_df.mtype == self.mtype]
            mtype = self.mtype

        if self.mode == "single_compartment":
            exemplar = single_compartment_exemplar(morph_df)
            name = f"exemplar_{mtype}.swc"
            exemplar.write(self.output().pathlib_path / name)
            exemplar_data = {"path": name}

        elif self.mode == "full":
            exemplar_data = full_exemplar(morph_df, figure_folder=self.output().path)
        else:
            raise Exception(f"Mode {self.mode} not understood.")

        yaml.safe_dump(
            exemplar_data, open(self.output().pathlib_path / f"exemplar_data_{mtype}.yaml", "w")
        )

    def output(self):
        return TaggedOutputLocalTarget(self.exemplar_morphology)
