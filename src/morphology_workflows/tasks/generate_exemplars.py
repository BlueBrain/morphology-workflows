"""Workflow to create exemplar morphologies."""
import yaml
import json
from collections import defaultdict
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
    cell_composition_path = PathParameter(
        default="cell_composition.yaml", description=":str: Path to the cell composition."
    )
    mode = ChoiceParameter(
        default="single_compartment",
        choices=["single_compartment", "full"],
        description=":str: Mode to generate exemplar",
    )

    def run(self):
        morph_df = pd.read_csv(self.input_morph_df_path)
        if "path" not in morph_df:
            morph_df["path"] = morph_df["morph_path"]

        self.output().pathlib_path.mkdir(exist_ok=True)

        mapping = {"all": None}
        if self.mtype is None and self.cell_composition_path.exists():
            with open(self.cell_composition_path) as f:
                cell_composition = yaml.safe_load(f)["neurons"]
            mapping = defaultdict(list)
            for metype in cell_composition:
                etype = metype["traits"]["etype"]
                mtype = metype["traits"]["mtype"]
                if etype == "cADpyr":
                    layer = metype["traits"]["layer"]
                    if mtype[1] == layer:
                        etype = f"cADpyr_L{layer}"
                        mapping[etype].append(mtype)
                else:
                    mapping[etype].append(mtype)

        elif self.mtype is not None:
            morph_df = morph_df[morph_df.mtype == self.mtype].reset_index(drop=True)
            mapping = {self.mtype: [self.mtype]}

        print(json.dumps(mapping, indent=4))
        with (self.output().pathlib_path / "mapping.json").open("w") as f:
            json.dump(mapping, f, indent=4)

        exemplar_data = {}
        for exemplar_name, mtypes in mapping.items():
            _morph_df = morph_df[morph_df.mtype.isin(mtypes)].reset_index(drop=True).copy()

            if self.mode == "single_compartment":
                exemplar = single_compartment_exemplar(_morph_df)
                name = f"exemplar_{exemplar_name}.swc"
                exemplar.write(self.output().pathlib_path / name)
                exemplar_data = {"path": name}

            elif self.mode == "full":
                out_folder = self.output().pathlib_path / f"exemplar_{exemplar_name}"
                out_folder.mkdir(exist_ok=True)
                exemplar_data[exemplar_name] = full_exemplar(_morph_df, figure_folder=out_folder)
            else:
                raise Exception(f"Mode {self.mode} not understood.")

        yaml.safe_dump(exemplar_data, open(self.output().pathlib_path / "exemplar_data.yaml", "w"))

    def output(self):
        return TaggedOutputLocalTarget(self.exemplar_morphology)
