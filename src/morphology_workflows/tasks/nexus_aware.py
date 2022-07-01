"""Nexus aware dvf tasks."""
import warnings
from pathlib import Path

import luigi
import pandas as pd
from luigi_tools.parameter import OptionalPathParameter
from data_validation_framework.target import ReportTarget
from data_validation_framework.task import INDEX_LABEL
from data_validation_framework.task import BaseValidationTask
from data_validation_framework.task import ElementValidationTask as _Element_task
from data_validation_framework.task import SetValidationTask as _Set_task
from data_validation_framework.task import ValidationWorkflow as _Workflow_task
from kgforge.core import KnowledgeGraphForge  # pylint: disable=import-error


class NexusConfig(luigi.Config):
    """Configuration for Nexus Forge."""

    organisation = luigi.Parameter(default="public")
    project = luigi.Parameter(default="sscx")
    token = luigi.Parameter(default=None)
    endpoint = luigi.OptionalParameter(default=None)
    nexus_yml = luigi.Parameter(
        default="https://raw.githubusercontent.com/BlueBrain/nexus-forge/master/examples/notebooks/use-cases/prod-forge-nexus.yml"  # noqa, pylint: disable=line-too-long
    )
    read_mode = luigi.BoolParameter(default=True)
    write_mode = luigi.BoolParameter(default=False)


class NexusBaseValidationTask(BaseValidationTask):  # pylint: disable=abstract-method
    """Nexus aware Base validation task."""

    nexus_dataset_version = luigi.Parameter(default=None)
    nexus_dataset_url = luigi.Parameter(default=None, description=":str: Url to the input dataset.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        NexusBaseValidationTask.forge = None
        if NexusConfig().token is not None:
            NexusBaseValidationTask.forge = self._set_forge()
            # uncomment to remove old tries
            # self.forge.deprecate(self.forge.search({'type': 'NeuronMorphology'}))

    @classmethod
    def _set_forge(cls):
        """Set a forge instance."""
        if not cls.forge:
            kwargs = dict(
                bucket=f"{NexusConfig().organisation}/{NexusConfig().project}",
                token=NexusConfig().token,
            )
            if NexusConfig().endpoint is not None:
                kwargs["endpoint"] = NexusConfig().endpoint
            cls.forge = KnowledgeGraphForge(NexusConfig().nexus_yml, **kwargs)
        return cls.forge

    def read_dataset(self):
        """Can read dataset from nexus."""
        read_forge = self.forge is not None and NexusConfig().read_mode
        if read_forge is None and self.nexus_dataset_url is None:
            raise Exception("Nexus Forge is not set and dataset_df is an url!")
        if read_forge and self.nexus_dataset_url is not None:
            dataset = self.forge.retrieve(
                self.nexus_dataset_url, version=self.nexus_dataset_version
            )
            df = self.forge.as_dataframe([self.forge.retrieve(p.id) for p in dataset.hasPart])
            print(df.columns)
            # distr[0] is to get .asc file ([1] is .swc), but better to use encodingFormat entry
            # [7:] is to get rid of file://, but can we do it better?
            df["morph_path"] = df["distribution"].apply(
                lambda distr: distr[0]["atLocation"]["location"][7:]
            )
            df[self.input_index_col] = df["name"]
            if "annotation.hasBody.label" in df.columns:
                df["mtype"] = df["annotation.hasBody.label"]
            else:
                df["mtype"] = "undefined"
            return df[[self.input_index_col, "morph_path", "mtype"]].set_index(self.input_index_col)
        return super().read_dataset()

    def run(self):
        """Can run in nexus write mode."""
        if not NexusConfig().write_mode:
            return super().run()
        return self._run_nexus()

    def _add_new_morphs(self, res, morph_col, task_name, parents=None):
        """Add new morphology resource to nexus."""
        _res = pd.DataFrame()
        _res["name"] = res.index.to_list()
        _res["type"] = "NeuronMorphology"
        _res["mcar.task"] = task_name
        if parents is not None:
            _res["parent"] = parents
        resources = self.forge.from_dataframe(_res)
        for path, resource in zip(res[morph_col].to_list(), resources):
            resource.distribution = self.forge.attach(str(Path(path).absolute()))

        self.forge.register(resources)
        nexus_ids = [r.id for r in resources]
        return nexus_ids, resources

    def _run_nexus(self):
        """Main run method to write to nexus and save nexus_report."""
        if self.forge is None and NexusConfig().write_mode:
            raise ValueError("Please define a forge to write to Nexus")
        task_name = type(self).__name__

        if not self.output()["report"].pathlib_path.exists():
            raise Exception("Please run the workflow in non nexus write mode first")

        # load report of the task
        res = pd.read_csv(self.output()["report"].pathlib_path, index_col=INDEX_LABEL)

        # create new resource, or gather nexus id from parent task
        nexus_ids = None
        if self.input():
            _df_input = pd.read_csv(self.input()[0]["nexus_report"].path, index_col=INDEX_LABEL)
            nexus_ids = _df_input.nexus_id.to_list()

        # if we have no resources add the morphologies (this happens in Collect tasks)
        if nexus_ids is None:
            nexus_ids, resources = self._add_new_morphs(res, "morph_path", task_name)
        else:
            # else retrieve resources locally
            resources = [self.forge.retrieve(_id) for _id in nexus_ids]

        # update local nexus_id
        res["nexus_id"] = nexus_ids

        # add new data
        if hasattr(self, "output_columns") and self.output_columns is not None:
            # if new data is a new morphology, we create a new resource
            path_col = [col.endswith("morph_path") for col in self.output_columns.keys()]
            if any(path_col):
                if len([p for p in path_col if p]) > 1:
                    warnings.warn(
                        f"More than one morph_path entry in {task_name}, we will use the first."
                    )
                    path_index = path_col.index(True)[0]
                else:
                    path_index = path_col.index(True)

                morph_col = list(self.output_columns.keys())[path_index]
                nexus_ids, resources = self._add_new_morphs(
                    res, morph_col, task_name, parents=nexus_ids
                )
                # update local nexus_id for new resources
                res["nexus_id"] = nexus_ids
            # if new data is str in a column, we just add it
            for col in self.output_columns.keys():
                if not col.endswith("morph_path"):
                    _res = res.set_index("nexus_id")
                    for resource in resources:
                        resource.mcar = self.forge.from_json(
                            {task_name: {col: str(_res.loc[resource.id, col])}}
                        )
                    self.forge.update(resources)

        # save df as nexus_report
        res.to_csv(self.output()["nexus_report"].path, index=True, index_label=INDEX_LABEL)

    def output(self):
        """Adds a nexus_report target to save nexus aware information."""
        output = super().output()
        if NexusConfig().write_mode:
            class_path = Path(self.task_name)
            prefix = None if self.result_path is None else self.result_path.absolute()
            output["nexus_report"] = ReportTarget(
                class_path / "nexus_report.csv",
                prefix=prefix,
                create_parent=False,
                task_name=self.task_name,
            )
        return output


class ElementValidationTask(NexusBaseValidationTask, _Element_task):
    """Nexus aware Element validation task."""


class SetValidationTask(NexusBaseValidationTask, _Set_task):
    """Nexus aware Set validation task."""


class ValidationWorkflow(NexusBaseValidationTask, _Workflow_task):
    """Nexus aware Validation Workflow task."""
