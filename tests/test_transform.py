"""Test the transform phase."""

# pylint: disable=redefined-outer-name
import shutil

import luigi
import numpy as np
import pytest
from dir_content_diff import assert_equal_trees

from morphology_workflows.tasks import workflows


@pytest.fixture
def example_transform(tmp_working_dir, examples_dir):
    """Setup the working directory."""
    shutil.copyfile(examples_dir / "transform" / "logging.conf", tmp_working_dir / "logging.conf")
    shutil.copyfile(examples_dir / "transform" / "luigi.cfg", tmp_working_dir / "luigi.cfg")
    shutil.copyfile(
        examples_dir / "transform" / "transform_dataset.csv",
        tmp_working_dir / "transform_dataset.csv",
    )
    shutil.copyfile(
        examples_dir / "transform" / "mouse_dataset.csv", tmp_working_dir / "mouse_dataset.csv"
    )
    shutil.copytree(examples_dir / "transform" / "morphologies", tmp_working_dir / "morphologies")
    shutil.copytree(
        examples_dir / "transform" / "mouse_morphologies", tmp_working_dir / "mouse_morphologies"
    )

    # Set current config in luigi
    luigi_config = luigi.configuration.get_config()
    luigi_config.read("./luigi.cfg")

    yield tmp_working_dir

    # Reset luigi config
    luigi_config.clear()


def test_transform(example_transform, data_dir):
    """Test the transform step on the example."""
    np.random.seed(0)

    # Run the Transform workflow
    assert luigi.build([workflows.Transform()], local_scheduler=True)

    # Define post-processes to check the results
    out_dir_pattern = (str(example_transform) + "/?", "")
    transform_expected_dir = data_dir / "test_transform" / "out_transformed"
    transformed_specific_args = {
        "transformed_dataset.csv": {
            "format_data_kwargs": {"replace_pattern": {out_dir_pattern: ["morph_path"]}}
        },
        "Transform/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "morph_path",
                        "('ApplyTransformation', 'morph_path')",
                        "('ApplyTransformation', 'transformed_morph_path')",
                        "('ApplyTransformation', 'diameter_transform')",
                        "('CompareTransformed', 'morph_path')",
                        "('CompareTransformed', 'transformed_morph_path')",
                    ]
                }
            }
        },
        "CollectSourceDataset/report.csv": {
            "format_data_kwargs": {"replace_pattern": {out_dir_pattern: ["morph_path"]}}
        },
        "CollectTargetDataset/report.csv": {
            "format_data_kwargs": {"replace_pattern": {out_dir_pattern: ["morph_path"]}}
        },
        "LearnDiameterTransform/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {out_dir_pattern: ["morph_path", "diameter_transform"]}
            }
        },
        "LearnMorphologyTransform/report.csv": {
            "format_data_kwargs": {"replace_pattern": {out_dir_pattern: ["morph_path"]}}
        },
        "LearnSomaTransform/report.csv": {
            "format_data_kwargs": {"replace_pattern": {out_dir_pattern: ["morph_path"]}}
        },
        "ApplyTransformation/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: ["morph_path", "transformed_morph_path", "diameter_transform"]
                }
            }
        },
        "CompareTransformed/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {out_dir_pattern: ["morph_path", "transformed_morph_path"]}
            }
        },
    }

    # Check the results
    assert_equal_trees(
        transform_expected_dir,
        example_transform / "out_transformed",
        specific_args=transformed_specific_args,
    )
