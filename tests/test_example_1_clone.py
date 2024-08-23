"""Test the complete workflow on the example."""
# pylint: disable=redefined-outer-name
import shutil

import luigi
import numpy as np
import pytest
from dir_content_diff import assert_equal_trees

from morphology_processing_workflow.tasks import workflows


@pytest.fixture()
def example_1(tmp_working_dir, examples_dir):
    """Setup the working directory."""
    shutil.copyfile(examples_dir / "logging.conf", tmp_working_dir / "logging.conf")
    shutil.copyfile(examples_dir / "luigi.cfg", tmp_working_dir / "luigi.cfg")
    shutil.copyfile(examples_dir / "dataset.csv", tmp_working_dir / "dataset.csv")
    shutil.copyfile(examples_dir / "builder_recipe.xml", tmp_working_dir / "builder_recipe.xml")
    shutil.copyfile(examples_dir / "placement_rules.xml", tmp_working_dir / "placement_rules.xml")
    shutil.copyfile(examples_dir / "transform_rules.xml", tmp_working_dir / "transform_rules.xml")
    shutil.copytree(examples_dir / "out_repaired", tmp_working_dir / "out_repaired")
    shutil.copytree(examples_dir / "repair_release", tmp_working_dir / "repair_release")

    # Set current config in luigi
    luigi_config = luigi.configuration.get_config()
    luigi_config.read("./luigi.cfg")

    yield tmp_working_dir

    # Reset luigi config
    luigi_config.clear()


def test_example_1(example_1, data_dir):
    """Test the workflow on the example."""
    np.random.seed(0)
    out_dir_pattern = (str(example_1) + "/?", "")

    # Run the Clone workflow
    assert luigi.build([workflows.Clone()], local_scheduler=True)

    # Define post-processes to check the results
    clone_expected_dir = data_dir / "test_example_1" / "out_clone"
    clone_specific_args = {
        "clone_dataset.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "annotation_path",
                        "apical_point_path",
                        "cut_leaves_path",
                        "error_annotated_path",
                        "error_marker_path",
                        "hard_limit_path",
                        "marker_path",
                        "morph_path",
                        "smooth_morph_path",
                        "repair_morph_db_path_h5",
                        "repair_morph_path_h5",
                        "unravel_morph_db_path",
                        "unravel_morph_path",
                        "zero_diameter_morph_db_path",
                        "zero_diameter_morph_path",
                    ]
                }
            }
        },
        "Clone/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "('CloneMorphologies', 'annotation_path')",
                        "('CloneMorphologies', 'morph_path')",
                        "('CollectAnnotations', 'annotation_path')",
                        "('CollectRepaired', 'apical_point_path')",
                        "('CollectRepaired', 'cut_leaves_path')",
                        "('CollectRepaired', 'morph_path')",
                        "annotation_path",
                        "morph_path",
                    ],
                },
            }
        },
        "CloneMorphologies/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "annotation_path",
                        "morph_path",
                    ],
                },
            }
        },
        "CollectAnnotations/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "annotation_path",
                    ],
                },
            }
        },
        "CollectRepaired/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "apical_point_path",
                        "cut_leaves_path",
                        "morph_path",
                    ],
                },
            }
        },
    }

    # Check the results
    assert_equal_trees(
        clone_expected_dir, example_1 / "out_clone", specific_args=clone_specific_args
    )
