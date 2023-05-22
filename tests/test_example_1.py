"""Test the complete workflow on the example."""
# pylint: disable=redefined-outer-name
import shutil

import luigi
import numpy as np
import pandas as pd
import pytest
from dir_content_diff import assert_equal_trees
from dir_content_diff import compare_files
from dir_content_diff.base_comparators import IniComparator
from dir_content_diff.base_comparators import JsonComparator

from morphology_workflows.tasks import cli
from morphology_workflows.tasks import workflows
from morphology_workflows.utils import EXTS

from . import clean_exception


@pytest.fixture()
def example_1(tmp_working_dir, examples_test_dir):
    """Setup the working directory."""
    shutil.copyfile(examples_test_dir / "dataset.csv", tmp_working_dir / "dataset.csv")
    shutil.copytree(examples_test_dir / "morphologies", tmp_working_dir / "morphologies")
    cli.main(["Initialize"])

    # Set current config in luigi
    luigi_config = luigi.configuration.get_config()
    luigi_config.read("./luigi.cfg")

    yield tmp_working_dir

    # Reset luigi config
    luigi_config.clear()


def test_example_1(example_1, data_dir):
    """Test the workflow on the example."""
    np.random.seed(0)

    # Run the Curate workflow
    assert luigi.build([workflows.Curate()], local_scheduler=True)

    # Define post-processes to check the results
    out_dir_pattern = (str(example_1) + "/?", "")
    curation_expected_dir = data_dir / "test_example_1" / "out_curated"
    curated_specific_args = {
        "curated_dataset.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "error_annotated_path",
                        "error_marker_path",
                        "marker_path",
                        "morph_path",
                    ]
                }
            }
        },
        "Curate/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "('Align', 'morph_path')",
                        "('CheckNeurites', 'morph_path')",
                        "('Collect', 'morph_path')",
                        "('DetectErrors', 'error_annotated_path')",
                        "('DetectErrors', 'error_marker_path')",
                        "('DetectErrors', 'morph_path')",
                        "('ExtractMarkers', 'marker_path')",
                        "('ExtractMarkers', 'morph_path')",
                        "('Orient', 'morph_path')",
                        "('EnsureNeuritesOutsideSoma', 'morph_path')",
                        "('PlotErrors', 'error_marker_path')",
                        "('PlotErrors', 'morph_path')",
                        "('PlotErrors', 'plot_errors_path')",
                        "('PlotMarkers', 'marker_path')",
                        "('PlotMarkers', 'plot_marker_path')",
                        "('PlotMorphologies', 'morph_path')",
                        "('PlotMorphologies', 'plot_path')",
                        "('Recenter', 'morph_path')",
                        "('Resample', 'morph_path')",
                        "('Sanitize', 'comment')",
                        "('Sanitize', 'morph_path')",
                        "('ErrorsReport', 'error_marker_path')",
                        "error_annotated_path",
                        "error_marker_path",
                        "marker_path",
                        "morph_path",
                    ],
                },
            }
        },
        "Collect/report.csv": {
            "format_data_kwargs": {"replace_pattern": {out_dir_pattern: ["morph_path"]}}
        },
        "CheckNeurites/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: ["morph_path"],
                },
            }
        },
        "ExtractMarkers/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: ["morph_path", "marker_path"],
                },
            }
        },
        "PlotMarkers/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: ["marker_path", "plot_marker_path"],
                }
            }
        },
        "Recenter/report.csv": {
            "format_data_kwargs": {"replace_pattern": {out_dir_pattern: ["morph_path"]}}
        },
        "Orient/report.csv": {
            "format_data_kwargs": {"replace_pattern": {out_dir_pattern: ["morph_path"]}}
        },
        "Align/report.csv": {
            "format_data_kwargs": {"replace_pattern": {out_dir_pattern: ["morph_path"]}}
        },
        "EnsureNeuritesOutsideSoma/report.csv": {
            "format_data_kwargs": {"replace_pattern": {out_dir_pattern: ["morph_path"]}}
        },
        "DetectErrors/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "morph_path",
                        "error_annotated_path",
                        "error_marker_path",
                    ]
                }
            }
        },
        "PlotErrors/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: ["morph_path", "error_marker_path", "plot_errors_path"]
                }
            }
        },
        "Sanitize/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {out_dir_pattern: ["morph_path", "comment"]},
            }
        },
        "PlotMorphologies/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {out_dir_pattern: ["morph_path", "plot_path"]}
            }
        },
    }

    # Check the results
    assert_equal_trees(
        curation_expected_dir, example_1 / "out_curated", specific_args=curated_specific_args
    )

    # Run the Annotate workflow
    assert luigi.build([workflows.Annotate()], local_scheduler=True)

    # Define post-processes to check the results
    annotation_expected_dir = data_dir / "test_example_1" / "out_annotated"
    annotated_specific_args = {
        "annotated_dataset.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "apical_point_path",
                        "cut_leaves_path",
                        "error_annotated_path",
                        "error_marker_path",
                        "hard_limit_path",
                        "marker_path",
                        "morph_path",
                    ]
                }
            }
        },
        "Annotate/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "('ApicalPoint', 'apical_point_path')",
                        "('ApicalPoint', 'morph_path')",
                        "('CollectCurated', 'morph_path')",
                        "('CutLeaves', 'cut_leaves_path')",
                        "('CutLeaves', 'morph_path')",
                        "('HardLimit', 'hard_limit_path')",
                        "('HardLimit', 'morph_path')",
                        "('PlotApicalPoint', 'apical_point_path')",
                        "('PlotApicalPoint', 'morph_path')",
                        "('PlotApicalPoint', 'plot_apical_point_path')",
                        "('PlotCutLeaves', 'cut_leaves_path')",
                        "('PlotCutLeaves', 'morph_path')",
                        "('PlotCutLeaves', 'plot_cut_leaves_path')",
                        "('PlotHardLimit', 'hard_limit_path')",
                        "('PlotHardLimit', 'morph_path')",
                        "('PlotHardLimit', 'plot_hard_limit_path')",
                        "apical_point_path",
                        "cut_leaves_path",
                        "hard_limit_path",
                    ],
                    clean_exception(
                        "ValueError: Can not search for cut leaves for a neuron with no neurites"
                    ): ["('CutLeaves', 'exception')"],
                    clean_exception(
                        "ValueError: Can not plot axis marker for a neuron with no neurites"
                    ): ["('PlotHardLimit', 'exception')"],
                },
            }
        },
        "ApicalPoint/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "apical_point_path",
                        "morph_path",
                    ]
                }
            }
        },
        "PlotApicalPoint/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: ["morph_path", "apical_point_path", "plot_apical_point_path"]
                }
            }
        },
        "CollectCurated/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: ["morph_path"],
                }
            }
        },
        "CutLeaves/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: ["morph_path", "cut_leaves_path"],
                    clean_exception(
                        "ValueError: Can not search for cut leaves for a neuron with no neurites."
                    ): ["exception"],
                },
            }
        },
        "PlotCutLeaves/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: ["morph_path", "cut_leaves_path", "plot_cut_leaves_path"]
                }
            }
        },
        "HardLimit/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {out_dir_pattern: ["morph_path", "hard_limit_path"]}
            }
        },
        "PlotHardLimit/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "hard_limit_path",
                        "morph_path",
                        "plot_hard_limit_path",
                    ],
                    clean_exception(
                        "ValueError: Can not plot axis marker for a neuron with no neurites."
                    ): ["exception"],
                }
            }
        },
    }

    # Check the results
    assert_equal_trees(
        annotation_expected_dir, example_1 / "out_annotated", specific_args=annotated_specific_args
    )

    # Run the Repair workflow
    assert luigi.build([workflows.Repair()], local_scheduler=True)

    # Define post-processes to check the results
    repair_expected_dir = data_dir / "test_example_1" / "out_repaired"
    repaired_specific_args = {
        "repaired_dataset.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "apical_point_path",
                        "cut_leaves_path",
                        "error_annotated_path",
                        "error_marker_path",
                        "hard_limit_path",
                        "marker_path",
                        "morph_path",
                        "repair_morph_db_path_asc",
                        "repair_morph_db_path_h5",
                        "repair_morph_db_path_swc",
                        "repair_morph_path",
                        "unravel_morph_db_path_asc",
                        "unravel_morph_db_path_h5",
                        "unravel_morph_db_path_swc",
                        "unravel_morph_path",
                        "zero_diameter_morph_db_path_asc",
                        "zero_diameter_morph_db_path_h5",
                        "zero_diameter_morph_db_path_swc",
                        "zero_diameter_morph_path",
                        "smooth_morph_path",
                    ]
                }
            }
        },
        "Repair/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "('CollectAnnotated', 'apical_point_path')",
                        "('CollectAnnotated', 'cut_leaves_path')",
                        "('CollectAnnotated', 'morph_path')",
                        "('FixZeroDiameters', 'morph_path')",
                        "('MakeRelease', 'repair_morph_db_path_asc')",
                        "('MakeRelease', 'repair_morph_db_path_h5')",
                        "('MakeRelease', 'repair_morph_db_path_swc')",
                        "('MakeRelease', 'repair_morph_path')",
                        "('MakeRelease', 'unravel_morph_db_path_asc')",
                        "('MakeRelease', 'unravel_morph_db_path_h5')",
                        "('MakeRelease', 'unravel_morph_db_path_swc')",
                        "('MakeRelease', 'unravel_morph_path')",
                        "('MakeRelease', 'zero_diameter_morph_db_path_asc')",
                        "('MakeRelease', 'zero_diameter_morph_db_path_h5')",
                        "('MakeRelease', 'zero_diameter_morph_db_path_swc')",
                        "('MakeRelease', 'zero_diameter_morph_path')",
                        "('PlotRepair', 'cut_leaves_path')",
                        "('PlotRepair', 'morph_path')",
                        "('PlotRepair', 'plot_repair_path')",
                        "('RepairNeurites', 'morph_path')",
                        "('RepairNeurites', 'unravelled_apical_point_path')",
                        "('RepairNeurites', 'unravelled_cut_leaves_path')",
                        "('Unravel', 'apical_point_path')",
                        "('Unravel', 'cut_leaves_path')",
                        "('Unravel', 'morph_path')",
                        "('Unravel', 'unravelled_apical_point_path')",
                        "('Unravel', 'unravelled_cut_leaves_path')",
                        "('SmoothDiameters', 'morph_path')",
                        "('SmoothDiameters', 'apical_point_path')",
                        "('PlotSmoothDiameters', 'morph_path')",
                        "('PlotSmoothDiameters', 'smooth_morph_path')",
                        "('PlotSmoothDiameters', 'plot_smooth_path')",
                        "('MakeCollage', 'morph_path')",
                        "apical_point_path",
                        "cut_leaves_path",
                        "morph_path",
                        "repair_morph_db_path_asc",
                        "repair_morph_db_path_h5",
                        "repair_morph_db_path_swc",
                        "repair_morph_path",
                        "unravel_morph_db_path_asc",
                        "unravel_morph_db_path_h5",
                        "unravel_morph_db_path_swc",
                        "unravel_morph_path",
                        "zero_diameter_morph_db_path_asc",
                        "zero_diameter_morph_db_path_h5",
                        "zero_diameter_morph_db_path_swc",
                        "zero_diameter_morph_path",
                        "smooth_morph_path",
                    ],
                },
            }
        },
        "CollectAnnotated/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "apical_point_path",
                        "cut_leaves_path",
                        "morph_path",
                    ],
                }
            }
        },
        "RepairNeurites/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "morph_path",
                        "unravelled_apical_point_path",
                        "unravelled_cut_leaves_path",
                    ],
                },
            }
        },
        "FixZeroDiameters/report.csv": {
            "format_data_kwargs": {"replace_pattern": {out_dir_pattern: ["morph_path"]}}
        },
        "MakeRelease/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "repair_morph_db_path_asc",
                        "repair_morph_db_path_h5",
                        "repair_morph_db_path_swc",
                        "repair_morph_path",
                        "unravel_morph_db_path_asc",
                        "unravel_morph_db_path_h5",
                        "unravel_morph_db_path_swc",
                        "unravel_morph_path",
                        "zero_diameter_morph_db_path_asc",
                        "zero_diameter_morph_db_path_h5",
                        "zero_diameter_morph_db_path_swc",
                        "zero_diameter_morph_path",
                    ]
                }
            }
        },
        "Unravel/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: [
                        "apical_point_path",
                        "cut_leaves_path",
                        "morph_path",
                        "unravelled_apical_point_path",
                        "unravelled_cut_leaves_path",
                    ]
                }
            }
        },
        "PlotRepair/report.csv": {
            "format_data_kwargs": {
                "replace_pattern": {
                    out_dir_pattern: ["morph_path", "cut_leaves_path", "plot_repair_path"]
                }
            }
        },
        "MakeCollage/report.csv": {
            "format_data_kwargs": {"replace_pattern": {out_dir_pattern: ["morph_path"]}}
        },
    }

    # Check the results
    assert_equal_trees(
        repair_expected_dir, example_1 / "out_repaired", specific_args=repaired_specific_args
    )


def test_example_generation(tmp_working_dir, examples_dir):
    """Test that the example is the same as the one generated by the Initialize workflow."""
    db_source = "NeuroMorpho"
    cli.main(["Initialize", "--source-database", db_source])

    assert (tmp_working_dir / "logging.conf").exists()
    assert (tmp_working_dir / "luigi.cfg").exists()
    assert (tmp_working_dir / "neuromorpho_config.json").exists()

    compare_files(
        tmp_working_dir / "logging.conf",
        examples_dir / "logging.conf",
        IniComparator(),
    )
    compare_files(
        tmp_working_dir / "luigi.cfg",
        examples_dir / "luigi.cfg",
        IniComparator(),
    )
    compare_files(
        tmp_working_dir / f"{db_source.lower()}_config.json",
        examples_dir / f"{db_source.lower()}_config.json",
        JsonComparator(),
    )


def test_initialize_from_dir(tmp_working_dir, examples_dir, examples_test_dir):
    """Test that the example is the same as the one generated by the Initialize workflow."""
    cli.main(["Initialize", "--input-dir", str(examples_test_dir / "morphologies")])

    assert (tmp_working_dir / "logging.conf").exists()
    assert (tmp_working_dir / "luigi.cfg").exists()

    compare_files(
        tmp_working_dir / "logging.conf",
        examples_dir / "logging.conf",
        IniComparator(),
    )
    compare_files(
        tmp_working_dir / "luigi.cfg",
        examples_dir / "luigi.cfg",
        IniComparator(),
    )

    res = pd.read_csv(tmp_working_dir / "dataset.csv")
    assert len(res) == len(
        [i for i in (examples_test_dir / "morphologies").iterdir() if i.suffix.lower() in EXTS]
    )
    assert res.columns.tolist() == ["morph_name", "morph_path"]
