"""Test curation functions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from morphio import PointLevel
from morphio import SectionType
from morphio import SomaType
from morphio.mut import Morphology
from numpy.testing import assert_array_almost_equal

from morphology_workflows import curation

from . import create_morphology
from . import create_morphology_file


class Test_fix_soma_radius:
    """Test the function curation.fix_soma_radius()."""

    def test_nothing_to_fix(self):
        """No point in soma."""
        morph = create_morphology(
            """
            1 1 0 0 0 1. -1
            2 2 1 0 0 1. 1
            3 2 2 0 0 1. 2
            4 2 3 0 0 1. 3
            """,
            "swc",
        )

        former_radius, new_radius = curation.fix_soma_radius(morph)
        assert former_radius == 1
        assert new_radius is None

    def test_first_point_in_soma(self):
        """First point in soma."""
        morph = create_morphology(
            """
            1 1 0 0 0 1.5 -1
            2 2 1 0 0 1. 1
            3 2 2 0 0 1. 2
            4 2 3 0 0 1. 3
            """,
            "swc",
        )

        former_radius, new_radius = curation.fix_soma_radius(morph)
        assert former_radius == 1.5
        assert new_radius is None

    def test_last_point_in_soma(self):
        """All points in soma."""
        morph = create_morphology(
            """
            1 1 0 0 0 10 -1
            2 2 1 0 0 1. 1
            3 2 2 0 0 1. 2
            4 2 3 0 0 1. 3
            """,
            "swc",
        )

        former_radius, new_radius = curation.fix_soma_radius(morph)
        assert former_radius == 10
        assert new_radius == 2.5

    def test_last_point_closer_than_before_last(self):
        """All points in soma and the last points is closer than the before last point."""
        morph = create_morphology(
            """
            1 1 0 0 0 10 -1
            2 2 1 0 0 1. 1
            3 2 3 0 0 1. 2
            4 2 2 0 0 1. 3
            """,
            "swc",
        )

        former_radius, new_radius = curation.fix_soma_radius(morph)
        assert former_radius == 10
        assert new_radius == 2.5

    def test_two_points(self):
        """Only 2 points and both in soma."""
        morph = create_morphology(
            """
            1 1 0 0 0 10 -1
            2 2 2 0 0 1. 1
            3 2 3 0 0 1. 2
            """,
            "swc",
        )

        former_radius, new_radius = curation.fix_soma_radius(morph)
        assert former_radius == 10
        assert new_radius == 2.5

    def test_two_points_with_last_closer(self):
        """Only 2 points and the last points is closer than the before last point."""
        morph = create_morphology(
            """
            1 1 0 0 0 10 -1
            2 2 3 0 0 1. 1
            3 2 2 0 0 1. 2
            """,
            "swc",
        )

        former_radius, new_radius = curation.fix_soma_radius(morph)
        assert former_radius == 10
        assert new_radius == 2.5


class TestFixRootSections:
    """Test the function curation.fix_root_sections()."""

    def test_nothing_to_fix(self):
        """The first section is long enough."""
        morph = create_morphology(
            """
            1 1 0 0 0 1. -1
            2 2 1 0 0 1. 1
            3 2 1.5 0 0 1. 2
            4 2 2 0 0 1. 3
            5 2 3 0 0 1. 3
            """,
            "swc",
        )

        curation.fix_root_section(morph)
        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.5, 0.0, 0.0, 1.0],
                [1.5, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 1.0],
                [1.5, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
            ],
        )

    def test_small_section(self):
        """The first section is too small."""
        morph = create_morphology(
            """
            1 1 0 0 0 1. -1
            2 2 1 0 0 1. 1
            3 2 1.05 0 0 1. 2
            4 2 2 0 0 1. 3
            5 2 3 0 0 1. 3
            """,
            "swc",
        )

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.05, 0.0, 0.0, 1.0],
                [1.05, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 1.0],
                [1.05, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
            ],
        )

        curation.fix_root_section(morph, min_length=0.5)

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.5, 0.0, 0.0, 1.0],
                [1.5, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 1.0],
                [1.5, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
            ],
        )

    def test_small_section_and_small_child(self):
        """The first section is too small and one of its children is also too small."""
        morph = create_morphology(
            """
            1 1 0 0 0 1. -1
            2 2 1 0 0 1. 1
            3 2 1.05 0 0 1. 2
            4 2 1.06 0 0 1. 3
            5 2 3 0 0 1. 4
            6 2 3 1 0 1. 3
            """,
            "swc",
        )

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.05, 0.0, 0.0, 1.0],
                [1.05, 0.0, 0.0, 1.0],
                [1.06, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [1.05, 0.0, 0.0, 1.0],
                [3.0, 1.0, 0.0, 1.0],
            ],
        )

        curation.fix_root_section(morph, min_length=0.5)

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.486032, 0.117358, 0.0, 1.0],
                [1.486032, 0.117358, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [1.486032, 0.117358, 0.0, 1.0],
                [3.0, 1.0, 0.0, 1.0],
            ],
        )

    def test_two_small_sections(self):
        """The two first sections are too small."""
        morph = create_morphology(
            """
            1 1 0 0 0 1. -1
            2 2 1 0 0 1. 1
            3 2 1.05 0 0 1. 2
            4 2 1.1 0 0 1. 3
            5 2 3 0.5 0 1. 4
            6 2 3 0 0 1. 4
            7 2 3 1 0 1. 3
            """,
            "swc",
        )

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.05, 0.0, 0.0, 1.0],
                [1.05, 0.0, 0.0, 1.0],
                [1.1, 0.0, 0.0, 1.0],
                [1.1, 0.0, 0.0, 1.0],
                [3.0, 0.5, 0.0, 1.0],
                [1.1, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [1.05, 0.0, 0.0, 1.0],
                [3.0, 1.0, 0.0, 1.0],
            ],
        )

        curation.fix_root_section(morph, min_length=0.5)

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.4777161, 0.1476055, 0.0, 1.0],
                [1.4777161, 0.1476055, 0.0, 1.0],
                [1.5277162, 0.1476055, 0.0, 1.0],
                [1.5277162, 0.1476055, 0.0, 1.0],
                [3.0, 0.5, 0.0, 1.0],
                [1.5277162, 0.1476055, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [1.4777161, 0.1476055, 0.0, 1.0],
                [3.0, 1.0, 0.0, 1.0],
            ],
        )

    def test_three_small_sections(self):
        """The three first sections are too small."""
        morph = create_morphology(
            """
            1 1 0 0 0 1. -1
            2 2 1 0 0 1. 1
            3 2 1.05 0 0 1. 2
            4 2 1.1 0 0 1. 3
            5 2 1.15 0 0 1. 4
            6 2 3 0.5 0 1. 5
            7 2 3 0 0 1. 5
            8 2 3 0.75 0 1. 4
            9 2 3 1 0 1. 3
            """,
            "swc",
        )

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.05, 0.0, 0.0, 1.0],
                [1.05, 0.0, 0.0, 1.0],
                [1.1, 0.0, 0.0, 1.0],
                [1.1, 0.0, 0.0, 1.0],
                [1.15, 0.0, 0.0, 1.0],
                [1.15, 0.0, 0.0, 1.0],
                [3.0, 0.5, 0.0, 1.0],
                [1.15, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [1.1, 0.0, 0.0, 1.0],
                [3.0, 0.75, 0.0, 1.0],
                [1.05, 0.0, 0.0, 1.0],
                [3.0, 1.0, 0.0, 1.0],
            ],
        )

        curation.fix_root_section(morph, min_length=0.5)

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.4679317, 0.17618132, 0.0, 1.0],
                [1.4679317, 0.17618132, 0.0, 1.0],
                [1.5179318, 0.17618132, 0.0, 1.0],
                [1.5179318, 0.17618132, 0.0, 1.0],
                [1.5679318, 0.17618132, 0.0, 1.0],
                [1.5679318, 0.17618132, 0.0, 1.0],
                [3.0, 0.5, 0.0, 1.0],
                [1.5679318, 0.17618132, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [1.5179318, 0.17618132, 0.0, 1.0],
                [3.0, 0.75, 0.0, 1.0],
                [1.4679317, 0.17618132, 0.0, 1.0],
                [3.0, 1.0, 0.0, 1.0],
            ],
        )

    def test_three_zero_sections(self):
        """The three first sections have 0 lengths."""
        morph = create_morphology(
            """
            1 1 0 0 0 1. -1
            2 2 1. 0 0 1. 1
            3 2 1. 0 0 1. 2
            4 2 1. 0 0 1. 3
            5 2 1. 0 0 1. 4
            6 2 3 0.5 0 1. 5
            7 2 3 0 0 1. 5
            8 2 3 0.75 0 1. 4
            9 2 3 1 0 1. 3
            """,
            "swc",
        )

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [3.0, 0.5, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [3.0, 0.75, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [3.0, 1.0, 0.0, 1.0],
            ],
        )

        curation.fix_root_section(morph)

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.0000938, 3.4490615e-05, 0.0, 1.0],
                [1.0000938, 3.4490615e-05, 0.0, 1.0],
                [1.0000938, 3.4490615e-05, 0.0, 1.0],
                [1.0000938, 3.4490615e-05, 0.0, 1.0],
                [3.0, 5.0e-01, 0.0, 1.0],
                [1.0000938, 3.4490615e-05, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [1.0000938, 3.4490615e-05, 0.0, 1.0],
                [3.0, 7.5e-01, 0.0, 1.0],
                [1.0000938, 3.4490615e-05, 0.0, 1.0],
                [3.0, 1.0, 0.0, 1.0],
            ],
        )

    def test_zero_sections_no_child(self):
        """The first sections has 0 length and no child."""
        morph = create_morphology(
            """
            1 1 0 0 0 1. -1
            2 2 1. 0 0 1. 1
            3 2 1. 0 0 1. 2
            """,
            "swc",
        )

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
        )

        curation.fix_root_section(morph)

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.0001, 0.0, 0.0, 1.0],
            ],
        )

    def test_zero_sections_no_child_and_overlap_soma(self):
        """The first sections has 0 length and no child."""
        morph = create_morphology(
            """
            1 1 1. 0 0 1. -1
            2 2 1. 0 0 1. 1
            3 2 1. 0 0 1. 2
            4 2 2. 0 0 1. 1
            5 2 3. 0 0 1. 4
            """,
            "swc",
        )

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
            ],
        )

        curation.fix_root_section(morph)

        assert_array_almost_equal(
            morph.points,
            [
                [2.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
            ],
        )

    def test_too_small_min_length(self, caplog):
        """The min length is too small compared to the precision to move a point."""
        morph = create_morphology(
            """
            1 1 0 0 0 1. -1
            2 2 1 0 0 1. 1
            3 2 1 0 0 1. 2
            4 2 2 0 0 1. 3
            5 2 3 0 0 1. 3
            """,
            "swc",
        )

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
            ],
        )

        caplog.clear()
        with caplog.at_level("DEBUG"):
            curation.fix_root_section(morph, min_length=1e-9)

        assert_array_almost_equal(
            morph.points,
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.0000001, 0.0, 0.0, 1.0],
                [1.0000001, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 1.0],
                [1.0000001, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
            ],
        )

        assert caplog.messages == [
            (
                "The min_length was too small to move the point [1.0, 0.0, 0.0] so "
                "it was moved to [1.0000001, 0.0, 0.0]"
            )
        ]


class TestCheckNeurites:
    """Test the function curation.check_neurites()."""

    @pytest.fixture()
    def res_path(self, tmpdir):
        """Result to which the result morphologies are exported."""
        path = tmpdir / "res"
        path.mkdir()
        return path

    @pytest.fixture()
    def simple_morph(self, tmpdir):
        """A simple morphology used for testing."""
        return create_morphology_file(
            """
            1 1 0 0 0 1 -1
            5 3 -1 0 0 1. 1
            6 3 -2 0 0 1. 5
            7 3 -3 0 0 1. 6
            8 4 0 1 0 1. 1
            9 4 0 6 0 1. 8
            10 4 1 7 0 1. 9
            """,
            "swc",
            tmpdir / "morph.swc",
        )

    def test_default(self, simple_morph, res_path):
        """Check neurites with default options."""
        row = pd.Series({"morph_path": simple_morph}, name="test_name")
        res = curation.check_neurites(
            row,
            res_path,
        )

        res_morph = Morphology(res["morph_path"])
        expected = {
            0: [[-1, 0, 0], [-2, 0, 0], [-3, 0, 0]],
            1: [[0, 1, 0], [0, 6, 0], [1, 7, 0]],
            2: [[0, 0, 0], [0, -100, 0]],
        }
        for i, j in zip(res_morph.sections.items(), expected.items()):
            assert i[0] == j[0]
            assert_array_almost_equal(i[1].points, j[1])

    def test_no_soma(self, simple_morph, res_path):
        """Check neurites with default options on a morph without any soma."""
        morph_swc = str(Path(simple_morph).with_suffix(".swc"))
        morph_asc = str(Path(simple_morph).with_suffix(".asc"))

        morph = Morphology(simple_morph)

        morph.soma.type = SomaType.SOMA_UNDEFINED
        morph.soma.points = np.array([], dtype=morph.soma.points.dtype).reshape((0, 3))
        morph.soma.diameters = np.array([], dtype=morph.soma.diameters.dtype)
        morph.write(morph_swc)
        morph.write(morph_asc)

        # Test with spherical soma (the default value)
        row = pd.Series({"morph_path": morph_swc}, name="test_name")
        res = curation.check_neurites(
            row,
            res_path,
        )

        res_morph = Morphology(res["morph_path"])
        expected = {
            0: [[-1, 0, 0], [-2, 0, 0], [-3, 0, 0]],
            1: [[0, 1, 0], [0, 6, 0], [1, 7, 0]],
            2: [[0, 0, 0], [0, -100, 0]],
        }
        for i, j in zip(res_morph.sections.items(), expected.items()):
            assert i[0] == j[0]
            assert_array_almost_equal(i[1].points, j[1])
        assert_array_almost_equal(res_morph.soma.points, [[-0.5, 0.5, 0]])

        # Test with contour soma
        row = pd.Series({"morph_path": morph_asc}, name="test_name")
        res = curation.check_neurites(
            row,
            res_path,
            mock_soma_type="contour",
        )

        res_morph = Morphology(res["morph_path"])
        expected = {
            0: [[-1, 0, 0], [-2, 0, 0], [-3, 0, 0]],
            1: [[0, 1, 0], [0, 6, 0], [1, 7, 0]],
            2: [[0, 0, 0], [0, -100, 0]],
        }
        for i, j in zip(res_morph.sections.items(), expected.items()):
            assert i[0] == j[0]
            assert_array_almost_equal(i[1].points, j[1])
        assert_array_almost_equal(
            res_morph.soma.points,
            [
                [-1, 0, 0],
                [0.20710672, -0.20710684, 0],
                [0, 1, 0],
                [-1.2071068, 1.2071067, 0],
            ],
        )

    @pytest.mark.parametrize("nb_root_points", list(range(6)))
    @pytest.mark.parametrize("soma_type", ["contour", "spherical"])
    def test__add_soma(self, nb_root_points, soma_type):
        """Test _add_soma for contour type with multiple numbers of root sections."""
        # pylint: disable=protected-access
        morph = Morphology()
        morph.soma.type = SomaType.SOMA_UNDEFINED
        morph.soma.points = np.array([], dtype=morph.soma.points.dtype).reshape((0, 3))
        morph.soma.diameters = np.array([], dtype=morph.soma.diameters.dtype)

        interval = np.pi / (nb_root_points + 1)
        for i in range(nb_root_points):
            angle = i * interval
            x = np.cos(angle)
            y = np.sin(angle)
            z = 0
            stub = PointLevel([np.array([x, y, z]), 2 * np.array([x, y, z])], [1, 1])
            morph.append_root_section(stub, SectionType.axon)

        if soma_type == "contour":
            if nb_root_points < 2:
                with pytest.raises(ValueError, match="At least 2 root points are needed"):
                    curation._add_soma(morph, soma_type="contour")  # noqa: SLF001
            else:
                curation._add_soma(morph, soma_type="contour")  # noqa: SLF001

                assert len(morph.soma.points) == max(4, nb_root_points)
        else:
            curation._add_soma(morph, soma_type="spherical")  # noqa: SLF001

            assert len(morph.soma.points) == 1

    def test_no_mock_but_stub(self, simple_morph, res_path):
        """Check neurites with no mock soma but with stub axon."""
        row = pd.Series({"morph_path": simple_morph}, name="test_name")
        res = curation.check_neurites(
            row,
            res_path,
            mock_soma_type=None,
            ensure_stub_axon=True,
        )

        res_morph = Morphology(res["morph_path"])
        expected = {
            0: [[-1, 0, 0], [-2, 0, 0], [-3, 0, 0]],
            1: [[0, 1, 0], [0, 6, 0], [1, 7, 0]],
            2: [[0, 0, 0], [0, -100, 0]],
        }
        for i, j in zip(res_morph.sections.items(), expected.items()):
            assert i[0] == j[0]
            assert_array_almost_equal(i[1].points, j[1])


class TestFixNeuritesInSoma:
    """Test the function curation.fix_neurites_in_soma()."""

    @pytest.fixture()
    def res_path(self, tmpdir):
        """Result to which the result morphologies are exported."""
        path = tmpdir / "res"
        path.mkdir()
        return path

    @pytest.fixture()
    def simple_morph(self, tmpdir):
        """A simple morphology used for testing."""
        return create_morphology_file(
            """
            1 1 0 0 0 10 -1
            2 2 1 0 0 1. 1
            3 2 2 0 0 1. 2
            4 2 6 0 0 1. 3
            5 3 -1 0 0 1. 1
            6 3 -2 0 0 1. 5
            7 3 -3 0 0 1. 6
            8 4 0 1 0 1. 1
            9 4 0 6 0 1. 8
            10 4 1 7 0 1. 9
            """,
            "swc",
            tmpdir / "morph.swc",
        )

    def test_default(self, simple_morph, res_path):
        """Align with default options."""
        row = pd.Series({"morph_path": simple_morph}, name="test_name")
        res = curation.fix_neurites_in_soma(
            row,
            res_path,
        )

        res_morph = Morphology(res["morph_path"])
        expected = {
            0: [
                [2.5, 0, 0],
                [6, 0, 0],
            ],
            1: [
                [-2.5, 0, 0],
                [-3, 0, 0],
            ],
            2: [
                [0, 2.5, 0],
                [0, 6, 0],
                [1, 7, 0],
            ],
        }
        for i, j in zip(res_morph.sections.items(), expected.items()):
            assert i[0] == j[0]
            assert_array_almost_equal(i[1].points, j[1])


class TestAlign:
    """Test the function curation.align()."""

    @pytest.fixture()
    def res_path(self, tmpdir):
        """Result to which the result morphologies are exported."""
        path = tmpdir / "res"
        path.mkdir()
        return path

    @pytest.fixture()
    def simple_morph(self, tmpdir):
        """A simple morphology used for testing."""
        return create_morphology_file(
            """
            1 1 0 0 0 1. -1
            2 2 1 0 0 1. 1
            3 2 2 0 0 1. 2
            4 2 3 0 0 1. 3
            5 3 0 1 0 1. 1
            6 3 0 2 0 1. 5
            7 3 0 3 0 1. 6
            8 4 -1 0 0 1. 1
            9 4 -2 0 0 1. 8
            10 4 -3 0 0 1. 9
            """,
            "swc",
            tmpdir / "morph.swc",
        )

    def check_res(self, res, expected):
        """Compare the result points to the expected ones."""
        res_morph = Morphology(res["morph_path"])
        for i, j in zip(res_morph.sections.items(), expected.items()):
            assert i[0] == j[0]
            assert_array_almost_equal(i[1].points, j[1])

    def test_default(self, simple_morph, res_path):
        """Align with default options."""
        row = pd.Series({"morph_path": simple_morph}, name="test_name")
        res = curation.align(
            row,
            res_path,
        )

        expected = {
            0: [[0, -1, 0], [0, -2, 0], [0, -3, 0]],
            1: [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
            2: [[0, 1, 0], [0, 2, 0], [0, 3, 0]],
        }
        self.check_res(res, expected)

    def test_neurite_type(self, simple_morph, res_path):
        """Align with a given neurite type."""
        row = pd.Series({"morph_path": simple_morph}, name="test_name")
        res = curation.align(
            row,
            res_path,
            neurite_type="axon",
        )

        expected = {
            0: [[0, 1, 0], [0, 2, 0], [0, 3, 0]],
            1: [[-1, 0, 0], [-2, 0, 0], [-3, 0, 0]],
            2: [[0, -1, 0], [0, -2, 0], [0, -3, 0]],
        }
        self.check_res(res, expected)

    def test_given_direction(self, simple_morph, res_path):
        """Align with a given direction."""
        row = pd.Series({"morph_path": simple_morph}, name="test_name")
        res = curation.align(
            row,
            res_path,
            direction=[1, 0, 0],
        )

        expected = {
            0: [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
            1: [[0, 1, 0], [0, 2, 0], [0, 3, 0]],
            2: [[-1, 0, 0], [-2, 0, 0], [-3, 0, 0]],
        }
        self.check_res(res, expected)

    def test_custom_method(self, tmpdir, simple_morph, res_path):
        """Align with a custom rotation method."""
        custom_orientation_json_path = tmpdir / "custom_orientation.json"
        with open(custom_orientation_json_path, mode="w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_name": [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                },
                f,
            )
        row = pd.Series({"morph_path": simple_morph}, name="test_name")
        res = curation.align(
            row,
            res_path,
            method="custom",
            custom_orientation_json_path=custom_orientation_json_path,
        )

        expected = {
            0: [
                [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
                [np.sqrt(2), -np.sqrt(2), 0],
                [3 * np.sqrt(2) / 2, -3 * np.sqrt(2) / 2, 0],
            ],
            1: [
                [np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                [np.sqrt(2), np.sqrt(2), 0],
                [3 * np.sqrt(2) / 2, 3 * np.sqrt(2) / 2, 0],
            ],
            2: [
                [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                [-np.sqrt(2), np.sqrt(2), 0],
                [-3 * np.sqrt(2) / 2, 3 * np.sqrt(2) / 2, 0],
            ],
        }
        self.check_res(res, expected)

        # Test with unknown morphology name => should not be rotated
        row = pd.Series({"morph_path": simple_morph}, name="unknown_name")
        res = curation.align(
            row,
            res_path,
            method="custom",
            custom_orientation_json_path=custom_orientation_json_path,
        )

        expected = {i: j.points.tolist() for i, j in Morphology(simple_morph).sections.items()}
        self.check_res(res, expected)

        # Test with custom method but no custom_orientation_json_path
        with pytest.raises(
            ValueError,
            match="Provide a custom_orientation_json_path parameter when method=='custom'",
        ):
            res = curation.align(
                row,
                res_path,
                method="custom",
            )
