"""Test curation functions."""
from numpy.testing import assert_array_almost_equal

from morphology_workflows import curation

from . import create_morphology


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
