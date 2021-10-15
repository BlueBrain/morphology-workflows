"""Test curation functions."""
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
