"""Util functions."""
from luigi_tools.parameter import BoolParameter


def SkippableMixin(default_value=False):
    """Create a mixin class to add a ``skip`` parameter to a :class:`luigi.task`."""

    class Mixin:
        """A mixin to add a ``skip`` parameter to a :class:`luigi.task`."""

        skip = BoolParameter(default=default_value, description=":bool: Skip the task")

    return Mixin
