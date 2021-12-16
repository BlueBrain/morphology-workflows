"""Util functions."""
from functools import partial

from data_validation_framework.result import ValidationResult
from luigi_tools.parameter import BoolParameter

from morphology_workflows.utils import SKIP_COMMENT


def _skippable_validation_function(validation_function, skip, *args, **kwargs):
    if skip:
        return ValidationResult(is_valid=True, comment=SKIP_COMMENT)
    return validation_function(*args, **kwargs)


def SkippableMixin(default_value=False):
    """Create a mixin class to add a ``skip`` parameter.

    This mixin must be applied to a :class:`data_validation_framework.ElementValidationTask`.
    It will create a ``skip`` parameter and wrap the validation function to just skip it if the
    ``skip`` argument is set to ``True``.

    .. todo::
        * Move this class into the ``data-validation-framework`` package?
        * Make it also work for :class:`data_validation_framework.SetValidationTask`?
        * Change it into a class decorator?

    Args:
        default_value (bool): The default value for the ``skip`` argument.
    """

    class Mixin:
        """A mixin to add a ``skip`` parameter to a :class:`luigi.task`."""

        skip = BoolParameter(default=default_value, description=":bool: Skip the task")

        def __init__(self, *args, **kwargs):

            super().__init__(*args, **kwargs)

            self._skippable_validation_function = self.validation_function
            self.validation_function = partial(
                _skippable_validation_function,
                self._skippable_validation_function,
                self.skip,
            )

    return Mixin
