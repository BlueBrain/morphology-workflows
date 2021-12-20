"""Util functions."""
from functools import partial

from data_validation_framework.result import ValidationResult
from data_validation_framework.task import ElementValidationTask
from data_validation_framework.task import SetValidationTask
from data_validation_framework.task import ValidationWorkflow
from luigi_tools.parameter import BoolParameter

from morphology_workflows.utils import SKIP_COMMENT


def _skippable_element_validation_function(validation_function, skip, *args, **kwargs):
    if skip:
        return ValidationResult(is_valid=True, comment=SKIP_COMMENT)
    return validation_function(*args, **kwargs)


def _skippable_set_validation_function(validation_function, skip, df, *args, **kwargs):
    if skip:
        df.loc[df["is_valid"], "comment"] = SKIP_COMMENT
    else:
        validation_function(df, *args, **kwargs)


def SkippableMixin(default_value=False):
    """Create a mixin class to add a ``skip`` parameter.

    This mixin must be applied to a :class:`data_validation_framework.ElementValidationTask`.
    It will create a ``skip`` parameter and wrap the validation function to just skip it if the
    ``skip`` argument is set to ``True``. If skipped, it will keep the ``is_valid`` values as is and
    add a specific comment to inform the user.

    .. todo::
        * Move this class into the ``data-validation-framework`` package?

    Args:
        default_value (bool): The default value for the ``skip`` argument.
    """

    class Mixin:
        """A mixin to add a ``skip`` parameter to a :class:`luigi.task`."""

        skip = BoolParameter(default=default_value, description=":bool: Skip the task")

        def __init__(self, *args, **kwargs):

            super().__init__(*args, **kwargs)

            if isinstance(self, ElementValidationTask):
                new_validation_function = partial(
                    _skippable_element_validation_function,
                    self.validation_function,
                    self.skip,
                )
            elif isinstance(self, SetValidationTask) and not isinstance(self, ValidationWorkflow):
                new_validation_function = partial(
                    _skippable_set_validation_function,
                    self.validation_function,
                    self.skip,
                )
            else:
                raise TypeError(
                    "The SkippableMixin can only be associated with childs of ElementValidationTask"
                    " or SetValidationTask"
                )
            self._skippable_validation_function = self.validation_function
            self.validation_function = new_validation_function

    return Mixin
