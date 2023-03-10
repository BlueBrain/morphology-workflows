"""Package containing the workflow tasks."""
import os

import luigi_tools.util

from morphology_workflows.utils import _TEMPLATES

# Get configuration templates from _templates/config_templates
_template_dir = os.environ.get("MORPHOLOGY_WORKFLOWS_TEMPLATE_DIR")
if _template_dir is None:
    _template_dir = str(_TEMPLATES / "config_templates")
luigi_tools.util.register_templates(directory=_template_dir, name="default.cfg")
