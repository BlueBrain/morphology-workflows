"""Package containing the workflow tasks."""
import os

import luigi_tools.util
from pkg_resources import resource_filename

# Get configuration templates from _templates/config_templates
_template_dir = os.environ.get("MORPHOLOGY_WORKFLOWS_TEMPLATE_DIR")
if _template_dir is None:
    _template_dir = resource_filename(
        "morphology_workflows",
        "_templates/config_templates",
    )
luigi_tools.util.register_templates(directory=_template_dir, name="default.cfg")
