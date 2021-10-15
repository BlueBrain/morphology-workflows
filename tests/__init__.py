"""Some testing tools."""
import re
from tempfile import NamedTemporaryFile

from neurom.core import Morphology


def remove_between(start_part, end_part, tags=None):
    """Prepare a regex to replace a part of a string by three dots."""
    if tags:
        return (f"{start_part}.*{end_part}", f"{start_part} ... {end_part}", tags)
    else:
        return (f"{start_part}.*{end_part}", f"{start_part} ... {end_part}")


def clean_exception(exc):
    """Prepare a regex to clean an exception."""
    return remove_between("Traceback ", exc, re.DOTALL)


def create_morphology(content, extension):
    """Create a morphology from a string."""
    with NamedTemporaryFile(suffix=f".{extension}", mode="w", encoding="utf-8") as tmp_file:
        tmp_file.write(content)
        tmp_file.flush()

        return Morphology(tmp_file.name)
