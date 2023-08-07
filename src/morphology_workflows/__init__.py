"""Workflow for morphology processing."""
import importlib.metadata

__version__ = importlib.metadata.version("morphology-workflows")


class MorphologyWorkflowsError(Exception):
    """Base exception for morphology-workflows errors."""
