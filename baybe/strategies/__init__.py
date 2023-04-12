"""Recommendation functionality."""

# This ensures that the specified modules are executed and the classes defined therein
# are properly registered with their base classes via the __init_subclass__ hook.
from . import bayesian, clustering, sampling  # noqa: F401
