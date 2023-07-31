"""Recommendation functionality."""

# This ensures that the specified modules are executed and the subclasses defined
# therein are properly registered and become visible.
from baybe.strategies import bayesian, clustering, sampling  # noqa: F401
