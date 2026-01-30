"""Tests for fidelity parameters."""

from baybe.parameters.fidelity import CategoricalFidelityParameter


def test_construction():
    """Equivalent zeta formats and value orderings produce equal objects."""
    p1 = CategoricalFidelityParameter("p", values=["h", "l"], costs=[1, 2], zeta=5)
    p2 = CategoricalFidelityParameter("p", values=["l", "h"], costs=[2, 1], zeta=[5, 0])
    assert p1 == p2
