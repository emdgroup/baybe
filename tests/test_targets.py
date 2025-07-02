"""Target tests."""

import pytest

from baybe.exceptions import IncompatibilityError
from baybe.targets.numerical import NumericalTarget as ModernTarget
from baybe.utils.interval import Interval


def test_target_normalization():
    """Target normalization works as expected."""
    t = ModernTarget("t")
    with pytest.raises(IncompatibilityError, match="Only bounded targets"):
        t.normalize()
    assert t.clamp(-2, 4).get_image() == Interval(-2, 4)
    assert t.clamp(-2, 4).normalize().get_image() == Interval(0, 1)
