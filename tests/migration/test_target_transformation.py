import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from pytest import param

from baybe.targets._deprecated import NumericalTarget as LegacyTarget
from baybe.targets.numerical import NumericalTarget as ModernTarget
from baybe.targets.transforms import (
    AbsoluteTransformation,
    AffineTransformation,
    BellTransformation,
    ChainedTransformation,
    ClampingTransformation,
)


@pytest.fixture
def series() -> pd.Series:
    return pd.Series(np.linspace(-10, 10, 20))


@pytest.mark.parametrize(
    ("legacy", "modern"),
    [
        param(
            LegacyTarget("t", "MAX"),
            ModernTarget("t"),
            id="max",
        ),
        param(
            LegacyTarget("t", "MAX", (0, 1), "LINEAR"),
            ModernTarget("t", ClampingTransformation(min=0, max=1)),
            id="max_clamped",
        ),
        param(
            LegacyTarget("t", "MAX", (2, 5), "LINEAR"),
            ModernTarget(
                "t",
                ChainedTransformation(
                    AffineTransformation.from_unit_interval(2, 5),
                    ClampingTransformation(min=0, max=1),
                ),
            ),
            id="max_shifted_clamped",
        ),
        param(
            None,
            ModernTarget("t", AffineTransformation(factor=-1)),
            marks=pytest.mark.xfail(
                reason="Minimization transformation without bounds "
                "is not possible with legacy interface."
            ),
            id="min_no_bounds",
        ),
        param(
            # NOTE: Minimization without bounds has no effect on the transformation
            #   of the legacy target since minimization is handled in the construction
            #   of the acquisition function.
            LegacyTarget("t", "MIN"),
            ModernTarget("t"),
            id="min",
        ),
        param(
            LegacyTarget("t", "MIN", (0, 1), "LINEAR"),
            ModernTarget(
                "t",
                ChainedTransformation(
                    AffineTransformation(factor=-1, shift=1),
                    ClampingTransformation(min=0, max=1),
                ),
            ),
            id="min_clamped",
        ),
        param(
            LegacyTarget("t", "MIN", (2, 5), "LINEAR"),
            ModernTarget(
                "t",
                ChainedTransformation(
                    AffineTransformation.from_unit_interval(5, 2),
                    ClampingTransformation(min=0, max=1),
                ),
            ),
            id="min_shifted_clamped",
        ),
        param(
            LegacyTarget("t", "MATCH", (-1, 1), "BELL"),
            ModernTarget("t", BellTransformation()),
            id="match_bell_unit_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (1, 3), "BELL"),
            ModernTarget("t", BellTransformation(center=2)),
            id="match_bell_unit_shifted",
        ),
        param(
            LegacyTarget("t", "MATCH", (-5, 5), "BELL"),
            ModernTarget("t", BellTransformation(width=5)),
            id="match_bell_scaled_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (2, 6), "BELL"),
            ModernTarget("t", BellTransformation(center=4, width=2)),
            id="match_bell_scaled_shifted",
        ),
        param(
            LegacyTarget("t", "MATCH", (-1, 1), "TRIANGULAR"),
            ModernTarget(
                "t",
                ChainedTransformation(
                    AbsoluteTransformation(),
                    AffineTransformation(factor=-1, shift=1),
                    ClampingTransformation(min=0, max=1),
                ),
            ),
            id="match_triangular_unit_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (1, 3), "TRIANGULAR"),
            ModernTarget(
                "t",
                ChainedTransformation(
                    AffineTransformation.from_unit_interval(2, 3),
                    AbsoluteTransformation(),
                    AffineTransformation(factor=-1, shift=1),
                    ClampingTransformation(min=0, max=1),
                ),
            ),
            id="match_triangular_unit_shifted",
        ),
        param(
            LegacyTarget("t", "MATCH", (-5, 5), "TRIANGULAR"),
            ModernTarget(
                "t",
                ChainedTransformation(
                    AbsoluteTransformation(),
                    AffineTransformation(factor=-1 / 5, shift=1),
                    ClampingTransformation(min=0, max=1),
                ),
            ),
            id="match_triangular_scaled_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (2, 6), "TRIANGULAR"),
            ModernTarget(
                "t",
                ChainedTransformation(
                    AffineTransformation.from_unit_interval(4, 6),
                    AbsoluteTransformation(),
                    AffineTransformation(factor=-1, shift=1),
                    ClampingTransformation(min=0, max=1),
                ),
            ),
            id="match_triangular_scaled_shifted",
        ),
    ],
)
def test_target_transformation(series, legacy: LegacyTarget, modern: ModernTarget):
    assert_series_equal(legacy.transform(series), modern.transform(series))


def test_transformation_chaining():
    t1 = AffineTransformation()
    t2 = ClampingTransformation()
    t3 = AbsoluteTransformation()

    c = ChainedTransformation(t1, t2)
    t = c.append(t3).append(c)

    assert t == ChainedTransformation(t1, t2, t3, t1, t2)
