"""Tests pending experiments mechanism."""

from contextlib import nullcontext

import numpy as np
import pytest
from pytest import param

from baybe.objectives import DesirabilityObjective, ParetoObjective
from baybe.targets import NumericalTarget


@pytest.mark.parametrize(
    "objective",
    [
        param(NumericalTarget("t1", "MAX").to_objective(), id="single_target"),
        param(
            DesirabilityObjective(
                [
                    NumericalTarget("t1", "MAX", (0, 1)),
                    NumericalTarget("t2", "MIN", (0, 1)),
                ]
            ),
            id="desirability",
        ),
        param(
            ParetoObjective(
                [NumericalTarget("t1", "MAX"), NumericalTarget("t2", "MIN")],
            ),
            id="pareto",
        ),
    ],
)
@pytest.mark.parametrize("n_iterations", [2], ids=["i2"])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
@pytest.mark.parametrize("n_grid_points", [5], ids=["g5"])
def test_pending_measurements(ongoing_campaign, objective):
    c = ongoing_campaign
    m = c.measurements
    ts = c.targets

    # Mark some measurements as pending and failed
    m.iloc[0, m.columns.get_loc(ts[0].name)] = np.nan
    m.iloc[-1, m.columns.get_loc(ts[-1].name)] = np.nan
    c.update_measurements(m)

    # Failure is expected when single surrogates are used, which is exactly the case
    # for non-Pareto objectives. For ParetoObjectives, the filtering is expected to
    # remove affected rows, resulting in no error.
    context = nullcontext()
    if not isinstance(objective, ParetoObjective):
        context = pytest.raises(ValueError, match="Bad input in the rows")

    # Trigger refit
    with context:
        c.recommend(2)
