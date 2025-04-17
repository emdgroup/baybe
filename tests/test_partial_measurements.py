"""Tests for partial measurements."""

from contextlib import nullcontext

import numpy as np
import pytest
from pytest import param

from baybe.acquisition import qLogNEHVI
from baybe.exceptions import IncompleteMeasurementsError
from baybe.objectives import DesirabilityObjective, ParetoObjective
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import create_fake_input


@pytest.mark.parametrize(
    "objective",
    [
        param(NumericalTarget("t1").to_objective(), id="single_target"),
        param(
            DesirabilityObjective(
                [
                    NumericalTarget.clamped_affine("t1", cutoffs=(0, 1)),
                    NumericalTarget.clamped_affine(
                        "t2", cutoffs=(0, 1), descending=True
                    ),
                ]
            ),
            id="desirability",
        ),
        param(
            ParetoObjective(
                [NumericalTarget("t1"), NumericalTarget("t2", minimize=True)],
            ),
            id="pareto",
        ),
    ],
)
@pytest.mark.parametrize("n_iterations", [2], ids=["i2"])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
@pytest.mark.parametrize("n_grid_points", [5], ids=["g5"])
def test_partial_measurements(ongoing_campaign, objective):
    c = ongoing_campaign
    m = c.measurements
    ts = c.targets

    # Mark some measurements as unmeasured / failed
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


@pytest.mark.parametrize(
    "acqf",
    [
        param(qLogNEHVI(), id="ref_point_default"),
        param(qLogNEHVI(reference_point=[0.4, 0.2]), id="ref_point_provided"),
    ],
)
@pytest.mark.parametrize(
    "objective",
    [
        param(
            ParetoObjective(
                [NumericalTarget("t1"), NumericalTarget("t2", minimize=True)],
            ),
            id="pareto",
        ),
    ],
)
@pytest.mark.parametrize("n_grid_points", [5], ids=["g5"])
def test_pareto_ref_point_from_incomplete_measurements(campaign):
    measurements = create_fake_input(
        campaign.parameters, campaign.targets, 2
    ).reset_index(drop=True)
    measurements.loc[0, "t1"] = None
    measurements.loc[1, "t2"] = None
    campaign.add_measurements(measurements)

    context = nullcontext()
    if campaign.recommender.recommender.acquisition_function.reference_point is None:
        context = pytest.raises(IncompleteMeasurementsError, match="at least one")
    with context:
        campaign.recommend(5)
