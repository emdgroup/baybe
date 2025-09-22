"""Tests for partial measurements."""

import re
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.acquisition import qLogNEHVI
from baybe.exceptions import IncompleteMeasurementsError
from baybe.objectives import DesirabilityObjective, ParetoObjective
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import create_fake_input

desirability_targets = [
    NumericalTarget.normalized_ramp("t1", cutoffs=(0, 1)),
    NumericalTarget.normalized_ramp("t2", cutoffs=(0, 1), descending=True),
]


@pytest.mark.parametrize(
    "objective",
    [
        param(NumericalTarget("t1").to_objective(), id="single_target"),
        param(
            DesirabilityObjective(desirability_targets, as_pre_transformation=True),
            id="desirability",
        ),
    ],
)
def test_invalid_partial_measurements(campaign):
    """Objectives that require complete measurements raise an error when encountering
    incomplete measurements.
    """  # noqa: D205
    target_names = [t.name for t in campaign.targets]
    with pytest.raises(
        IncompleteMeasurementsError,
        match=f"missing values for the following targets: "
        f"{re.escape(str(target_names))}",
    ):
        campaign.add_measurements(
            pd.DataFrame({t.name: [float("nan")] for t in campaign.targets})
        )


@pytest.mark.parametrize(
    "objective",
    [
        param(
            DesirabilityObjective(desirability_targets),
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
def test_partial_measurements(ongoing_campaign):
    """Objectives that can handle partial measurements do not complain when
    encountering incomplete measurements.
    """  # noqa: D205
    m = ongoing_campaign.measurements
    ts = ongoing_campaign.targets

    # Mark some measurements as unmeasured / failed
    m.iloc[0, m.columns.get_loc(ts[0].name)] = np.nan
    m.iloc[-1, m.columns.get_loc(ts[-1].name)] = np.nan
    ongoing_campaign.update_measurements(m)

    # Trigger refit
    ongoing_campaign.recommend(2)


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
