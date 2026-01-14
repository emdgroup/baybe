"""Tests for using analytic acquisition functions with affine transformations."""

from contextlib import nullcontext

import pandas as pd
import pytest

from baybe.acquisition import ProbabilityOfImprovement
from baybe.exceptions import IncompatibilityError
from baybe.objectives import DesirabilityObjective
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders import BotorchRecommender
from baybe.targets import NumericalTarget

searchspace = NumericalContinuousParameter("p", (0, 1)).to_searchspace()
candidates = pd.DataFrame({"p": [0.2, 0.3]})
measurements = pd.DataFrame({"p": [0.1], "t1": [0.2], "t2": [0.3]})


@pytest.mark.parametrize(
    ("objective", "allowed"),
    [
        ((NumericalTarget("t1") * 2 + 1).to_objective(), True),
        (NumericalTarget("t1").abs().to_objective(), False),
        (
            DesirabilityObjective(
                [NumericalTarget("t1"), NumericalTarget("t2")],
                scalarizer="MEAN",
                require_normalization=False,
            ),
            True,
        ),
    ],
    ids=["single-affine", "single-nonlinear", "multi-affine"],
)
def test_analytic_acqf_with_affine_target(objective, allowed):
    """Analytic acquisition functions can be used with affine target transformations."""
    with nullcontext() if allowed else pytest.raises(IncompatibilityError):
        BotorchRecommender().acquisition_values(
            candidates,
            searchspace,
            objective,
            measurements,
            acquisition_function=ProbabilityOfImprovement(),
        )
