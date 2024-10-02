"""Deprecation tests."""

import pandas as pd
import pytest

from baybe.acquisition.base import AcquisitionFunction
from baybe.exceptions import DeprecationError
from baybe.objective import Objective as OldObjective
from baybe.objectives.base import Objective
from baybe.objectives.desirability import DesirabilityObjective
from baybe.parameters import SubstanceEncoding
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders.pure.bayesian import (
    BotorchRecommender,
    SequentialGreedyRecommender,
)
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.targets.numerical import NumericalTarget
from baybe.utils.chemistry import convert_fingeprint_parameters


def test_deprecated_objective_class():
    """Using the deprecated objective class raises a warning."""
    with pytest.warns(DeprecationWarning):
        OldObjective(mode="SINGLE", targets=[NumericalTarget(name="a", mode="MAX")])


deprecated_objective_config = """
{
    "mode": "DESIRABILITY",
    "targets": [
        {
            "type": "NumericalTarget",
            "name": "Yield",
            "mode": "MAX",
            "bounds": [0, 1]
        },
        {
            "type": "NumericalTarget",
            "name": "Waste",
            "mode": "MIN",
            "bounds": [0, 1]
        }
    ],
    "combine_func": "MEAN",
    "weights": [1, 2]
}
"""


def test_deprecated_objective_config_deserialization():
    """The deprecated objective config format can still be parsed."""
    expected = DesirabilityObjective(
        targets=[
            NumericalTarget("Yield", "MAX", bounds=(0, 1)),
            NumericalTarget("Waste", "MIN", bounds=(0, 1)),
        ],
        scalarizer="MEAN",
        weights=[1, 2],
    )
    actual = Objective.from_json(deprecated_objective_config)
    assert expected == actual, (expected, actual)


@pytest.mark.parametrize("acqf", ("VarUCB", "qVarUCB"))
def test_deprecated_acqfs(acqf):
    """Using the deprecated acqf raises a warning."""
    with pytest.warns(DeprecationWarning):
        BotorchRecommender(acquisition_function=acqf)

    with pytest.warns(DeprecationWarning):
        AcquisitionFunction.from_dict({"type": acqf})


def test_deprecated_acqf_keyword(acqf):
    """Using the deprecated keyword raises an error."""
    with pytest.raises(DeprecationError):
        BotorchRecommender(acquisition_function_cls="qEI")


def test_deprecated_sequentialgreedyrecommender_class():
    """Using the deprecated `SequentialGreedyRecommender` class raises a warning."""
    with pytest.warns(DeprecationWarning):
        SequentialGreedyRecommender()


def test_deprecated_samples_random():
    """Using the deprecated `samples_random` method raises a warning."""
    with pytest.warns(DeprecationWarning):
        parameters = [NumericalContinuousParameter("x", (0, 1))]
        SubspaceContinuous(parameters).samples_random(n_points=1)


def test_deprecated_samples_full_factorial():
    """Using the deprecated `samples_full_factorial` method raises a warning."""
    with pytest.warns(DeprecationWarning):
        parameters = [NumericalContinuousParameter("x", (0, 1))]
        SubspaceContinuous(parameters).samples_full_factorial(n_points=1)


def test_deprecated_transform_interface(searchspace):
    """Using the deprecated transform interface raises a warning."""
    # Not providing `allow_extra` when there are additional columns
    with pytest.warns(DeprecationWarning):
        searchspace.discrete.transform(
            pd.DataFrame(columns=["additional", *searchspace.discrete.exp_rep.columns])
        )

    # Passing dataframe via `data`
    with pytest.warns(DeprecationWarning):
        searchspace.discrete.transform(
            data=searchspace.discrete.exp_rep, allow_extra=True
        )


def test_deprecated_surrogate_registration():
    """Using the deprecated registration mechanism raises a warning."""
    from baybe.surrogates import register_custom_architecture

    with pytest.raises(DeprecationError):
        register_custom_architecture()


def test_deprecated_morgan_fp(acqf):
    """Deprecated fingerprint name raises warning and uses ECFP replacement."""
    with pytest.warns(DeprecationWarning):
        # Check that ECFP is used instead of Morgan with correct pre-defined kwargs
        morgan_class, morgan_kwargs = convert_fingeprint_parameters(
            name=SubstanceEncoding("MORGAN_FP").name, kwargs_fingerprint=None
        )
        ecfp_class, _ = convert_fingeprint_parameters(
            name=SubstanceEncoding("ECFP").name, kwargs_fingerprint=None
        )
        assert morgan_class == ecfp_class
        assert morgan_kwargs == {"fp_size": 1024, "radius": 4}

        # Check that user-specified kwargs override the defaults
        _, morgan_custom_kwargs = convert_fingeprint_parameters(
            name=SubstanceEncoding("MORGAN_FP").name, kwargs_fingerprint={"radius": 5}
        )
        assert morgan_custom_kwargs == {"fp_size": 1024, "radius": 5}
