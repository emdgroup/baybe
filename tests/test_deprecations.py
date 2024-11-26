"""Deprecation tests."""

import warnings
from unittest.mock import patch

import pandas as pd
import pytest
from pytest import param

from baybe._optional.info import CHEM_INSTALLED
from baybe.acquisition.base import AcquisitionFunction
from baybe.constraints import (
    ContinuousLinearConstraint,
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
from baybe.constraints.base import Constraint
from baybe.exceptions import DeprecationError
from baybe.objective import Objective as OldObjective
from baybe.objectives.base import Objective
from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.enum import SubstanceEncoding
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian import (
    BotorchRecommender,
    SequentialGreedyRecommender,
)
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.searchspace.discrete import SubspaceDiscrete
from baybe.searchspace.validation import get_transform_parameters
from baybe.targets.binary import BinaryTarget
from baybe.targets.numerical import NumericalTarget


def test_objective_class():
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


def test_objective_config_deserialization():
    """The deprecated objective config format can still be parsed."""
    expected = DesirabilityObjective(
        targets=[
            NumericalTarget("Yield", "MAX", bounds=(0, 1)),
            NumericalTarget("Waste", "MIN", bounds=(0, 1)),
        ],
        scalarizer="MEAN",
        weights=[1, 2],
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        actual = Objective.from_json(deprecated_objective_config)
    assert expected == actual, (expected, actual)


@pytest.mark.parametrize("acqf", ("VarUCB", "qVarUCB"))
def test_acqfs(acqf):
    """Using the deprecated acqf raises a warning."""
    with pytest.warns(DeprecationWarning):
        BotorchRecommender(acquisition_function=acqf)

    with pytest.warns(DeprecationWarning):
        AcquisitionFunction.from_dict({"type": acqf})


def test_acqf_keyword(acqf):
    """Using the deprecated keyword raises an error."""
    with pytest.raises(DeprecationError):
        BotorchRecommender(acquisition_function_cls="qEI")


def test_sequentialgreedyrecommender_class():
    """Using the deprecated `SequentialGreedyRecommender` class raises a warning."""
    with pytest.warns(DeprecationWarning):
        SequentialGreedyRecommender()


def test_samples_random():
    """Using the deprecated `samples_random` method raises a warning."""
    with pytest.warns(DeprecationWarning):
        parameters = [NumericalContinuousParameter("x", (0, 1))]
        SubspaceContinuous(parameters).samples_random(n_points=1)


def test_samples_full_factorial():
    """Using the deprecated `samples_full_factorial` method raises a warning."""
    with pytest.warns(DeprecationWarning):
        parameters = [NumericalContinuousParameter("x", (0, 1))]
        SubspaceContinuous(parameters).samples_full_factorial(n_points=1)


def test_subspace_transform_interface(searchspace):
    """Using the deprecated transform interface raises a warning."""
    # Not providing `allow_extra` when there are additional columns
    with pytest.warns(DeprecationWarning):
        searchspace.discrete.transform(
            pd.DataFrame(columns=["additional", *searchspace.discrete.exp_rep.columns]),
        )
    with pytest.warns(DeprecationWarning):
        searchspace.continuous.transform(
            pd.DataFrame(columns=["additional", *searchspace.discrete.exp_rep.columns]),
        )

    # Passing dataframe via `data`
    with pytest.warns(DeprecationWarning):
        searchspace.discrete.transform(
            data=searchspace.discrete.exp_rep, allow_extra=True
        )
    with pytest.warns(DeprecationWarning):
        searchspace.continuous.transform(
            data=searchspace.discrete.exp_rep, allow_extra=True
        )


def test_surrogate_registration():
    """Using the deprecated registration mechanism raises a warning."""
    from baybe.surrogates import register_custom_architecture

    with pytest.raises(DeprecationError):
        register_custom_architecture()


def test_surrogate_access():
    """Public attribute access to the surrogate model raises a warning."""
    recommender = BotorchRecommender()
    with pytest.warns(DeprecationWarning):
        recommender.surrogate_model


def test_continuous_linear_eq_constraint():
    """Usage of deprecated continuous linear eq constraint raises a warning."""
    with pytest.warns(DeprecationWarning):
        ContinuousLinearEqualityConstraint(["p1", "p2"])


def test_continuous_linear_inq_constraint():
    """Usage of deprecated continuous linear ineq constraint raises a warning."""
    with pytest.warns(DeprecationWarning):
        ContinuousLinearInequalityConstraint(["p1", "p2"])


@pytest.mark.parametrize(
    ("type_", "op"),
    [
        ("ContinuousLinearEqualityConstraint", "="),
        ("ContinuousLinearInequalityConstraint", ">="),
    ],
    ids=["lin_eq", "lin_ineq"],
)
def test_constraint_config_deserialization(type_, op):
    """The deprecated constraint config format can still be parsed."""
    config = """
    {
        "type": "__replace__",
        "parameters": ["p1", "p2", "p3"],
        "coefficients": [1.0, 2.0, 3.0],
        "rhs": 2.0
    }
    """
    config = config.replace("__replace__", type_)

    expected = ContinuousLinearConstraint(
        parameters=["p1", "p2", "p3"],
        operator=op,
        coefficients=[1.0, 2.0, 3.0],
        rhs=2.0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        actual = Constraint.from_json(config)
    assert expected == actual, (expected, actual)


def test_objective_transform_interface():
    """Using the deprecated transform interface raises a warning."""
    single = SingleTargetObjective(NumericalTarget("A", "MAX"))
    desirability = DesirabilityObjective(
        [
            NumericalTarget("A", "MAX", (0, 1)),
            NumericalTarget("B", "MIN", (-1, 1)),
        ]
    )

    # Not providing `allow_extra` when there are additional columns
    with pytest.warns(DeprecationWarning):
        single.transform(
            pd.DataFrame(columns=["A", "additional"]),
        )
    with pytest.warns(DeprecationWarning):
        desirability.transform(
            pd.DataFrame(columns=["A", "B", "additional"]),
        )

    # Passing dataframe via `data`
    with pytest.warns(DeprecationWarning):
        single.transform(data=pd.DataFrame(columns=["A"]), allow_extra=True)
    with pytest.warns(DeprecationWarning):
        desirability.transform(data=pd.DataFrame(columns=["A", "B"]), allow_extra=True)


def test_deprecated_get_transform_parameters():
    """Using the deprecated utility raises a warning."""
    with pytest.warns(
        DeprecationWarning, match="'get_transform_parameters' has been deprecated"
    ):
        get_transform_parameters(pd.DataFrame(), [])


def test_target_transform_interface():
    """Using the deprecated transform interface raises a warning."""
    numerical = NumericalTarget("num", "MAX")
    binary = BinaryTarget("bin")

    # Passing dataframe via `data`
    with pytest.warns(DeprecationWarning):
        numerical.transform(data=pd.DataFrame(columns=["num"]))
    with pytest.warns(DeprecationWarning):
        binary.transform(data=pd.DataFrame(columns=["bin"]))


@pytest.mark.parametrize(
    ("deprecated", "replacement"),
    [
        param(SubstanceEncoding.MORGAN_FP, "ECFPFingerprint", id="morgan"),
        param(SubstanceEncoding.RDKIT, "RDKit2DDescriptorsFingerprint", id="rdkit"),
    ],
)
@pytest.mark.skipif(
    not CHEM_INSTALLED, reason="Optional chem dependency not installed."
)
def test_deprecated_encodings(deprecated, replacement):
    """Deprecated encoding raises a warning and uses correct replacement."""
    import skfp.fingerprints

    from baybe.utils.chemistry import smiles_to_fingerprint_features

    path = f"skfp.fingerprints.{replacement}"

    with patch(path, wraps=getattr(skfp.fingerprints, replacement)) as patched:
        # Assert warning
        with pytest.warns(DeprecationWarning):
            smiles_to_fingerprint_features(["C"], deprecated)

        # Check that equivalent is used instead of deprecated encoding
        if deprecated is SubstanceEncoding.MORGAN_FP:
            patched.assert_called_once_with(**{"fp_size": 1024, "radius": 4})
        else:
            patched.assert_called_once()


def test_migrated_metadata_attribute():
    """Accessing the migrated metadata search space attribute raises an error."""
    with pytest.raises(DeprecationError, match="no longer carry any metadata"):
        subspace = SubspaceDiscrete.from_parameter(
            NumericalDiscreteParameter("p", [0, 1])
        )
        subspace.metadata


def test_deprecated_meta_recommender_methods():
    """Calling the deprecated methods of meta recommender raises an error."""
    recommender = TwoPhaseMetaRecommender()
    with pytest.raises(DeprecationError, match="has been deprecated."):
        recommender.get_current_recommender()
    with pytest.raises(DeprecationError, match="has been deprecated"):
        recommender.get_next_recommender()


@pytest.mark.parametrize(
    "flag",
    [
        "allow_repeated_recommendations",
        "allow_recommending_already_measured",
        "allow_recommending_pending_experiments",
    ],
)
@pytest.mark.parametrize("recommender_cls", [RandomRecommender, BotorchRecommender])
def test_migrated_allow_flags(flag, recommender_cls):
    """Passing and accessing the migrated 'allow_*' flags raises an error."""
    with pytest.raises(DeprecationError, match=r"Passing 'allow_\*' flags"):
        recommender_cls(**{flag: True})

    with pytest.raises(DeprecationError, match=f"The attribute '{flag}' is no longer"):
        getattr(recommender_cls(), flag)
