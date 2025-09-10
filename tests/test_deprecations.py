"""Deprecation tests."""

import warnings
from itertools import pairwise
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from pytest import param

from baybe._optional.info import CHEM_INSTALLED
from baybe.constraints import (
    ContinuousLinearConstraint,
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
from baybe.constraints.base import Constraint
from baybe.exceptions import DeprecationError
from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.enum import SubstanceEncoding
from baybe.parameters.numerical import (
    NumericalDiscreteParameter,
)
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian import (
    BotorchRecommender,
)
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace.discrete import SubspaceDiscrete
from baybe.searchspace.validation import get_transform_parameters
from baybe.targets import NumericalTarget
from baybe.targets import NumericalTarget as ModernTarget
from baybe.targets._deprecated import (
    LegacyTarget,
    bell_transform,
    linear_transform,
    triangular_transform,
)
from baybe.targets.binary import BinaryTarget
from baybe.transformations.basic import AffineTransformation


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
    single = SingleTargetObjective(NumericalTarget("A"))
    desirability = DesirabilityObjective(
        [
            NumericalTarget.normalized_ramp("A", cutoffs=(0, 1)),
            NumericalTarget.normalized_ramp("B", cutoffs=(-1, 1), descending=True),
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
    numerical = NumericalTarget("num")
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


def test_legacy_target_construction():
    """Constructing a target using legacy arguments raises a warning."""
    with pytest.warns(
        DeprecationWarning,
        match="Creating numerical targets by specifying MAX/MIN/MATCH modes",
    ):
        NumericalTarget("t", "MAX")


def test_target_deprecation_helpers():
    """Calling the target deprecation helper constructors raises a warning."""
    with pytest.warns(
        DeprecationWarning, match="The helper constructor 'from_legacy_interface'"
    ):
        NumericalTarget.from_legacy_interface("t", "MIN")
    with pytest.warns(
        DeprecationWarning, match="The helper constructor 'from_modern_interface'"
    ):
        NumericalTarget.from_modern_interface("t", minimize=True)


def test_target_legacy_deserialization():
    """Deserialization also works from legacy arguments."""
    actual = NumericalTarget.from_dict({"name": "t", "mode": "MATCH", "bounds": (1, 2)})
    expected = NumericalTarget("t", "MATCH", (1, 2))
    assert actual == expected


def sample_input() -> pd.Series:
    return pd.Series(np.linspace(-10, 10, 20))


@pytest.fixture
def series() -> pd.Series:
    return sample_input()


@pytest.mark.parametrize("mode", ["MAX", "MIN"])
def test_constructor_equivalence_min_max(mode):
    """
    Calling the new target class with legacy arguments yields an object equivalent
    to the legacy object.
    """  # noqa
    groups = [
        (
            # ------------
            # Legacy style
            ModernTarget("t", mode),
            ModernTarget("t", mode=mode),
            ModernTarget(name="t", mode=mode),
            # ------------
            # Modern style
            ModernTarget("t", minimize=mode == "MIN"),
        ),
        (
            # ------------
            # Legacy style
            ModernTarget("t", mode, (1, 2)),
            ModernTarget("t", mode, bounds=(1, 2)),
            ModernTarget("t", mode=mode, bounds=(1, 2)),
            ModernTarget(name="t", mode=mode, bounds=(1, 2)),
            ModernTarget("t", mode, (1, 2), "LINEAR"),
            ModernTarget("t", mode, (1, 2), transformation="LINEAR"),
            ModernTarget("t", mode, bounds=(1, 2), transformation="LINEAR"),
            ModernTarget("t", mode=mode, bounds=(1, 2), transformation="LINEAR"),
            ModernTarget(name="t", mode=mode, bounds=(1, 2), transformation="LINEAR"),
            # ------------
            # Modern style
            ModernTarget.normalized_ramp(
                name="t", cutoffs=(1, 2), descending=mode == "MIN"
            ),
        ),
    ]
    for targets in groups:
        for t1, t2 in pairwise(targets):
            assert t1 == t2


@pytest.mark.parametrize("transformation", ["TRIANGULAR", "BELL"])
def test_constructor_equivalence_match(transformation):
    """
    Calling the new target class with legacy arguments yields an object equivalent
    to the legacy object.
    """  # noqa
    # ------------
    # Legacy style
    targets = (
        ModernTarget("t", "MATCH", (1, 2), transformation),
        ModernTarget("t", "MATCH", (1, 2), transformation=transformation),
        ModernTarget("t", "MATCH", bounds=(1, 2), transformation=transformation),
        ModernTarget("t", mode="MATCH", bounds=(1, 2), transformation=transformation),
        ModernTarget(
            name="t", mode="MATCH", bounds=(1, 2), transformation=transformation
        ),
    )
    if transformation == "TRIANGULAR":
        targets += (
            ModernTarget("t", "MATCH", (1, 2)),
            ModernTarget("t", "MATCH", bounds=(1, 2)),
            ModernTarget("t", mode="MATCH", bounds=(1, 2)),
            ModernTarget(name="t", mode="MATCH", bounds=(1, 2)),
        )

    # ------------
    # Modern style
    if transformation == "BELL":
        targets += (ModernTarget.match_bell("t", match_value=1.5, sigma=0.5),)
    else:
        targets += (
            ModernTarget.match_triangular("t", cutoffs=(1, 2)),
            ModernTarget.match_triangular("t", match_value=1.5, width=1),
            ModernTarget.match_triangular("t", match_value=1.5, margins=(0.5, 0.5)),
        )

    for t1, t2 in pairwise(targets):
        assert t1 == t2


@pytest.mark.parametrize(
    ("legacy", "deprecation", "modern", "expected"),
    [
        param(
            LegacyTarget("t", "MAX"),
            ModernTarget("t", "MAX"),
            ModernTarget("t"),
            sample_input(),
            id="max",
        ),
        param(
            LegacyTarget("t", "MAX", (0, 1), "LINEAR"),
            ModernTarget("t", "MAX", (0, 1), "LINEAR"),
            ModernTarget("t").clamp(0, 1),
            linear_transform(sample_input(), 0, 1, descending=False),
            id="max_clamped",
        ),
        param(
            LegacyTarget("t", "MAX", (2, 5), "LINEAR"),
            ModernTarget("t", "MAX", (2, 5), "LINEAR"),
            ModernTarget.normalized_ramp("t", (2, 5)),
            linear_transform(sample_input(), 2, 5, descending=False),
            id="max_shifted_clamped",
        ),
        param(
            # NOTE: Minimization transformation without bounds is not possible with
            #   legacy interface."
            None,
            None,
            ModernTarget("t", AffineTransformation(factor=-1)),
            -sample_input(),
            id="min_no_bounds",
        ),
        param(
            # NOTE: Minimization transformation without bounds is not possible with
            #   legacy interface."
            None,
            None,
            ModernTarget("t", minimize=True),
            sample_input(),
            id="min_no_bounds_with_flag",
        ),
        param(
            # NOTE: Minimization without bounds had no effect on the transformation
            #   of the legacy target since minimization was handled in the construction
            #   of the acquisition function:
            #   * https://github.com/emdgroup/baybe/pull/462
            #   * https://github.com/emdgroup/baybe/issues/460
            None,  # should be `LegacyTarget("t", "MIN")` but see explanation above
            ModernTarget("t", "MIN"),
            ModernTarget("t", minimize=True),
            sample_input(),
            id="min",
        ),
        param(
            LegacyTarget("t", "MIN", (0, 1), "LINEAR"),
            ModernTarget("t", "MIN", (0, 1), "LINEAR"),
            ModernTarget.normalized_ramp("t", (0, 1), descending=True),
            linear_transform(sample_input(), 0, 1, descending=True),
            id="min_clamped",
        ),
        param(
            LegacyTarget("t", "MIN", (2, 5), "LINEAR"),
            ModernTarget("t", "MIN", (2, 5), "LINEAR"),
            ModernTarget.normalized_ramp("t", (2, 5), descending=True),
            linear_transform(sample_input(), 2, 5, descending=True),
            id="min_shifted_clamped",
        ),
        param(
            LegacyTarget("t", "MATCH", (-1, 1), "BELL"),
            ModernTarget("t", "MATCH", (-1, 1), "BELL"),
            ModernTarget.match_bell("t", match_value=0, sigma=1),
            bell_transform(sample_input(), -1, 1),
            id="match_bell_unit_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (1, 3), "BELL"),
            ModernTarget("t", "MATCH", (1, 3), "BELL"),
            ModernTarget.match_bell("t", match_value=2, sigma=1),
            bell_transform(sample_input(), 1, 3),
            id="match_bell_unit_shifted",
        ),
        param(
            LegacyTarget("t", "MATCH", (-5, 5), "BELL"),
            ModernTarget("t", "MATCH", (-5, 5), "BELL"),
            ModernTarget.match_bell("t", match_value=0, sigma=5),
            bell_transform(sample_input(), -5, 5),
            id="match_bell_scaled_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (2, 6), "BELL"),
            ModernTarget("t", "MATCH", (2, 6), "BELL"),
            ModernTarget.match_bell("t", match_value=4, sigma=2),
            bell_transform(sample_input(), 2, 6),
            id="match_bell_scaled_shifted",
        ),
        param(
            LegacyTarget("t", "MATCH", (-1, 1), "TRIANGULAR"),
            ModernTarget("t", "MATCH", (-1, 1), "TRIANGULAR"),
            ModernTarget.match_triangular("t", cutoffs=(-1, 1)),
            triangular_transform(sample_input(), -1, 1),
            id="match_triangular_unit_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (1, 3), "TRIANGULAR"),
            ModernTarget("t", "MATCH", (1, 3), "TRIANGULAR"),
            ModernTarget.match_triangular("t", cutoffs=(1, 3)),
            triangular_transform(sample_input(), 1, 3),
            id="match_triangular_unit_shifted",
        ),
        param(
            LegacyTarget("t", "MATCH", (-5, 5), "TRIANGULAR"),
            ModernTarget("t", "MATCH", (-5, 5), "TRIANGULAR"),
            ModernTarget.match_triangular("t", cutoffs=(-5, 5)),
            triangular_transform(sample_input(), -5, 5),
            id="match_triangular_scaled_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (2, 6), "TRIANGULAR"),
            ModernTarget("t", "MATCH", (2, 6), "TRIANGULAR"),
            ModernTarget.match_triangular("t", cutoffs=(2, 6)),
            triangular_transform(sample_input(), 2, 6),
            id="match_triangular_scaled_shifted",
        ),
    ],
)
def test_target_transformation(
    series,
    legacy: LegacyTarget,
    deprecation: ModernTarget,
    modern: ModernTarget,
    expected,
):
    """The legacy and modern target variants transform equally."""
    expected = pd.Series(expected)
    if legacy is not None:
        assert_series_equal(legacy.transform(series), expected)
    if deprecation is not None:
        assert_series_equal(deprecation.transform(series), expected)
    assert_series_equal(modern.transform(series), expected)
