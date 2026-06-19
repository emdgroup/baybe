"""Deprecation tests."""

import os
import warnings
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest import param

from baybe._optional.info import CHEM_INSTALLED, POLARS_INSTALLED
from baybe.constraints import (
    ContinuousLinearConstraint,
)
from baybe.constraints.conditions import SubSelectionCondition
from baybe.constraints.continuous import ContinuousCardinalityConstraint
from baybe.constraints.discrete import (
    DiscreteBatchConstraint,
    DiscreteExcludeConstraint,
)
from baybe.exceptions import DeprecationError
from baybe.kernels.basic import MaternKernel
from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.categorical import CategoricalParameter, TaskParameter
from baybe.parameters.enum import SubstanceEncoding
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian import (
    BotorchRecommender,
)
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.searchspace.core import SearchSpace
from baybe.searchspace.discrete import SubspaceDiscrete
from baybe.searchspace.validation import get_transform_parameters
from baybe.settings import Settings
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.targets import NumericalTarget
from baybe.targets import NumericalTarget as ModernTarget
from baybe.targets._deprecated import (
    LegacyTarget,
    bell_transform,
    linear_transform,
    triangular_transform,
)
from baybe.targets.base import Target
from baybe.targets.binary import BinaryTarget
from baybe.transformations.basic import AffineTransformation
from baybe.utils.dataframe import create_fake_input
from baybe.utils.random import set_random_seed, temporary_seed


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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
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


# NOTE: The parametrize values below use the deprecated legacy interface of
# ModernTarget (e.g. ModernTarget("t", "MAX")), which emits DeprecationWarning.
# Since these are evaluated at module/collection time, we suppress the warning here
# to avoid failures when running with `-W error`.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    _target_transformation_params = [
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
    ]


@pytest.mark.parametrize(
    ("legacy", "deprecation", "modern", "expected"),
    _target_transformation_params,
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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_deserialization_using_constructor():
    """Deserialization using the 'constructor' field works despite having other
    deprecation mechanisms in place."""  # noqa
    config = """
    {
        "type": "NumericalTarget",
        "name": "t_max_bounds",
        "constructor": "normalized_ramp",
        "cutoffs": [0, 100]
    }
    """
    t_old = NumericalTarget("t_max_bounds", mode="MAX", bounds=(0, 100))
    t_new = NumericalTarget.normalized_ramp("t_max_bounds", cutoffs=(0, 100))
    t_new_config = Target.from_json(config)
    assert t_old == t_new == t_new_config


def test_deprecated_random_seed_control():
    """Using the deprecated random seed helpers raises warnings."""
    with pytest.warns(DeprecationWarning, match="'set_random_seed' is deprecated"):
        set_random_seed(42)
    with pytest.warns(DeprecationWarning, match="'temporary_seed' is deprecated"):
        temporary_seed(42)


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("library", ["Numpy", "Torch"])
def test_deprecated_floating_point_environment_variables(
    monkeypatch, library: str, value: bool
):
    """Using the deprecated precision environment variables raises warnings."""
    monkeypatch.setenv(f"BAYBE_{library.upper()}_USE_SINGLE_PRECISION", str(value))
    with pytest.warns(
        DeprecationWarning,
        match=f"'BAYBE_{library.upper()}_USE_SINGLE_PRECISION' has been deprecated",
    ):
        attr = getattr(Settings(restore_environment=True), f"DTypeFloat{library}")
    namespace = np if library == "Numpy" else torch
    assert attr is namespace.float32 if value else namespace.float64


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            False,
            marks=pytest.mark.skipif(
                not POLARS_INSTALLED, reason="Optional polars dependency not installed."
            ),
        ),
        True,
    ],
)
def test_deprecated_polars_environment_variables(monkeypatch, value: bool):
    """Using the deprecated polars environment variables raises warnings."""
    monkeypatch.setenv("BAYBE_DEACTIVATE_POLARS", str(value))
    with pytest.warns(
        DeprecationWarning, match="'BAYBE_DEACTIVATE_POLARS' has been deprecated"
    ):
        assert (
            Settings(restore_environment=True).use_polars_for_constraints is not value
        )


@pytest.mark.parametrize("value", [True, False])
def test_deprecated_parallelization_environment_variables(monkeypatch, value: bool):
    """Using the deprecated parallelization environment variables raises warnings."""
    monkeypatch.setenv("BAYBE_PARALLEL_SIMULATION_RUNS", str(value))
    with pytest.warns(
        DeprecationWarning, match="'BAYBE_PARALLEL_SIMULATION_RUNS' has been deprecated"
    ):
        assert Settings(restore_environment=True).parallelize_simulation_runs is value


@pytest.fixture
def set_cache_directory_env_var(monkeypatch, value: str):
    """Remove the translated environment variable after the test."""
    monkeypatch.setenv("BAYBE_CACHE_DIR", value)
    yield
    os.environ.pop("BAYBE_CACHE_DIRECTORY", None)


@pytest.mark.parametrize(
    ("value", "expected"), [("test", Path("test")), ("", None)], ids=["set", "None"]
)
@pytest.mark.usefixtures("set_cache_directory_env_var")
def test_deprecated_cache_environment_variables(monkeypatch, value: str, expected: str):
    """Using the deprecated cache environment variables raises warnings."""
    monkeypatch.setenv("BAYBE_CACHE_DIR", value)
    with pytest.warns(
        DeprecationWarning, match="'BAYBE_CACHE_DIR' has been deprecated"
    ):
        assert Settings(restore_environment=True).cache_directory == expected


@pytest.mark.parametrize("custom", [False, True], ids=["default", "custom"])
@pytest.mark.parametrize("env", [False, True], ids=["no_env", "env"])
@pytest.mark.parametrize("task", [False, True], ids=["no_task", "task"])
def test_multitask_kernel_deprecation(monkeypatch, custom: bool, env: bool, task: bool):
    """Providing a custom kernel in a transfer learning context raises a deprecation
    error unless explicitly disabled via environment variable."""  # noqa
    parameters = [NumericalDiscreteParameter("p", [0, 1])]
    if task:
        parameters.append(TaskParameter("task", ["a", "b"]))
    searchspace = SearchSpace.from_product(parameters)
    objective = NumericalTarget("t").to_objective()
    measurements = create_fake_input(
        searchspace.parameters, objective.targets, n_rows=2
    )
    args = (MaternKernel(),) if custom else ()

    if env:
        monkeypatch.setenv("BAYBE_DISABLE_CUSTOM_KERNEL_WARNING", "True")

    context = (
        pytest.raises(DeprecationError)
        if task and custom and not env
        else nullcontext()
    )
    with context:
        GaussianProcessSurrogate(*args).fit(searchspace, objective, measurements)


@pytest.mark.parametrize(
    "attr", ["n_fits_done", "n_batches_done"], ids=["fits", "batches"]
)
def test_deprecated_campaign_counters(campaign, attr):
    """Accessing the deprecated campaign counter properties raises an error."""
    with pytest.raises(DeprecationError, match=attr):
        getattr(campaign, attr)


@pytest.mark.parametrize("batch_size", [3], ids=["b3"])
@pytest.mark.parametrize("n_iterations", [1], ids=["i1"])
def test_legacy_campaign_counter_deserialization(ongoing_campaign):
    """Deserializing a campaign with legacy counter fields and columns still works."""
    from baybe.campaign import Campaign
    from baybe.serialization import converter

    # Serialize, then inject legacy fields
    data = ongoing_campaign.to_dict()
    assert "n_fits_done" not in data
    assert "n_batches_done" not in data
    data["n_fits_done"] = 3
    data["n_batches_done"] = 2

    # Inject legacy columns into measurements
    # (use legacy key name "measurements_exp" to test migration hook)
    meas = converter.structure(data.pop("measurements"), pd.DataFrame)
    meas["FitNr"] = 1.0
    meas["BatchNr"] = 1
    data["measurements_exp"] = converter.unstructure(meas)

    # Deserialization must not raise and legacy columns must be stripped
    restored = Campaign.from_dict(data)
    assert "FitNr" not in restored._measurements.columns
    assert "BatchNr" not in restored._measurements.columns


@pytest.mark.parametrize("batch_size", [3], ids=["b3"])
@pytest.mark.parametrize("n_iterations", [1], ids=["i1"])
def test_legacy_recommended_metadata_deserialization(ongoing_campaign):
    """Legacy searchspace_metadata 'recommended' column migrates to new field."""
    from baybe.campaign import _RECOMMENDED, Campaign
    from baybe.serialization import converter

    # Recommend to mark some entries as recommended
    rec = ongoing_campaign.recommend(batch_size=2)
    n_recommended = len(rec)

    # Serialize and simulate legacy format (no recommended/excluded_experiments)
    data = ongoing_campaign.to_dict()
    del data["recommended_experiments"]
    del data["excluded_experiments"]

    # Construct legacy searchspace_metadata with a "recommended" column
    exp_rep = ongoing_campaign.searchspace.discrete.exp_rep
    metadata = pd.DataFrame(False, index=exp_rep.index, columns=[_RECOMMENDED])
    idxs = rec.index[:n_recommended]
    metadata.loc[idxs, _RECOMMENDED] = True
    data["searchspace_metadata"] = converter.unstructure(metadata)

    # Deserialization must reconstruct _recommended_experiments with correct content
    restored = Campaign.from_dict(data)
    expected = exp_rep.loc[idxs]
    # Compare as sets of rows (order may differ)
    restored_sorted = restored._recommended_experiments.sort_values(
        restored._recommended_experiments.columns.tolist()
    ).reset_index(drop=True)
    expected_sorted = expected.sort_values(expected.columns.tolist()).reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(restored_sorted, expected_sorted)


def test_legacy_empty_dataframe_schema_deserialization():
    """Legacy campaigns with schema-less empty DataFrames get correct columns."""
    from baybe.campaign import Campaign
    from baybe.parameters.numerical import NumericalDiscreteParameter
    from baybe.serialization import converter
    from baybe.targets.numerical import NumericalTarget

    p = NumericalDiscreteParameter("x", [1, 2, 3])
    t = NumericalTarget("y")
    campaign = Campaign(p.to_searchspace(), t.to_objective())

    # Simulate legacy serialization: replace with column-less empty DataFrames
    data = campaign.to_dict()
    data["measurements"] = converter.unstructure(pd.DataFrame())

    restored = Campaign.from_dict(data)
    assert restored._measurements.columns.tolist() == ["x", "y"]
    assert restored._recommended_experiments.columns.tolist() == ["x"]
    assert restored._measurements.empty
    assert restored._recommended_experiments.empty
    assert restored == campaign


def test_legacy_measured_metadata_deserialization():
    """Legacy searchspace_metadata 'measured' column is discarded during loading."""
    from baybe.campaign import _MEASURED, Campaign
    from baybe.parameters.numerical import NumericalDiscreteParameter
    from baybe.serialization import converter
    from baybe.targets.numerical import NumericalTarget

    p = NumericalDiscreteParameter("x", [1, 2, 3])
    t = NumericalTarget("y")
    campaign = Campaign(p.to_searchspace(), t.to_objective())

    # Simulate legacy format: searchspace_metadata with a "measured" column
    data = campaign.to_dict()
    metadata = pd.DataFrame(
        {_MEASURED: [True, False, False]},
        index=campaign.searchspace.discrete.exp_rep.index,
    )
    data["searchspace_metadata"] = converter.unstructure(metadata)

    # Deserialization must handle the legacy column without errors
    restored = Campaign.from_dict(data)
    assert restored == campaign


def test_legacy_excluded_metadata_deserialization():
    """Legacy searchspace_metadata 'excluded' column migrates to new field."""
    from baybe.campaign import _EXCLUDED, Campaign
    from baybe.parameters.numerical import NumericalDiscreteParameter
    from baybe.serialization import converter
    from baybe.targets.numerical import NumericalTarget

    p = NumericalDiscreteParameter("x", [1, 2, 3])
    t = NumericalTarget("y")
    campaign = Campaign(p.to_searchspace(), t.to_objective())

    # Simulate legacy format: searchspace_metadata with an "excluded" column,
    # and no excluded_experiments field
    data = campaign.to_dict()
    del data["excluded_experiments"]
    exp_rep = campaign.searchspace.discrete.exp_rep
    metadata = pd.DataFrame(
        {_EXCLUDED: [True, False, True]},
        index=exp_rep.index,
    )
    data["searchspace_metadata"] = converter.unstructure(metadata)

    # Deserialization must reconstruct _excluded_experiments
    restored = Campaign.from_dict(data)
    excluded_idxs = metadata.index[metadata[_EXCLUDED]]
    expected = exp_rep.loc[excluded_idxs].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        restored._excluded_experiments.sort_values(
            restored._excluded_experiments.columns.tolist()
        ).reset_index(drop=True),
        expected.sort_values(expected.columns.tolist()).reset_index(drop=True),
    )


@pytest.mark.parametrize("positional", [True, False])
def test_deprecated_constraints_arguments(positional):
    """Using the deprecated subspace constraint arguments raises a warning."""
    p = NumericalContinuousParameter("p", (0, 1))
    c = ContinuousLinearConstraint(["p"], "=", [0], 0)
    c_lin_eq = ContinuousLinearConstraint(["p"], "=", [1], 0)
    c_lin_ineq = ContinuousLinearConstraint(["p"], ">=", [1], 0)
    c_nonlin = ContinuousCardinalityConstraint(["p"], 1)

    with pytest.warns(DeprecationWarning):
        if positional:
            subspace = SubspaceContinuous(
                parameters=(p,),
                constraints=(c,),
                constraints_lin_eq=(c_lin_eq,),
                constraints_lin_ineq=(c_lin_ineq,),
                constraints_nonlin=(c_nonlin,),
            )
        else:
            subspace = SubspaceContinuous(
                (p,),
                (c, c_lin_eq),
                (c_lin_ineq,),
                (c_nonlin,),
            )

    assert c in subspace.constraints
    assert c_lin_eq in subspace.constraints
    assert c_lin_ineq in subspace.constraints
    assert c_nonlin in subspace.constraints


def test_deprecated_constraints_arguments_deserialization():
    """Deserialization from legacy JSON with deprecated constraint attributes works."""
    p1 = NumericalContinuousParameter("p", (0, 1))
    c_lin_eq = ContinuousLinearConstraint(["p"], "=", [1], 1)
    c_lin_ineq = ContinuousLinearConstraint(["p"], ">=", [1], 0)
    c_nonlin = ContinuousCardinalityConstraint(["p"], 1)

    # Construct the expected object using the modern interface
    expected = SubspaceContinuous(
        parameters=(p1,),
        constraints=(c_lin_eq, c_lin_ineq, c_nonlin),
    )

    # Build a legacy dict with the deprecated constraint field names
    legacy_dict = {
        "type": "SubspaceContinuous",
        "parameters": [p1.to_dict()],
        "constraints_lin_eq": [c_lin_eq.to_dict()],
        "constraints_lin_ineq": [c_lin_ineq.to_dict()],
        "constraints_nonlin": [c_nonlin.to_dict()],
    }

    with pytest.warns(DeprecationWarning):
        actual = SubspaceContinuous.from_dict(legacy_dict)

    assert actual == expected


@pytest.mark.parametrize(
    ("arg", "error"), [("empty_encoding", False), ("comp_rep", True)]
)
def test_deprecated_subspace_discrete_arguments(arg, error):
    """Providing deprecated arguments to `SubspaceDiscrete` raises an error / a warning."""  # noqa
    context = (
        pytest.raises(DeprecationError, match=f"Providing '{arg}'")
        if error
        else pytest.warns(DeprecationWarning, match=f"Providing '{arg}'")
    )
    with context:
        SubspaceDiscrete(parameters=[], exp_rep=pd.DataFrame(), **{arg: 0})


def test_deprecated_empty_encoding_from_product():
    """Passing `empty_encoding` to `SubspaceDiscrete.from_product` raises a warning."""  # noqa
    with pytest.warns(DeprecationWarning, match="Providing 'empty_encoding'"):
        SubspaceDiscrete.from_product(
            parameters=[NumericalDiscreteParameter("p", [0, 1])],
            empty_encoding=True,
        )


def test_deprecated_empty_encoding_from_dataframe():
    """Passing `empty_encoding` to `SubspaceDiscrete.from_dataframe` raises a warning."""  # noqa
    with pytest.warns(DeprecationWarning, match="Providing 'empty_encoding'"):
        SubspaceDiscrete.from_dataframe(
            parameters=[NumericalDiscreteParameter("p", [0, 1])],
            df=pd.DataFrame({"p": [0, 1]}),
            empty_encoding=True,
        )


def test_deprecated_discrete_subspace_deserialization():
    """Deserialization from legacy JSON with `empty_encoding`/`comp_rep` works."""
    p = NumericalDiscreteParameter("p", [0, 1])
    expected = SubspaceDiscrete.from_product(parameters=[p])

    # Build a legacy dict containing the deprecated fields
    legacy_dict = expected.to_dict()
    legacy_dict["empty_encoding"] = False
    legacy_dict["comp_rep"] = legacy_dict["exp_rep"]

    actual = SubspaceDiscrete.from_dict(legacy_dict)
    assert actual == expected


def test_deprecated_constraints_deserialization():
    """Deserialization of legacy ``constraints`` key migrates batch constraints."""
    p = NumericalDiscreteParameter("p", [0, 1, 2])
    batch_c = DiscreteBatchConstraint(["p"])
    expected = SubspaceDiscrete.from_product(parameters=[p], constraints=[batch_c])

    # Simulate a legacy dict with `constraints` instead of `batch_constraints`
    legacy_dict = expected.to_dict()
    legacy_dict["constraints"] = [batch_c.to_dict()]
    del legacy_dict["batch_constraints"]

    with pytest.warns(DeprecationWarning, match="Providing 'constraints'"):
        actual = SubspaceDiscrete.from_dict(legacy_dict)

    assert actual == expected


def test_deprecated_constraints_argument():
    """Passing `constraints` to `SubspaceDiscrete` raises a deprecation warning."""
    p = NumericalDiscreteParameter("p", [0, 1, 2])
    batch_c = DiscreteBatchConstraint(["p"])
    with pytest.warns(DeprecationWarning, match="Providing 'constraints'"):
        subspace = SubspaceDiscrete(
            parameters=[p],
            exp_rep=pd.DataFrame({"p": [0, 1, 2]}),
            constraints=[batch_c],
        )
    # The batch constraint must be migrated to `batch_constraints`
    assert subspace.batch_constraints == (batch_c,)


def test_deprecated_constraints_argument_from_product():
    """Passing mixed constraints to ``from_product`` routes batch constraints correctly."""  # noqa: E501
    p = CategoricalParameter("p", ["a", "b"])
    q = CategoricalParameter("q", ["x", "y"])
    batch_c = DiscreteBatchConstraint(["p"])
    no_dup_c = DiscreteExcludeConstraint(["p"], [SubSelectionCondition(["a"])])

    ss_both = SubspaceDiscrete.from_product(
        parameters=[p, q], constraints=[batch_c, no_dup_c]
    )
    ss_none = SubspaceDiscrete.from_product(parameters=[p, q], constraints=[])
    ss_with_batch = SubspaceDiscrete.from_product(
        parameters=[p, q], constraints=[batch_c]
    )
    ss_without_batch = SubspaceDiscrete.from_product(
        parameters=[p, q], constraints=[no_dup_c]
    )

    assert ss_both.batch_constraints == ss_with_batch.batch_constraints == (batch_c,)
    assert ss_without_batch.batch_constraints == ss_none.batch_constraints == ()
    assert_frame_equal(ss_both.exp_rep, ss_without_batch.exp_rep)
    assert_frame_equal(ss_with_batch.exp_rep, ss_none.exp_rep)
    assert len(ss_both.exp_rep) == 2
    assert len(ss_none.exp_rep) == 4
