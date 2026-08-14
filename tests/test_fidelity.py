"""Tests for fidelity parameters."""

import sys

import pandas as pd
import pytest
from botorch.models import SingleTaskGP, SingleTaskMultiFidelityGP
from gpytorch.likelihoods import GaussianLikelihood
from pytest import param

from baybe.exceptions import IncompatibleSurrogateError
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.fidelity import (
    CategoricalFidelityParameter,
    NumericalDiscreteFidelityParameter,
)
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.presets.core import GaussianProcessPreset
from baybe.targets.numerical import NumericalTarget
from baybe.utils.dataframe import create_fake_input

_num_fid_param = NumericalDiscreteFidelityParameter(
    "fidelity", values=[0.5, 1.0], costs=[1.0, 10.0]
)
_cat_fid_param = CategoricalFidelityParameter(
    "fidelity", values=["lo", "hi"], costs=[1.0, 10.0], zeta=[0.5, 0.0]
)
_design_param = NumericalDiscreteParameter("x", values=[1.0, 2.0, 3.0])

searchspace_num_fid = SearchSpace.from_product([_design_param, _num_fid_param])
searchspace_cat_fid = SearchSpace.from_product([_design_param, _cat_fid_param])

objective = NumericalTarget("t").to_objective()
measurements_num_fid = create_fake_input(
    searchspace_num_fid.parameters, objective.targets, n_rows=3
)
measurements_cat_fid = create_fake_input(
    searchspace_cat_fid.parameters, objective.targets, n_rows=3
)


def test_categorical_fidelity_parameter_construction():
    """Equivalent zeta formats and value orderings produce equal objects."""
    p1 = CategoricalFidelityParameter("p", values=["h", "l"], costs=[1, 2], zeta=5)
    p2 = CategoricalFidelityParameter("p", values=["l", "h"], costs=[2, 1], zeta=[5, 0])
    assert p1 == p2


def test_numerical_discrete_fidelity_parameter_construction():
    """Fidelity values and costs are sorted according to numerical fidelity values."""
    p1 = NumericalDiscreteFidelityParameter("p", values=[0, 0.5, 1], costs=[1, 2, 3])
    p2 = NumericalDiscreteFidelityParameter("p", values=[0.5, 1, 0], costs=[2, 3, 1])
    assert p1 == p2


@pytest.mark.parametrize(
    ("parameter", "series", "expected"),
    [
        param(
            CategoricalFidelityParameter(
                "fidelity", values=["low", "high"], costs=[1, 2], zeta=[1, 0]
            ),
            pd.Series(["low", "high", "low"], name="fidelity"),
            [1.0, 0.0, 1.0],
            id="categorical",
        ),
        param(
            CategoricalFidelityParameter(
                "fidelity", values=["low", "high"], costs=[1, 2], zeta=5
            ),
            pd.Series(["low", "high", "low"], name="fidelity"),
            [1.0, 0.0, 1.0],
            id="categorical_scalar_zeta",
        ),
        param(
            NumericalDiscreteFidelityParameter(
                "fidelity", values=[0, 0.5, 1], costs=[1, 2, 3]
            ),
            pd.Series([0.5, 1.0, 0.0], name="fidelity"),
            [0.5, 1.0, 0.0],
            id="numerical_discrete",
        ),
    ],
)
def test_fidelity_parameter_transform(parameter, series, expected):
    """Transform must correctly map fidelity values to computational representation."""
    result = parameter.transform(series)
    assert list(result["fidelity"]) == expected


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        param(
            [
                CategoricalFidelityParameter(
                    "f1", values=["lo", "hi"], costs=[1, 10], zeta=[0.5, 0.0]
                ),
                CategoricalFidelityParameter(
                    "f2", values=["a", "b"], costs=[1, 5], zeta=[0.3, 0.0]
                ),
            ],
            "at most one fidelity",
            id="two_categorical_fidelity",
        ),
        param(
            [
                CategoricalFidelityParameter(
                    "f1", values=["lo", "hi"], costs=[1, 10], zeta=[0.5, 0.0]
                ),
                NumericalDiscreteFidelityParameter(
                    "f2", values=[0.5, 1.0], costs=[1, 10]
                ),
            ],
            "at most one fidelity",
            id="mixed_fidelity_types",
        ),
        param(
            [
                NumericalDiscreteFidelityParameter(
                    "f1", values=[0.5, 1.0], costs=[1, 10]
                ),
                NumericalDiscreteFidelityParameter(
                    "f2", values=[0.2, 1.0], costs=[1, 5]
                ),
            ],
            "at most one fidelity",
            id="two_numerical_fidelity",
        ),
        param(
            [
                TaskParameter("task", values=["a", "b"]),
                CategoricalFidelityParameter(
                    "f", values=["lo", "hi"], costs=[1, 10], zeta=[0.5, 0.0]
                ),
            ],
            "task parameters with fidelity parameters",
            id="task_plus_fidelity",
        ),
    ],
)
def test_invalid_fidelity_parameter_combinations(parameters, match):
    """Search spaces with invalid fidelity parameter combinations are rejected."""
    with pytest.raises(NotImplementedError, match=match):
        SearchSpace.from_product(parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        param([TaskParameter("task", values=["a", "b"])], id="task_only"),
        param([_cat_fid_param], id="categorical_fidelity_only"),
        param([_num_fid_param], id="numerical_fidelity_only"),
    ],
)
def test_surrogate_rejects_index_only_searchspace(parameters):
    """GP surrogates raise for search spaces without regular model inputs."""
    searchspace = SearchSpace.from_product(parameters)
    measurements = create_fake_input(
        searchspace.parameters, objective.targets, n_rows=3
    )

    with pytest.raises(IncompatibleSurrogateError, match="at least one regular"):
        GaussianProcessSurrogate().fit(searchspace, objective, measurements)


def test_gp_rejects_custom_components_numerical_fidelity():
    """Custom components are rejected for numerical multi-fidelity search spaces."""
    surrogate = GaussianProcessSurrogate(likelihood_or_factory=GaussianLikelihood())
    with pytest.raises(IncompatibleSurrogateError, match="custom components"):
        surrogate.fit(searchspace_num_fid, objective, measurements_num_fid)


@pytest.mark.parametrize(
    ("searchspace", "measurements", "expected_model"),
    [
        param(
            searchspace_cat_fid, measurements_cat_fid, SingleTaskGP, id="categorical"
        ),
        param(
            searchspace_num_fid,
            measurements_num_fid,
            SingleTaskMultiFidelityGP,
            id="numerical",
        ),
    ],
)
def test_standard_gp_fit_fidelity(searchspace, measurements, expected_model):
    """GaussianProcessSurrogate fits a fidelity space with the expected model."""
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)
    # Exact type check: SingleTaskMultiFidelityGP subclasses SingleTaskGP, so the
    # categorical case must not accidentally match the multi-fidelity model.
    assert type(surrogate.to_botorch()) is expected_model
    stats = surrogate.posterior_stats(measurements)
    assert set(stats.columns) == {"t_mean", "t_std"}
    assert len(stats) == len(measurements)


@pytest.mark.parametrize(
    "preset",
    [
        param(
            preset,
            marks=pytest.mark.skipif(
                preset is GaussianProcessPreset.BOTORCH and sys.version_info < (3, 11),
                reason="BoTorch >=0.18.0 requires Python >=3.11.",
            ),
        )
        for preset in GaussianProcessPreset
    ],
    ids=lambda preset: preset.value,
)
def test_gp_presets_fit_categorical_fidelity(preset):
    """All GP presets can be fitted on a categorical fidelity space."""
    surrogate = GaussianProcessSurrogate.from_preset(preset)
    surrogate.fit(searchspace_cat_fid, objective, measurements_cat_fid)


@pytest.mark.parametrize(
    "preset", list(GaussianProcessPreset), ids=lambda preset: preset.value
)
def test_gp_presets_reject_numerical_fidelity(preset):
    """No GP preset can be fitted on a numerical fidelity space.

    Numerical multi-fidelity spaces are delegated to BoTorch's
    ``SingleTaskMultiFidelityGP``, which builds its own components. Since every
    preset supplies explicit component factories, fitting must be rejected.
    """
    surrogate = GaussianProcessSurrogate.from_preset(preset)
    with pytest.raises(IncompatibleSurrogateError, match="custom components"):
        surrogate.fit(searchspace_num_fid, objective, measurements_num_fid)
