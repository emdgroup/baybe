"""Scope and real-behavior regression for the product-free multistep fit path.

The product-free layout and its surrogate fit integration belong to the private
multi-step layer. The generic surrogate modules and the built-in EDBO preset must
remain byte-for-byte behaviorally untouched, while the EDBO factories must still
work against a layout without materializing a search-space product.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from baybe.exceptions import IncompatibilityError
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace.core import SearchSpace
from baybe.targets.numerical import NumericalTarget


def _make_parameters() -> tuple:
    """A mixed parameter set with continuous, discrete, and categorical inputs."""
    return (
        NumericalContinuousParameter("x", bounds=(0.0, 2.0)),
        NumericalDiscreteParameter("d", values=(0.0, 0.5, 1.0)),
        CategoricalParameter("c", values=("a", "b"), encoding=CategoricalEncoding.OHE),
    )


def _make_frame() -> pd.DataFrame:
    """Deterministic observations over the mixed parameter set."""
    return pd.DataFrame(
        {
            "x": [0.1, 0.9, 1.8, 0.4, 1.2],
            "d": [0.0, 0.5, 1.0, 0.5, 0.0],
            "c": ["a", "b", "a", "b", "a"],
            "y": [0.2, -0.1, 0.7, 0.0, 0.4],
        }
    )


def _seed() -> None:
    """Replay the same random stream for a paired fit."""
    import random

    import numpy as np
    import torch

    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)


def _forbid_product_builders(monkeypatch) -> None:
    """Fail if a product builder is called during the product-free fit."""

    def forbidden(*args, **kwargs):
        raise AssertionError("A parameter product builder was called.")

    from baybe.searchspace.continuous import SubspaceContinuous
    from baybe.searchspace.discrete import SubspaceDiscrete

    monkeypatch.setattr(SearchSpace, "from_product", forbidden)
    monkeypatch.setattr(SubspaceDiscrete, "from_product", forbidden)
    monkeypatch.setattr(SubspaceContinuous, "from_product", forbidden)


def test_one_factorized_layout_module() -> None:
    """Only the configuration layout survives; ``_layout.py`` is gone."""
    import baybe.multistep
    from baybe.multistep._configuration_space import _ConfigurationSpace

    package = Path(baybe.multistep.__file__).parent
    assert not (package / "_layout.py").exists()
    assert importlib.util.find_spec("baybe.searchspace._factorized") is None
    assert _ConfigurationSpace.__module__ == "baybe.multistep._configuration_space"
    for name in (
        "from_parameters",
        "from_dag",
        "scaling_bounds",
        "discrete",
        "task_idx",
        "n_tasks",
        "active_levels",
        "get_comp_rep_parameter_indices",
        "get_parameters_by_name",
        "_drop_parameters",
        "transform",
    ):
        assert hasattr(_ConfigurationSpace, name)


@pytest.mark.parametrize(
    "module_name", ["baybe.surrogates.base", "baybe.surrogates.gaussian_process.core"]
)
def test_generic_surrogate_modules_have_no_layout_integration(module_name) -> None:
    """The generic surrogate modules do not know about the private layout."""
    module = __import__(module_name, fromlist=["__file__"])
    source = Path(module.__file__).read_text()
    assert "_factorized" not in source
    assert "_fit_with_layout" not in source
    assert "_fit_input_context" not in source


def test_edbo_preset_keeps_its_original_discrete_access() -> None:
    """The original EDBO factories remain unmodified and need a local adaptation."""
    from baybe.surrogates.gaussian_process.presets import edbo

    source = Path(edbo.__file__).read_text()
    assert "searchspace.discrete" in source
    assert "SubspaceDiscrete" in source
    assert "searchspace.parameters" not in source


@pytest.mark.parametrize("preset", ["EDBO"])
def test_edbo_layout_fit_matches_real_searchspace(preset, monkeypatch) -> None:
    """Layout fitting preserves the original EDBO prior, likelihood, and posterior."""
    from baybe.multistep._configuration_space import _ConfigurationSpace
    from baybe.multistep._layout_fit import _fit_layout_surrogate
    from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
    from baybe.utils.dataframe import to_tensor

    parameters = _make_parameters()
    frame = _make_frame()
    objective = NumericalTarget("y").to_objective()
    legacy_space = SearchSpace.from_product(parameters)
    layout = _ConfigurationSpace.from_parameters(parameters)

    legacy = GaussianProcessSurrogate.from_preset(preset)
    free = GaussianProcessSurrogate.from_preset(preset)
    _seed()
    legacy.fit(legacy_space, objective, frame)

    _forbid_product_builders(monkeypatch)
    _seed()
    _fit_layout_surrogate(free, layout, objective, frame)

    assert free._searchspace is layout

    candidates = frame.drop(columns="y")
    expected = legacy._posterior_comp(
        to_tensor(legacy_space.transform(candidates)).unsqueeze(-2)
    )
    actual = free._posterior_comp(to_tensor(layout.transform(candidates)).unsqueeze(-2))

    import torch

    torch.testing.assert_close(actual.mean, expected.mean, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(actual.variance, expected.variance, atol=1e-8, rtol=1e-6)
    expected_state = legacy._model.state_dict()
    actual_state = free._model.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for key, value in expected_state.items():
        torch.testing.assert_close(actual_state[key], value, atol=1e-8, rtol=1e-6)


def test_layout_fit_reuses_and_invalidates_its_cache() -> None:
    """The layout fit keeps the regular cache key and invalidates on new data."""
    from joblib.hashing import hash

    from baybe.multistep._configuration_space import _ConfigurationSpace
    from baybe.multistep._layout_fit import _fit_layout_surrogate
    from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
    from baybe.utils.dataframe import to_tensor

    parameters = _make_parameters()
    frame = _make_frame()
    objective = NumericalTarget("y").to_objective()
    layout = _ConfigurationSpace.from_parameters(parameters)
    surrogate = GaussianProcessSurrogate()

    _seed()
    _fit_layout_surrogate(surrogate, layout, objective, frame)
    fitted_model = surrogate._model
    assert fitted_model is not None

    # An equivalent layout, the same objective, and the same data are a cache hit.
    _fit_layout_surrogate(
        surrogate, _ConfigurationSpace.from_parameters(parameters), objective, frame
    )
    assert surrogate._model is fitted_model

    # Experimental-representation posteriors use the retained layout context.
    candidates = frame.drop(columns="y")
    raw_posterior = surrogate.posterior(candidates)
    comp_posterior = surrogate._posterior_comp(
        to_tensor(layout.transform(candidates)).unsqueeze(-2)
    )
    import torch

    torch.testing.assert_close(
        raw_posterior.mean, comp_posterior.mean.squeeze(-2), atol=1e-8, rtol=1e-6
    )

    # Changed data invalidates the cache and refits the model.
    changed = frame.assign(y=frame["y"] + 1.0)
    _fit_layout_surrogate(surrogate, layout, objective, changed)
    assert surrogate._model is not fitted_model
    assert surrogate._measurements_hash == hash(changed)


def test_external_factory_is_rejected_before_publication() -> None:
    """A factory without verified layout compatibility is rejected before fitting."""
    from baybe.multistep._configuration_space import _ConfigurationSpace
    from baybe.multistep._layout_fit import _fit_layout_surrogate
    from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
    from baybe.surrogates.gaussian_process.presets.edbo import EDBOKernelFactory

    class _ExternalKernelFactory(EDBOKernelFactory):
        """A look-alike factory outside the verified first-party allowlist."""

    parameters = _make_parameters()
    frame = _make_frame()
    objective = NumericalTarget("y").to_objective()
    layout = _ConfigurationSpace.from_parameters(parameters)
    surrogate = GaussianProcessSurrogate(kernel_or_factory=_ExternalKernelFactory())

    with pytest.raises(IncompatibilityError, match="not verified"):
        _fit_layout_surrogate(surrogate, layout, objective, frame)

    assert surrogate._searchspace is None
    assert surrogate._model is None
