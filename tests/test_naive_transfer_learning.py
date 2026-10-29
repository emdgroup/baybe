"""Tests for the naive transfer learning mode, where the task kernel vanishes."""

from functools import partial

import pytest
from pytest import param
from torch.testing import assert_close

from baybe.kernels import RBFKernel
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.presets.hvarfner import (
    HvarfnerLikelihoodFactory,
    HvarfnerMeanFactory,
)
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import create_fake_input

_TARGET = NumericalTarget("y")
_TASK = TaskParameter(
    "task",
    ("A", "B", "C"),
    active_values=("A",),
    override_transfer_learning_mode="NAIVE",
)
_X1 = NumericalContinuousParameter("x1", (0, 1))
_X2 = NumericalDiscreteParameter("x2", (0, 1, 2))
_X3 = CategoricalParameter("x3", ("a", "b", "c"))


@pytest.mark.parametrize(
    "tasks",
    [
        param(("A", "B"), id="both"),
        param(("A",), id="target-only"),
        param(("B",), id="source-only"),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        param([_X1], id="continuous"),
        param([_X2, _X3], id="discrete"),
        param([_X1, _X2, _X3], id="hybrid"),
        param(
            [
                _X1,
                NumericalContinuousParameter("o", (0, 1), override_kernel=RBFKernel()),
            ],
            id="override",
        ),
    ],
)
@pytest.mark.parametrize(
    "make_surrogate",
    [
        param(partial(GaussianProcessSurrogate.from_preset, p), id=p)
        for p in ("BAYBE", "CHEN", "EDBO", "EDBO_SMOOTHED")
    ]
    + [
        param(
            partial(
                GaussianProcessSurrogate,
                mean_or_factory=HvarfnerMeanFactory(),
                likelihood_or_factory=HvarfnerLikelihoodFactory(),
            ),
            id="hvarfner-components",
        )
    ],
)
def test_naive_mode_matches_task_free_gp(make_surrogate, parameters, tasks):
    """The naive mode yields the same GP as dropping the task parameter."""
    measurements = create_fake_input(parameters, [_TARGET], n_rows=6)
    measurements["task"] = [tasks[i % len(tasks)] for i in range(len(measurements))]
    objective = _TARGET.to_objective()

    searchspace_naive = SearchSpace.from_product([*parameters, _TASK])
    searchspace_free = SearchSpace.from_product(parameters)
    naive = make_surrogate()
    naive.fit(searchspace_naive, objective, measurements)
    task_free = make_surrogate()
    task_free.fit(searchspace_free, objective, measurements.drop(columns="task"))

    # Same fitted state, where kernel dimensions are compared via their column names
    # since the task column shifts the indices
    state_naive = naive.to_botorch().state_dict()
    state_free = task_free.to_botorch().state_dict()
    columns_naive = searchspace_naive.comp_rep_columns
    columns_free = searchspace_free.comp_rep_columns
    assert state_naive.keys() == state_free.keys()
    for key, value in state_free.items():
        if key.startswith("input_transform"):
            continue
        if key.endswith("active_dims"):
            assert [columns_naive[i] for i in state_naive[key]] == [
                columns_free[i] for i in value
            ]
        else:
            assert_close(state_naive[key], value)

    # Same predictions, irrespective of the task label
    candidates = create_fake_input(parameters, [_TARGET], n_rows=4).drop(columns="y")
    posterior_free = task_free.posterior(candidates)
    for task in _TASK.values:
        posterior_naive = naive.posterior(candidates.assign(task=task))
        assert_close(posterior_naive.mean, posterior_free.mean)
        assert_close(
            posterior_naive.distribution.covariance_matrix,
            posterior_free.distribution.covariance_matrix,
        )
