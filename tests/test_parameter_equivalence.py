"""Tests for parameter equivalence."""

import pytest
from gpytorch import kernels as gk
from gpytorch.constraints import Interval
from gpytorch.priors import GammaPrior as GPyTorchGammaPrior
from pytest import param

from baybe.kernels import MaternKernel, RBFKernel
from baybe.kernels.composite import ScaleKernel
from baybe.parameters import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.parameters.enum import TransferLearningMode
from baybe.priors import GammaPrior

_INDEX = TransferLearningMode.INDEX_KERNEL
_POSITIVE_INDEX = TransferLearningMode.POSITIVE_INDEX_KERNEL


def _rbf(lengthscale, *, frozen=False):
    """Create a GPyTorch RBF kernel with a given, optionally frozen, lengthscale."""
    kernel = gk.RBFKernel()
    kernel.lengthscale = lengthscale
    kernel.raw_lengthscale.requires_grad_(not frozen)
    return kernel


def _pair(override, other_override, other_bounds=(0, 1)):
    """Create two differently named continuous parameters with given overrides."""
    return (
        NumericalContinuousParameter("p1", (0, 1), override_kernel=override),
        NumericalContinuousParameter(
            "p2", other_bounds, override_kernel=other_override
        ),
    )


def _task_pair(values, mode, other_values, other_mode):
    """Create two differently named task parameters."""
    return (
        TaskParameter("t1", values, override_transfer_learning_mode=mode),
        TaskParameter("t2", other_values, override_transfer_learning_mode=other_mode),
    )


_SAME_GPYTORCH_KERNEL = gk.RBFKernel()


@pytest.mark.parametrize(
    ("p1", "p2", "expected"),
    [
        # Equivalent: only the name differs
        param(*_pair(None, None), True, id="no-override"),
        param(*_pair(RBFKernel(), RBFKernel()), True, id="baybe"),
        param(
            *_pair(RBFKernel(), RBFKernel(parameter_names=("p2",))),
            True,
            id="baybe-unscoped-vs-scoped",
        ),
        param(
            *_pair(
                ScaleKernel(RBFKernel() + MaternKernel()),
                ScaleKernel(RBFKernel() + MaternKernel()),
            ),
            True,
            id="baybe-composite",
        ),
        param(*_pair(gk.MaternKernel(), gk.MaternKernel()), True, id="gpytorch"),
        param(
            *_pair(
                gk.ScaleKernel(
                    gk.RBFKernel(lengthscale_prior=GPyTorchGammaPrior(3.0, 6.0))
                ),
                gk.ScaleKernel(
                    gk.RBFKernel(lengthscale_prior=GPyTorchGammaPrior(3.0, 6.0))
                ),
            ),
            True,
            id="gpytorch-nested",
        ),
        param(
            *_pair(gk.IndexKernel(num_tasks=2), gk.IndexKernel(num_tasks=2)),
            True,
            id="gpytorch-random-init",
        ),
        param(*_pair(_rbf(2.0), _rbf(3.0)), True, id="gpytorch-initial-value"),
        param(
            *_pair(_rbf(2.0, frozen=True), _rbf(2.0, frozen=True)),
            True,
            id="gpytorch-same-frozen-value",
        ),
        param(
            *_pair(_SAME_GPYTORCH_KERNEL, _SAME_GPYTORCH_KERNEL),
            True,
            id="gpytorch-same-instance",
        ),
        param(
            *_task_pair(("a", "b"), _INDEX, ("a", "b"), _INDEX),
            True,
            id="task-same-mode",
        ),
        param(*_task_pair(("a", "b"), None, ("a", "b"), None), True, id="task-no-mode"),
        # Not equivalent: specification differs
        param(
            NumericalContinuousParameter("p1", (0, 1), override_kernel=RBFKernel()),
            NumericalDiscreteParameter("p2", (0, 1), override_kernel=RBFKernel()),
            False,
            id="parameter-type",
        ),
        param(
            *_pair(RBFKernel(), RBFKernel(), other_bounds=(0, 2)),
            False,
            id="parameter-spec",
        ),
        param(*_pair(RBFKernel(), None), False, id="override-vs-none"),
        param(*_pair(RBFKernel(), MaternKernel()), False, id="baybe-class"),
        param(
            *_pair(RBFKernel(), RBFKernel(lengthscale_prior=GammaPrior(3.0, 6.0))),
            False,
            id="baybe-setting",
        ),
        param(*_pair(RBFKernel(), gk.RBFKernel()), False, id="baybe-vs-gpytorch"),
        param(*_pair(gk.RBFKernel(), gk.MaternKernel()), False, id="gpytorch-class"),
        param(
            *_pair(gk.MaternKernel(nu=0.5), gk.MaternKernel()),
            False,
            id="gpytorch-attribute",
        ),
        param(
            *_pair(
                gk.RBFKernel(lengthscale_prior=GPyTorchGammaPrior(3.0, 6.0)),
                gk.RBFKernel(lengthscale_prior=GPyTorchGammaPrior(2.0, 6.0)),
            ),
            False,
            id="gpytorch-prior",
        ),
        param(
            *_pair(
                gk.RBFKernel(lengthscale_constraint=Interval(0.1, 2.0)),
                gk.RBFKernel(lengthscale_constraint=Interval(0.1, 3.0)),
            ),
            False,
            id="gpytorch-constraint",
        ),
        param(
            *_pair(_rbf(2.0, frozen=True), _rbf(3.0, frozen=True)),
            False,
            id="gpytorch-frozen-value",
        ),
        param(
            *_pair(_rbf(2.0, frozen=True), _rbf(2.0)),
            False,
            id="gpytorch-frozen-vs-learnable",
        ),
        param(
            *_pair(
                gk.IndexKernel(num_tasks=2, rank=1), gk.IndexKernel(num_tasks=2, rank=2)
            ),
            False,
            id="gpytorch-shape",
        ),
        param(
            *_pair(
                gk.RBFKernel() + gk.MaternKernel(), gk.MaternKernel() + gk.RBFKernel()
            ),
            False,
            id="gpytorch-summand-order",
        ),
        param(
            *_task_pair(("a", "b"), _INDEX, ("a", "b"), _POSITIVE_INDEX),
            False,
            id="task-mode",
        ),
        param(
            *_task_pair(("a", "b"), None, ("a", "b", "c"), None),
            False,
            id="task-values",
        ),
        param(
            *_task_pair(("a", "b"), _INDEX, ("a", "b"), None),
            False,
            id="task-mode-vs-none",
        ),
    ],
)
def test_parameter_equivalence(p1, p2, expected):
    """Parameter equivalence with kernel overrides produces the expected result."""
    assert p1.is_equivalent(p2) == expected
    assert p2.is_equivalent(p1) == expected
