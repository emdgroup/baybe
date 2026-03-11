"""Kernel tests."""

from typing import Any

import numpy as np
import pytest
import torch
from attrs import asdict, has
from hypothesis import given
from pytest import param

from baybe.kernels.base import BasicKernel, Kernel
from baybe.kernels.basic import IndexKernel, MaternKernel, RBFKernel
from baybe.kernels.composite import ProductKernel, ScaleKernel, SumKernel
from tests.hypothesis_strategies.kernels import kernels

# TODO: Consider deprecating these attribute names to avoid inconsistencies
_RENAME_DICT: dict[str, str] = {
    "base_kernels": "kernels",
}
"""Dictionary for resolving name differences between BayBE and GPyTorch attributes."""


def validate_gpytorch_kernel_components(obj: Any, mapped: Any, **kwargs) -> None:
    """Validate that all kernel components are correctly translated to GPyTorch.

    Args:
        obj: An object occurring as part of a BayBE kernel.
        mapped: The corresponding object in the translated GPyTorch kernel.
        **kwargs: Optional kernel arguments that were passed to the GPyTorch kernel.
    """
    # Assert that the kernel kwargs are correctly mapped
    if isinstance(obj, BasicKernel):
        for k, v in kwargs.items():
            assert torch.tensor(getattr(mapped, k)).equal(torch.tensor(v))

    # Compare attribute by attribute
    for name, component in asdict(obj, recurse=False).items():
        # If the BayBE component is `None`, the GPyTorch component might not even
        # exist, so we skip
        if component is None:
            continue

        # Resolve attribute naming differences
        mapped_name = _RENAME_DICT.get(name, name)

        # If the attribute does not exist in the GPyTorch version, ...
        if (mapped_component := getattr(mapped, mapped_name, None)) is None:
            # ... it must have some special handling on the GPyTorch side
            # >>>>>
            # TODO: this will be refactored in #748, which requires changes to the
            #   test logic anyways
            if isinstance(obj, IndexKernel) and name == "num_tasks":
                # Special case for IndexKernel.num_tasks, which is not an actual
                # attribute of the GPyTorch kernel
                assert mapped.covar_factor.shape[-2] == component
                continue
            elif isinstance(obj, IndexKernel) and name == "rank":
                # Special case for IndexKernel.rank, which is not an actual attribute of
                # the GPyTorch kernel
                assert mapped.covar_factor.shape[-1] == component
                continue
            # <<<<<

            # ... or it must be a trainability flag, which is a BayBE-only concept
            # applied after GPyTorch kernel construction.
            if name.endswith("_trainable"):
                continue

            # ... or it must be an initial value. Because setting initial values
            # involves going through constraint transformations on GPyTorch side (i.e.,
            # difference between `<attr>` and `raw_<attr>`), the numerical values will
            # not be exact, so we check only for approximate matches.
            assert name.endswith("_initial_value")
            assert np.allclose(
                component,
                getattr(mapped, name.removesuffix("_initial_value")).detach().numpy(),
            )

        # If the component is itself another attrs object, recurse
        elif has(component):
            validate_gpytorch_kernel_components(component, mapped_component, **kwargs)

        # Same for collections of BayBE objects (coming from composite kernels)
        elif isinstance(component, tuple) and all(has(c) for c in component):
            for c, m in zip(component, mapped_component):
                validate_gpytorch_kernel_components(c, m, **kwargs)

        # On the lowest component level, simply check for equality
        else:
            assert component == mapped_component


@given(kernels())
def test_kernel_assembly(kernel: Kernel):
    """Turning a BayBE kernel into a GPyTorch kernel raises no errors and all its
    components are translated correctly."""  # noqa
    # Create some arbitrary kernel kwargs to ensure that they are correctly translated
    kwargs = dict(
        ard_num_dims=np.random.randint(0, 32),
        batch_shape=torch.Size(
            [np.random.randint(0, 4) for _ in range(np.random.randint(0, 4))]
        ),
        active_dims=[np.random.randint(0, 4) for _ in range(np.random.randint(0, 4))],
    )

    k = kernel.to_gpytorch(**kwargs)
    validate_gpytorch_kernel_components(kernel, k, **kwargs)


# --- Operator tests ---


def test_add_produces_sum_kernel():
    """Adding two kernels produces a SumKernel."""
    result = MaternKernel() + RBFKernel()
    assert isinstance(result, SumKernel)
    assert len(result.base_kernels) == 2
    assert isinstance(result.base_kernels[0], MaternKernel)
    assert isinstance(result.base_kernels[1], RBFKernel)


def test_add_chain_flattens():
    """Chaining additions flattens into a single SumKernel."""
    result = MaternKernel() + RBFKernel() + MaternKernel(0.5)
    assert isinstance(result, SumKernel)
    assert len(result.base_kernels) == 3


def test_mul_produces_product_kernel():
    """Multiplying two kernels produces a ProductKernel."""
    result = MaternKernel() * RBFKernel()
    assert isinstance(result, ProductKernel)
    assert len(result.base_kernels) == 2


def test_mul_chain_flattens():
    """Chaining multiplications flattens into a single ProductKernel."""
    result = MaternKernel() * RBFKernel() * MaternKernel(0.5)
    assert isinstance(result, ProductKernel)
    assert len(result.base_kernels) == 3


@pytest.mark.parametrize(
    ("left", "right"),
    [
        param(MaternKernel(), 3.0, id="kernel_times_float"),
        param(MaternKernel(), 5, id="kernel_times_int"),
        param(3.0, MaternKernel(), id="float_times_kernel"),
    ],
)
def test_mul_constant_produces_scale_kernel(left, right):
    """Multiplying a kernel with a numeric constant produces a fixed ScaleKernel."""
    result = left * right
    assert isinstance(result, ScaleKernel)
    assert result.outputscale_trainable is False


def test_mul_constant_sets_initial_value():
    """The constant value is stored as the outputscale initial value."""
    result = MaternKernel() * 3.0
    assert result.outputscale_initial_value == 3.0


def test_scale_kernel_freezes_outputscale():
    """A ScaleKernel from constant multiplication freezes the outputscale."""
    gpytorch_kernel = (2.0 * RBFKernel()).to_gpytorch()
    assert not gpytorch_kernel.raw_outputscale.requires_grad


def test_scale_kernel_trainable_by_default():
    """An explicitly constructed ScaleKernel has a trainable outputscale."""
    gpytorch_kernel = ScaleKernel(
        base_kernel=RBFKernel(), outputscale_initial_value=2.0
    ).to_gpytorch()
    assert gpytorch_kernel.raw_outputscale.requires_grad


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        param(lambda: MaternKernel() + "string", TypeError, id="add_string"),
        param(lambda: MaternKernel() * "string", TypeError, id="mul_string"),
        param(lambda: 1 + MaternKernel(), TypeError, id="radd_int"),
    ],
)
def test_operator_unsupported_type(expression, error):
    """Using operators with unsupported types raises TypeError."""
    with pytest.raises(error):
        expression()


def test_radd_kernel():
    """Right-hand addition delegates to __add__ for kernel operands."""
    result = MaternKernel().__radd__(RBFKernel())
    assert isinstance(result, SumKernel)


@pytest.mark.parametrize(
    "source",
    [
        param("baybe.kernels.deprecation", id="direct"),
        param("baybe.kernels", id="top_level"),
    ],
)
def test_additive_kernel_deprecation(source):
    """Using AdditiveKernel emits a DeprecationWarning and returns a SumKernel."""
    import importlib

    module = importlib.import_module(source)
    AdditiveKernel = getattr(module, "AdditiveKernel")
    with pytest.warns(DeprecationWarning, match="AdditiveKernel"):
        k = AdditiveKernel([MaternKernel(), RBFKernel()])
    assert isinstance(k, SumKernel)


@pytest.mark.parametrize(
    "kernel",
    [
        param(MaternKernel() + RBFKernel(), id="sum"),
        param(MaternKernel() * RBFKernel(), id="product"),
        param(3.0 * MaternKernel(), id="scale"),
    ],
)
def test_operator_kernel_serialization_roundtrip(kernel):
    """Kernels created via operators survive a serialization roundtrip."""
    restored = Kernel.from_dict(kernel.to_dict())
    assert kernel == restored
