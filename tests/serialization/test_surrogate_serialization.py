"""Surrogate serialization tests."""

import pytest

from baybe._optional.info import NGBOOST_INSTALLED, ONNX_INSTALLED
from baybe.surrogates.base import Surrogate
from baybe.surrogates.composite import CompositeSurrogate
from baybe.surrogates.custom import CustomONNXSurrogate
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.ngboost import NGBoostSurrogate
from baybe.surrogates.random_forest import RandomForestSurrogate
from baybe.utils.basic import get_subclasses
from tests.serialization.utils import assert_roundtrip_consistency


@pytest.mark.parametrize("surrogate_cls", get_subclasses(Surrogate))
def test_surrogate_roundtrip(request, surrogate_cls: type[Surrogate]):
    """A serialization roundtrip yields an equivalent object."""
    if issubclass(surrogate_cls, CustomONNXSurrogate):
        if not ONNX_INSTALLED:
            pytest.skip("Optional onnx dependency not installed.")
        surrogate = request.getfixturevalue("onnx_surrogate")
    elif issubclass(surrogate_cls, NGBoostSurrogate) and not NGBOOST_INSTALLED:
        pytest.skip("Optional ngboost dependency not installed.")
    else:
        surrogate = surrogate_cls()

    assert_roundtrip_consistency(surrogate)


@pytest.mark.parametrize(
    "surrogate",
    [
        CompositeSurrogate(
            {"A": RandomForestSurrogate(), "B": GaussianProcessSurrogate()}
        ),
        CompositeSurrogate.from_replication(GaussianProcessSurrogate()),
    ],
    ids=["via_init", "via_template"],
)
def test_composite_surrogate_roundtrip(surrogate: CompositeSurrogate):
    """A serialization roundtrip yields an equivalent object."""
    assert_roundtrip_consistency(surrogate)
