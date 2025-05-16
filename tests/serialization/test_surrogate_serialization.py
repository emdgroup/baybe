"""Test serialization of surrogates."""

import pytest

from baybe._optional.info import NGBOOST_INSTALLED, ONNX_INSTALLED
from baybe.surrogates.base import Surrogate
from baybe.surrogates.composite import CompositeSurrogate
from baybe.surrogates.custom import CustomONNXSurrogate
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.ngboost import NGBoostSurrogate
from baybe.surrogates.random_forest import RandomForestSurrogate
from baybe.utils.basic import get_subclasses

from ..conftest import _onnx_issue


@pytest.mark.parametrize(
    "surrogate_cls",
    [
        pytest.param(
            s,
            marks=[pytest.mark.xfail(reason=_onnx_issue, strict=True)]
            if issubclass(s, CustomONNXSurrogate)
            else [],
        )
        for s in get_subclasses(Surrogate)
    ],
)
def test_surrogate_serialization(request, surrogate_cls):
    """A serialization roundtrip yields an equivalent object."""
    if issubclass(surrogate_cls, CustomONNXSurrogate):
        if not ONNX_INSTALLED:
            pytest.skip("Optional onnx dependency not installed.")
        surrogate = request.getfixturevalue("onnx_surrogate")
    elif issubclass(surrogate_cls, NGBoostSurrogate) and not NGBOOST_INSTALLED:
        pytest.skip("Optional ngboost dependency not installed.")
    else:
        surrogate = surrogate_cls()

    string = surrogate.to_json()
    surrogate2 = Surrogate.from_json(string)
    assert surrogate == surrogate2, (surrogate, surrogate2)


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
def test_composite_surrogate_serialization(surrogate):
    """A serialization roundtrip yields an equivalent object."""
    string = surrogate.to_json()
    surrogate2 = CompositeSurrogate.from_json(string)
    assert surrogate == surrogate2, (surrogate, surrogate2)
