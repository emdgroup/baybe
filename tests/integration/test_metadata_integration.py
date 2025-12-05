"""Integration tests for metadata with BayBE components."""

import pytest
from pytest import param

from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.targets.enum import TargetMode
from baybe.targets.numerical import MatchMode, NumericalTarget
from baybe.transformations import ExponentialTransformation
from baybe.utils.metadata import MeasurableMetadata, Metadata, to_metadata

TMetadata = Metadata | dict | None


def make_parameter(metadata: TMetadata = None) -> NumericalDiscreteParameter:
    return NumericalDiscreteParameter(
        "p", (1, 2), metadata=metadata or MeasurableMetadata()
    )


def make_target(metadata: TMetadata = None) -> NumericalTarget:
    return NumericalTarget(
        "yield", TargetMode.MAX, metadata=metadata or MeasurableMetadata()
    )


def make_objective(metadata: TMetadata = None) -> SingleTargetObjective:
    return SingleTargetObjective(
        target=make_target(), metadata=metadata or MeasurableMetadata()
    )


@pytest.mark.parametrize(
    ("constructor", "metadata_cls"),
    [
        param(make_parameter, MeasurableMetadata, id="parameter"),
        param(make_target, MeasurableMetadata, id="target"),
        param(make_objective, Metadata, id="objective"),
    ],
)
class TestMetadataIntegration:
    """Tests for metadata integration with BayBE objects."""

    @pytest.mark.parametrize("as_dict", [True, False])
    def test_with_metadata(self, constructor, metadata_cls, as_dict: bool):
        """BayBE objects accept, ingest, and surface metadata."""
        meta = dict(description="test", unit="m", other="value")
        container = constructor(
            metadata=meta if as_dict else to_metadata(meta, metadata_cls)
        )

        assert container.description == "test"
        if metadata_cls is MeasurableMetadata:
            assert container.metadata.misc == {"other": "value"}
            assert container.unit == "m"
        else:
            assert container.metadata.misc == {"unit": "m", "other": "value"}

    def test_without_metadata(self, metadata_cls, constructor):
        """BayBE objects without metadata have empty metadata and `None` properties."""
        container = constructor().metadata

        assert container.is_empty
        assert container.description is None
        if metadata_cls is MeasurableMetadata:
            assert container.unit is None


@pytest.mark.parametrize(
    ("constructor", "kwargs"),
    [
        param("__init__", {}, id="init1"),
        param("__init__", {"minimize": True}, id="init2"),
        param("__init__", {"transformation": ExponentialTransformation()}, id="init3"),
        param("match_absolute", {}, id="match_abs1"),
        param(
            "match_absolute",
            {"mismatch_instead": True, "match_mode": MatchMode.LE},
            id="match_abs2",
        ),
        param("match_quadratic", {}, id="match_quad1"),
        param(
            "match_quadratic",
            {"mismatch_instead": True, "match_mode": MatchMode.LE},
            id="match_quad2",
        ),
        param("match_power", {"exponent": 5}, id="match_pow1"),
        param(
            "match_power",
            {"exponent": 5, "mismatch_instead": True, "match_mode": MatchMode.LE},
            id="match_pow2",
        ),
        param("match_triangular", {"cutoffs": [100, 10000]}, id="match_tri1"),
        param("match_triangular", {"width": 10}, id="match_tri2"),
        param(
            "match_triangular",
            {"margins": [4, 9], "match_mode": MatchMode.LE},
            id="match_tri3",
        ),
        param("match_bell", {"sigma": 4.2}, id="match_bell1"),
        param(
            "match_bell",
            {"sigma": 4.2, "mismatch_instead": True, "match_mode": MatchMode.LE},
            id="match_bell2",
        ),
        param("normalized_ramp", {"cutoffs": [100, 10000]}, id="ramp"),
        param(
            "normalized_sigmoid", {"anchors": [[100, 0.1], [10000, 0.9]]}, id="sigmoid"
        ),
    ],
)
def test_constructor_history(constructor, kwargs):
    """The constructor metadata allows to reconstruct the object."""
    if constructor.startswith("match_"):
        kwargs["match_value"] = 1337

    if constructor == "__init__":
        t1 = NumericalTarget("t", **kwargs)
    else:
        t1 = getattr(NumericalTarget, constructor)("t", **kwargs)
    history = t1.constructor_history

    meta_constructor = history.pop("constructor")
    meta_kwargs = history
    if meta_constructor == "__init__":
        t2 = NumericalTarget(**meta_kwargs)
    else:
        t2 = getattr(NumericalTarget, meta_constructor)(**meta_kwargs)

    assert t1 == t2
