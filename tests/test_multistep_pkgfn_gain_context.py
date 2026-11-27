"""Tests for the shared finite-reference pKGFN gain context."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from baybe.exceptions import IncompatibilityError
from baybe.multistep import acquisition as multistep_acquisition
from baybe.multistep.acquisition import PartialKnowledgeGradientFN
from baybe.multistep.kgfn import gain as kgfn_gain
from baybe.multistep.kgfn.gain import _make_gain_context


class _FakeNodeModel:
    def __init__(self, batch_shape: torch.Size = torch.Size()) -> None:
        self.batch_shape = batch_shape

    def condition_on_observations(self, *args, **kwargs):
        return self


class _FakeGainModel:
    """Minimal model surface used by the gain context."""

    def __init__(
        self, *, offset: float, batch_shape: torch.Size = torch.Size()
    ) -> None:
        self._offset = offset
        self._node_model = _FakeNodeModel(batch_shape)
        self.dag = SimpleNamespace(
            topological_order=("leaf",),
            comp_rep_columns=("x",),
        )

    def parameters(self):
        return iter((torch.zeros(1, dtype=torch.double),))

    def get_node_model(self, name: str):
        assert name == "leaf"
        return self._node_model

    def condition_on_pending(self, pending, *, num_fantasies, sampling_seed):
        assert not pending
        return (self,)

    def make_node_fantasies(self, node, inputs, num_fantasies, sampling_seed):
        assert node.name == "leaf"
        batch_shape = torch.Size((num_fantasies, *inputs.shape[:-2]))
        return _FakeGainModel(offset=5.0, batch_shape=batch_shape)


def test_gain_context_uses_explicit_references_once(monkeypatch):
    """The context evaluates baseline and fantasy on the same explicit references."""
    references = torch.arange(3, dtype=torch.double).reshape(-1, 1, 1)
    seen: list[int] = []

    def fake_means(model, points, *, num_network_samples, sampling_seed):
        seen.append(points.shape[0])
        return points[..., 0, 0] + model._offset

    monkeypatch.setattr(
        kgfn_gain, "_terminal_means_from_model", fake_means
    )
    context = _make_gain_context(
        _FakeGainModel(offset=0.0),
        (),
        references,
        num_fantasies=2,
        num_network_samples=1,
        sampling_seed=0,
    )
    gain = context.evaluate(
        SimpleNamespace(name="leaf"),
        torch.zeros((1, 1, 1), dtype=torch.double),
    )

    assert gain.item() == pytest.approx(5.0)
    assert context.num_branches == 1
    assert context.num_references == 3
    assert seen == [3, 3]


def test_gain_context_rejects_nonfinite_or_misshaped_references():
    """Bad references are local errors, never silent sentinels."""
    model = _FakeGainModel(offset=0.0)
    with pytest.raises(IncompatibilityError, match="finite"):
        _make_gain_context(
            model,
            (),
            torch.tensor([[[float("nan")]]], dtype=torch.double),
            num_fantasies=1,
            num_network_samples=1,
            sampling_seed=0,
        )
    with pytest.raises(IncompatibilityError, match="width|shape"):
        _make_gain_context(
            model,
            (),
            torch.zeros((2, 1, 3), dtype=torch.double),
            num_fantasies=1,
            num_network_samples=1,
            sampling_seed=0,
        )


def test_legacy_make_gain_function_delegates_to_the_context(monkeypatch):
    """The legacy wrapper keeps its signature and calls the shared evaluator."""
    calls = []

    class _Context:
        evaluate = staticmethod(
            lambda node, inputs: calls.append((node.name, tuple(inputs.shape)))
            or torch.ones(1, dtype=torch.double)
        )
        num_branches = 1
        num_references = 1

    monkeypatch.setattr(
        multistep_acquisition,
        "_make_gain_context",
        lambda *args, **kwargs: _Context(),
    )
    monkeypatch.setattr(
        PartialKnowledgeGradientFN,
        "_make_terminal_reference_tensor",
        lambda self, dag, reference: torch.zeros((1, 1, 1), dtype=torch.double),
    )
    model = _FakeGainModel(offset=0.0)
    gain = PartialKnowledgeGradientFN().make_gain_function(model)
    value = gain(None, SimpleNamespace(name="leaf"), torch.zeros((1, 1, 1)))

    assert value.item() == 1.0
    assert calls == [("leaf", (1, 1, 1))]
