"""Tests for composite optimizers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from pytest import param

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.optimizers.composite import (
    BlockCoordinateOptimizer,
    CyclicOptimizationSchedule,
    OptimizationStep,
)
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace

_PC1 = NumericalContinuousParameter("x1", bounds=(0, 1))
_PC2 = NumericalContinuousParameter("x2", bounds=(-1, 0))

_SS = SearchSpace.from_product([_PC1, _PC2])
_N_COLS = len(_SS.comp_rep_columns)

_STEP1 = OptimizationStep(selector="x1", optimizer=MagicMock())
_STEP2 = OptimizationStep(selector="x2", optimizer=MagicMock())


def _make_inner_optimizer(n_cols: int = _N_COLS) -> MagicMock:
    """Return a mock inner optimizer whose return value has the expected shapes."""
    mock = MagicMock()
    mock.return_value = (torch.zeros(1, n_cols), torch.zeros(1))
    return mock


def _make_bco(
    selector: str = ".*",
    inner_optimizer: MagicMock | None = None,
    n_cycles: int = 1,
) -> BlockCoordinateOptimizer:
    """Construct a :class:`BlockCoordinateOptimizer` with a single step."""
    if inner_optimizer is None:
        inner_optimizer = _make_inner_optimizer()
    step = OptimizationStep(selector=selector, optimizer=inner_optimizer)
    schedule = CyclicOptimizationSchedule(steps=(step,), n_cycles=n_cycles)
    return BlockCoordinateOptimizer(schedule=schedule)


@pytest.fixture(name="mock_score_fn")
def fixture_mock_score_fn() -> MagicMock:
    """A mock acquisition function with no pending experiments."""
    mock = MagicMock()
    mock.X_pending = None
    mock.set_X_pending = MagicMock()
    return mock


class TestCyclicOptimizationSchedule:
    """Tests for ``CyclicOptimizationSchedule``."""

    @pytest.mark.parametrize(
        ("steps", "n_cycles", "expected"),
        [
            param((_STEP1,), 1, [_STEP1], id="1step_1cycle"),
            param((_STEP1,), 3, [_STEP1, _STEP1, _STEP1], id="1step_3cycles"),
            param((_STEP1, _STEP2), 1, [_STEP1, _STEP2], id="2steps_1cycle"),
            param(
                (_STEP1, _STEP2),
                2,
                [_STEP1, _STEP2, _STEP1, _STEP2],
                id="2steps_2cycles",
            ),
        ],
    )
    def test_cyclic_iteration_order(self, steps, n_cycles, expected):
        """Steps are yielded in round-robin order for the given number of cycles."""
        schedule = CyclicOptimizationSchedule(steps=steps, n_cycles=n_cycles)
        assert list(schedule(_SS)) == expected

    def test_skipped_step_warns_and_is_excluded(self):
        """A step matching no parameters emits *one* warning and is dropped."""
        step_active = OptimizationStep(selector=".*", optimizer=MagicMock())
        step_skipped = OptimizationStep(selector="nomatch", optimizer=MagicMock())
        schedule = CyclicOptimizationSchedule(
            steps=(step_active, step_skipped), n_cycles=2
        )

        with pytest.warns(UserWarning, match="will be skipped") as record:
            yielded = list(schedule(_SS))

        assert len(record) == 1
        assert yielded == [step_active, step_active]

    def test_all_steps_skipped_raises(self):
        """When no step matches any parameter an error is raised."""
        step = OptimizationStep(selector="nomatch", optimizer=MagicMock())
        schedule = CyclicOptimizationSchedule(steps=(step,), n_cycles=1)

        with pytest.raises(
            IncompatibleSearchSpaceError, match="none of the specified steps"
        ):
            list(schedule(_SS))


class TestBlockCoordinateOptimizer:
    """Tests for ``BlockCoordinateOptimizer``."""

    @pytest.mark.parametrize(
        ("n_cycles", "n_steps"),
        [
            param(1, 1, id="one_step_one_cycle"),
            param(3, 1, id="one_step_three_cycles"),
            param(1, 2, id="two_steps_one_cycle"),
            param(3, 2, id="two_steps_three_cycles"),
        ],
    )
    def test_inner_optimizer_call_count(self, n_cycles, n_steps, mock_score_fn):
        """Each inner optimizer is called once per active cycle."""
        inners = [_make_inner_optimizer() for _ in range(n_steps)]
        selectors = ["x1", "x2"][:n_steps]
        steps = tuple(
            OptimizationStep(selector=s, optimizer=i) for s, i in zip(selectors, inners)
        )
        schedule = CyclicOptimizationSchedule(steps=steps, n_cycles=n_cycles)
        bco = BlockCoordinateOptimizer(schedule=schedule)

        bco(1, mock_score_fn, _SS)

        for inner in inners:
            assert inner.call_count == n_cycles

    def test_inner_optimizer_receives_batch_size_one(self, mock_score_fn):
        """The inner optimizer is always invoked with ``batch_size=1``."""
        inner = _make_inner_optimizer()
        bco = _make_bco(inner_optimizer=inner)
        bco(1, mock_score_fn, _SS)
        assert inner.call_args[0][0] == 1

    @pytest.mark.parametrize(
        "batch_size", [param(1, id="single"), param(3, id="batch")]
    )
    def test_return_value_shape(self, batch_size, mock_score_fn):
        """Output shapes are ``(batch_size, n_comp_cols)`` and ``(batch_size,)``."""
        bco = _make_bco()
        pts, scores = bco(batch_size, mock_score_fn, _SS)
        assert tuple(pts.shape) == (batch_size, _N_COLS)
        assert tuple(scores.shape) == (batch_size,)

    @pytest.mark.parametrize(
        "original_pending",
        [
            param(None, id="none"),
            param(torch.zeros(2, _N_COLS), id="tensor"),
        ],
    )
    def test_x_pending_behaviour_in_batch(self, original_pending, mock_score_fn):
        """``set_X_pending`` is called after each greedy step and the original value is restored."""  # noqa: E501
        mock_score_fn.X_pending = original_pending
        bco = _make_bco()
        bco(3, mock_score_fn, _SS)
        assert (
            mock_score_fn.set_X_pending.call_count == 4
        )  # 3 greedy updates + 1 restore
        assert mock_score_fn.set_X_pending.call_args[0][0] is original_pending

    def test_block_coordinate_mechanism(self, mock_score_fn):
        """Each inner optimizer sees a space with the expected free/fixed parameters."""
        step1_x1_value = 0.5
        inner1 = MagicMock(
            return_value=(torch.tensor([[step1_x1_value, 0.0]]), torch.zeros(1))
        )
        inner2 = MagicMock(return_value=(torch.tensor([[0.0, -0.5]]), torch.zeros(1)))

        steps = (
            OptimizationStep(selector="x1", optimizer=inner1),
            OptimizationStep(selector="x2", optimizer=inner2),
        )
        schedule = CyclicOptimizationSchedule(steps=steps, n_cycles=1)
        bco = BlockCoordinateOptimizer(schedule=schedule)
        bco(1, mock_score_fn, _SS)

        # Step 1: x1 is free, so x2 must be fixed to some value
        space_seen_by_step1 = inner1.call_args[0][2]
        assert "x2" in space_seen_by_step1.continuous._fixed_values
        assert "x1" not in space_seen_by_step1.continuous._fixed_values

        # Step 2: x2 is free, so x1 must be fixed to exactly what Step 1 returned.
        space_seen_by_step2 = inner2.call_args[0][2]
        assert "x1" in space_seen_by_step2.continuous._fixed_values
        assert "x2" not in space_seen_by_step2.continuous._fixed_values
        assert space_seen_by_step2.continuous._fixed_values["x1"] == step1_x1_value
