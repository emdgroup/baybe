"""Shared risk- and cost-adjusted scoring for multi-step actions."""

from __future__ import annotations

import gc
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, cast

from typing_extensions import override

from baybe.exceptions import IncompatibilityError

if TYPE_CHECKING:
    from botorch.acquisition.acquisition import (
        AcquisitionFunction as BoAcquisitionFunction,
    )
    from botorch.models.model import Model
    from torch import Tensor


ScoreScale = Literal["linear", "log"]
"""The representation used by an acquisition function's gain."""


def _resolve_score_scale(acquisition: object) -> ScoreScale:
    """Resolve the declared gain scale of a supported acquisition function.

    Args:
        acquisition: The concrete BayBE acquisition function.

    Returns:
        The representation used by the acquisition's nonnegative gain.

    Raises:
        IncompatibilityError: If the acquisition does not have a supported gain
            contract.
    """
    from baybe.acquisition import (
        ExpectedImprovement,
        LogExpectedImprovement,
        qExpectedImprovement,
        qLogEI,
        qLogNEHVI,
        qLogNEI,
        qLogNParEGO,
        qNEHVI,
        qNEI,
    )
    from baybe.multistep.acquisition import PartialKnowledgeGradientFN

    if isinstance(
        acquisition,
        (
            ExpectedImprovement,
            qExpectedImprovement,
            qNEI,
            qNEHVI,
            PartialKnowledgeGradientFN,
        ),
    ):
        return "linear"
    if isinstance(
        acquisition,
        (
            LogExpectedImprovement,
            qLogEI,
            qLogNEI,
            qLogNEHVI,
            qLogNParEGO,
        ),
    ):
        return "log"

    raise IncompatibilityError(
        f"The acquisition function '{acquisition.__class__.__name__}' does not "
        "declare a supported nonnegative gain scale for multi-step action scoring. "
        "Use EI/qEI or LogEI/qLogEI, or an explicitly supported multi-output gain."
    )


def _score_actions(
    acquisition_values: Tensor,
    success_probabilities: Tensor,
    cost: float | Tensor,
    *,
    scale: ScoreScale,
    allow_negative_gains: bool = False,
) -> Tensor:
    """Apply local execution risk and cost once to acquisition gains.

    Args:
        acquisition_values: Raw acquisition gains, in linear or log form.
        success_probabilities: Local next-measurement success probabilities,
            with exactly the same shape, dtype, and device as the gains.
        cost: Positive cost of the next node measurement.
        scale: Representation of the raw gains.
        allow_negative_gains: Permit finite signed linear estimates for p-KGFN,
            whose Monte-Carlo estimate can be negative.

    Returns:
        The risk- and cost-adjusted gains. A probability of exactly zero maps to
        ``-inf`` and is therefore unselectable, with no epsilon substitution.

    Raises:
        IncompatibilityError: If values, probability, cost, scale, or shapes are
            invalid.
    """
    import torch

    if scale not in ("linear", "log"):
        raise IncompatibilityError(f"Unsupported acquisition score scale: {scale!r}.")
    if acquisition_values.shape != success_probabilities.shape:
        raise IncompatibilityError(
            "Acquisition values and success probabilities must have identical "
            f"shapes, got {tuple(acquisition_values.shape)} and "
            f"{tuple(success_probabilities.shape)}."
        )
    if (
        acquisition_values.device != success_probabilities.device
        or acquisition_values.dtype != success_probabilities.dtype
    ):
        raise IncompatibilityError(
            "Acquisition values and success probabilities must share dtype and device."
        )
    if (
        not torch.isfinite(success_probabilities).all()
        or (success_probabilities < 0).any()
        or (success_probabilities > 1).any()
    ):
        raise IncompatibilityError(
            "Local success probabilities must be finite values in [0, 1]."
        )
    if isinstance(cost, torch.Tensor):
        if (
            cost.shape != acquisition_values.shape
            or cost.device != acquisition_values.device
            or cost.dtype != acquisition_values.dtype
        ):
            raise IncompatibilityError(
                "Tensor action costs must match gain shape, dtype, and device."
            )
        if not torch.isfinite(cost).all() or (cost <= 0).any():
            raise IncompatibilityError(
                "Action costs must be finite and strictly positive."
            )
        cost_tensor = cost
    else:
        if not math.isfinite(cost) or cost <= 0:
            raise IncompatibilityError(
                "Action cost must be finite and strictly positive."
            )
        cost_tensor = acquisition_values.new_tensor(cost)

    if scale == "linear":
        if not torch.isfinite(acquisition_values).all():
            raise IncompatibilityError("Linear acquisition gains must be finite.")
        if not allow_negative_gains and (acquisition_values < 0).any():
            raise IncompatibilityError(
                "Linear gain acquisitions must not return negative values."
            )
        scores = acquisition_values * success_probabilities / cost_tensor
    else:
        if (
            torch.isnan(acquisition_values).any()
            or torch.isposinf(acquisition_values).any()
        ):
            raise IncompatibilityError(
                "Log acquisition gains may be finite or -inf, but not NaN or +inf."
            )
        # Avoid evaluating log(0): even an unselected torch.where branch can
        # otherwise introduce infinite gradients. This is not an epsilon rescue;
        # the zero-probability branch is still represented by exactly -inf.
        safe_probabilities = torch.where(
            success_probabilities > 0,
            success_probabilities,
            torch.ones_like(success_probabilities),
        )
        scores = acquisition_values + safe_probabilities.log() - cost_tensor.log()

    return torch.where(
        success_probabilities > 0,
        scores,
        torch.full_like(scores, -torch.inf),
    )


def _reduce_score_branches(branch_values: Tensor, *, scale: ScoreScale) -> Tensor:
    """Average pending branches in the gain's own representation.

    Args:
        branch_values: Raw acquisition values with a nonempty leading branch axis.
        scale: Representation of the acquisition values.

    Returns:
        The arithmetic mean of linear gains or log of the arithmetic mean of
        log-represented gains.

    Raises:
        IncompatibilityError: If branch values are empty, malformed, or non-finite
            outside the valid ``-inf`` log-gain case.
    """
    import torch

    if branch_values.ndim < 1 or branch_values.shape[0] < 1:
        raise IncompatibilityError(
            "At least one pending acquisition branch is required."
        )
    if scale == "linear":
        if not torch.isfinite(branch_values).all():
            raise IncompatibilityError("Linear pending gains must be finite.")
        return branch_values.mean(dim=0)
    if scale == "log":
        if torch.isnan(branch_values).any() or torch.isposinf(branch_values).any():
            raise IncompatibilityError(
                "Log pending gains may be finite or -inf, but not NaN or +inf."
            )
        return torch.logsumexp(branch_values, dim=0) - math.log(branch_values.shape[0])
    raise IncompatibilityError(f"Unsupported acquisition score scale: {scale!r}.")


def _make_tensor_score_acquisition(
    *,
    model: Model,
    raw_score: Callable[[Tensor], Tensor],
    success_probability: Callable[[Tensor], Tensor],
    cost: float,
    scale: ScoreScale,
    allow_negative_gains: bool = False,
) -> BoAcquisitionFunction:
    """Create a lazy BoTorch adapter that weights a raw score before optimization.

    Args:
        model: The model supplied to the native BoTorch acquisition base class.
        raw_score: The differentiable, unweighted score function.
        success_probability: The differentiable local action success predictor.
        cost: Cost of the next node measurement.
        scale: Representation of the raw score.
        allow_negative_gains: Permit signed linear p-KGFN Monte-Carlo estimates.

    Returns:
        A BoTorch acquisition function evaluating the shared weighted score.
    """
    from attrs import define, setters
    from botorch.acquisition.acquisition import (
        AcquisitionFunction as BoAcquisitionFunction,
    )
    from botorch.utils.transforms import t_batch_mode_transform

    @define(eq=False, repr=False, slots=False, init=False, on_setattr=setters.NO_OP)
    class _TensorScoreAcquisition(BoAcquisitionFunction):
        """Adapt a raw tensor score to shared multi-step action weighting."""

        @override
        @cast(
            "Callable[[Callable[..., Tensor]], Callable[..., Tensor]]",
            t_batch_mode_transform(expected_q=1),
        )
        def forward(self, X: Tensor) -> Tensor:
            return _score_actions(
                raw_score(X),
                success_probability(X),
                cost,
                scale=scale,
                allow_negative_gains=allow_negative_gains,
            )

    return _TensorScoreAcquisition(model=model)


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
