"""Execution helpers for reserved complete DAG configurations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd

from baybe.exceptions import IncompatibilityError

if TYPE_CHECKING:
    from baybe.multistep.campaign import MultiStepCampaign


def _values_match(expected: object, actual: object) -> bool:
    """Compare scalar frame values while treating paired missing values equally."""
    expected_missing = pd.isna(expected)
    actual_missing = pd.isna(actual)
    if expected_missing or actual_missing:
        return bool(expected_missing and actual_missing)
    return bool(expected == actual)


def evaluate_configuration(
    campaign: MultiStepCampaign,
    experiment_id: str,
    evaluator: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    """Execute a reserved configuration until success or a known failure.

    Each callback receives a defensive copy of one pending local node action. Its
    response is validated and stored before the next node action is requested. A
    technical exception is deliberately propagated without changing the pending
    action, so a caller may explicitly resume the same attempt later.

    Args:
        campaign: Campaign holding the reserved configuration and process state.
        experiment_id: Identifier returned when the complete configuration was
            registered, sampled, or recommended.
        evaluator: Function that turns one local action into one measurement row.

    Returns:
        ``None``. Measurements are stored in ``campaign.data``.

    Raises:
        IncompatibilityError: If the action state or evaluator response violates the
            one-action/one-measurement contract.
        Exception: Any technical exception raised by the evaluator is propagated.
    """
    if not callable(evaluator):
        raise TypeError("evaluator must be callable.")

    while True:
        action = campaign.get_next_configuration_action(experiment_id)
        if action.empty:
            return None

        response = evaluator(action.copy(deep=True))
        if not isinstance(response, pd.DataFrame):
            raise IncompatibilityError(
                "The evaluator response must be a pandas DataFrame."
            )
        if len(response) != 1:
            raise IncompatibilityError(
                "The evaluator response must contain exactly one measurement row."
            )
        if not response.columns.is_unique:
            raise IncompatibilityError(
                "The evaluator response must have unique measurement columns."
            )

        expected = action.iloc[0].to_dict()
        missing = sorted(set(expected) - set(response.columns))
        if missing:
            raise IncompatibilityError(
                f"The evaluator response is missing action identity columns {missing}."
            )
        actual = response.iloc[0]
        mismatched = [
            column
            for column, value in expected.items()
            if not _values_match(value, actual[column])
        ]
        if mismatched:
            raise IncompatibilityError(
                "The evaluator response does not match its reserved action in "
                f"columns {mismatched}."
            )

        campaign.add_measurements(response.copy(deep=True))


__all__ = ["evaluate_configuration"]
