"""Parsing and validation of language model responses."""

from __future__ import annotations

import warnings
from typing import Any, cast

import pandas as pd

from baybe._llm.exceptions import (
    LLMBatchConstraintValidationError,
    LLMBatchSizeError,
    LLMDuplicateRecommendationsError,
    LLMIneligiblePointsError,
    LLMMalformedResponseError,
    LLMUnknownParameterError,
)
from baybe._llm.extraction import extract_json, structure_json
from baybe.exceptions import LLMResponseWarning
from baybe.recommenders.pure.llm._schema import ResponseItem
from baybe.searchspace import SearchSpace
from baybe.utils.dataframe import normalize_input_dtypes
from baybe.utils.numerical import closest_element


def _is_object_array(value: Any) -> bool:
    """Return whether a decoded JSON value is a non-empty array of objects."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(entry, dict) for entry in value)
    )


def parse_llm_response(
    response: str, /, searchspace: SearchSpace, batch_size: int
) -> pd.DataFrame:
    """Parse a language model response into a dataframe of recommendations.

    The response must contain exactly ``batch_size`` recommendations. A mismatch is
    rejected before any individual recommendation is validated. The columns of the
    returned dataframe correspond to the parameters of the search space.

    Numerical discrete values accepted by their parameter's range check are snapped
    to the nearest grid value before configuration validation.

    Args:
        response: The response from the language model.
        searchspace: The search space to validate recommendations against.
        batch_size: The number of recommendations to return.

    Returns:
        A dataframe containing exactly ``batch_size`` distinct recommendations that
        belong to the search space and satisfy its batch constraints.

    Raises:
        LLMMalformedResponseError: If the response cannot be parsed into the expected
            JSON structure.
        LLMBatchSizeError: If the response does not contain exactly ``batch_size``
            recommendations.
        LLMUnknownParameterError: If a recommendation references parameters not in the
            search space.
        LLMIneligiblePointsError: If a recommendation does not belong to the search
            space, e.g. due to missing or invalid values, a violated constraint, or
            because it is currently not selectable.
        LLMBatchConstraintValidationError: If the recommendations violate a batch
            constraint of the search space.
        LLMDuplicateRecommendationsError: If the recommendations are not distinct.

    Warns:
        LLMResponseWarning: If the search space contains continuous constraints.
            Continuous constraints cannot be validated after the fact, so compliance
            of the LLM recommendations with such constraints is not guaranteed.
    """
    try:
        recommendations = extract_json(response, _is_object_array)
    except ValueError as e:
        raise LLMMalformedResponseError(
            "The language model response could not be parsed as a JSON array "
            "of recommendations."
        ) from e

    # Check the number of recommendations before validating any of them
    if len(recommendations) != batch_size:
        raise LLMBatchSizeError(
            f"The language model returned {len(recommendations)} recommendation(s) "
            f"instead of the requested {batch_size}.",
            requested=batch_size,
            received=len(recommendations),
        )

    # Structure the raw JSON into typed recommendation objects
    entries = structure_json(recommendations, list[ResponseItem])

    df = pd.DataFrame([entry.parameters for entry in entries])
    param_names = {p.name for p in searchspace.parameters}

    # Check for unknown parameter names
    if extra := set(df.columns) - param_names:
        raise LLMUnknownParameterError(
            f"Response contains unknown parameter names: {extra}.",
            unknown_names=extra,
            valid_names=param_names,
        )

    # Snap eligible numerical discrete values before candidate and constraint checks
    for p in searchspace.discrete.parameters:
        # Since validation report missing or non-numeric columns, only attempt snapping
        # for numeric columns that exist without already verifying
        if p.is_numerical and p.name in df.select_dtypes(include="number"):
            df[p.name] = df[p.name].map(
                lambda x: (
                    closest_element(p.values, cast(float, x)) if p.is_in_range(x) else x
                )
            )

    # The search space is responsible for validating the individual configurations
    try:
        searchspace._validate_configurations(df)
    except ValueError as e:
        raise LLMIneligiblePointsError(str(e)) from e
    df = normalize_input_dtypes(df, searchspace.parameters)

    continuous_constraints = (
        *searchspace.continuous.constraints_lin_eq,
        *searchspace.continuous.constraints_lin_ineq,
        *searchspace.continuous.constraints_nonlin,
    )
    if continuous_constraints:
        names = ", ".join(f"'{type(c).__name__}'" for c in continuous_constraints)
        # TODO: Adjust this warning once we have a mechanism to validate continuous
        # constraints for LLM
        warnings.warn(
            f"The search space contains continuous constraints ({names}) that cannot "
            f"be validated. The LLM recommendations may violate these constraints.",
            LLMResponseWarning,
            stacklevel=2,
        )

    # TODO: Batch constraints and duplicates concern the batch as a whole and are
    #   therefore still validated here instead of by the search space. Move them once
    #   the search space offers a batch-level validation.
    for constraint in searchspace.discrete.constraints_batch:
        param_name = constraint.parameters[0]
        unique_values = df[param_name].unique()
        if len(unique_values) > 1:
            raise LLMBatchConstraintValidationError(
                f"Recommendations violate the '{type(constraint).__name__}' "
                f"constraint on parameter '{param_name}': all recommendations in a "
                f"batch must share the same value, but received "
                f"{list(unique_values)}.",
                constraint_name=type(constraint).__name__,
                parameters=constraint.parameters,
            )

    if (n_duplicates := int(df.duplicated().sum())) > 0:
        raise LLMDuplicateRecommendationsError(
            f"{n_duplicates} recommendation(s) duplicate another recommendation of "
            f"the batch.",
            n_duplicates=n_duplicates,
        )
    return df.loc[:, [p.name for p in searchspace.parameters]]
