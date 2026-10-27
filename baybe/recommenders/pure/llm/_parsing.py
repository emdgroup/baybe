"""Parsing and validation of language model responses."""

from __future__ import annotations

import json
import warnings
from json import JSONDecodeError

import cattrs
import pandas as pd

from baybe.exceptions import (
    LLMBatchConstraintValidationError,
    LLMBatchSizeError,
    LLMDuplicateRecommendationsError,
    LLMIneligiblePointsError,
    LLMMalformedResponseError,
    LLMResponseWarning,
    LLMUnknownParameterError,
)
from baybe.recommenders.pure.llm._schema import ResponseItem
from baybe.searchspace import SearchSpace
from baybe.utils.dataframe import normalize_input_dtypes


def _extract_json_array(response: str, /) -> str:
    """Extract the JSON array payload from a raw language model response.

    Models often wrap the array in Markdown fences or prose, or emit several blocks
    (e.g. after reconsidering). Return the *last* array that is a list of objects (the
    intended answer), ignoring prose and stray brackets like ``x[0]``.

    Args:
        response: The raw response text.

    Returns:
        The substring spanning the extracted JSON array.

    Raises:
        ValueError: If no JSON array of objects is found in the response.
    """
    decoder = json.JSONDecoder()
    last_object_array: str | None = None
    search_start = 0
    while (start := response.find("[", search_start)) != -1:
        try:
            value, end = decoder.raw_decode(response, start)
        except JSONDecodeError:
            search_start = start + 1  # not a valid array here; try the next "["
            continue
        is_object_list = (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(x, dict) for x in value)
        )
        if is_object_list:
            last_object_array = response[start:end]
        search_start = end  # keep scanning; a later block wins
    if last_object_array is None:
        raise ValueError("No JSON array of objects found in the response.")
    return last_object_array


def parse_llm_response(
    response: str, /, searchspace: SearchSpace, batch_size: int
) -> pd.DataFrame:
    """Parse a language model response into a dataframe of recommendations.

    Only the first ``batch_size`` recommendations are validated and returned, surplus
    recommendations are discarded.

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
        LLMBatchSizeError: If the response contains fewer than ``batch_size``
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
        payload = _extract_json_array(response)
    except ValueError as e:
        raise LLMMalformedResponseError(
            "The language model response could not be parsed as a JSON array "
            "of recommendations."
        ) from e
    try:
        recommendations = json.loads(payload)
    except JSONDecodeError as e:
        raise LLMMalformedResponseError(f"Error parsing JSON output: {e}.") from e

    # Check the number of recommendations before validating any of them
    if len(recommendations) < batch_size:
        raise LLMBatchSizeError(
            f"The language model returned {len(recommendations)} recommendation(s) "
            f"instead of the requested {batch_size}.",
            requested=batch_size,
            received=len(recommendations),
        )
    # IMPROVE: Instead of discarding surplus recommendations, all recommendations could
    #   be validated individually and the first valid ones selected, saving a recovery
    #   attempt when only some of them are invalid. This requires recommendation-wise
    #   validation, a selection that respects batch constraints, and a decision on
    #   which error to report if too few valid recommendations remain.
    recommendations = recommendations[:batch_size]

    # Structure the raw JSON into typed recommendation objects
    try:
        entries: list[ResponseItem] = cattrs.structure(
            recommendations, list[ResponseItem]
        )
    except Exception as exc:
        raise LLMMalformedResponseError(
            f"The language model response does not match the expected format. {exc}"
        ) from exc

    df = pd.DataFrame([entry.parameters for entry in entries])
    param_names = {p.name for p in searchspace.parameters}

    # Check for unknown parameter names
    if extra := set(df.columns) - param_names:
        raise LLMUnknownParameterError(
            f"Response contains unknown parameter names: {extra}.",
            unknown_names=extra,
            valid_names=param_names,
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
    return df
