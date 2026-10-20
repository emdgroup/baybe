"""Parsing and validation of language model responses."""

from __future__ import annotations

import json
import warnings
from json import JSONDecodeError

import pandas as pd
from exceptiongroup import ExceptionGroup

from baybe.constraints.base import DiscreteFilteringConstraint
from baybe.exceptions import (
    LLMBatchSizeError,
    LLMConstraintViolationError,
    LLMDuplicateRecommendationsError,
    LLMIneligiblePointsError,
    LLMInvalidParameterValueError,
    LLMMalformedResponseError,
    LLMMissingParameterError,
    LLMNonNumericParameterError,
    LLMResponseWarning,
    LLMUnknownParameterError,
)
from baybe.recommenders.pure.llm._schema import (
    _EXPLANATION_FIELD,
    _PARAMETERS_FIELD,
)
from baybe.searchspace import SearchSpace
from baybe.searchspace.validation import validate_dataframe_active_values
from baybe.utils.dataframe import normalize_input_dtypes
from baybe.utils.validation import validate_parameter_input


def _extract_json_array(response: str, /) -> str | None:
    """Extract the JSON array payload from a raw language model response.

    Models often wrap the array in Markdown fences or prose, or emit several blocks
    (e.g. after reconsidering). Return the *last* array that is a list of objects (the
    intended answer), ignoring prose and stray brackets like ``x[0]``.

    Args:
        response: The raw response text.

    Returns:
        The substring spanning the extracted JSON array, or ``None`` if no array of
        objects is found.
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
    return last_object_array


def _count_forbidden_configurations(df: pd.DataFrame, searchspace: SearchSpace) -> int:
    """Count the suggestions that match a currently not selectable configuration.

    These are the discrete configurations filtered out of the search space passed to
    the recommender, i.e. excluded ones and, depending on the campaign settings, those
    already measured, recommended or pending.

    Args:
        df: The suggestions.
        searchspace: The search space passed to the recommender.

    Returns:
        The number of suggestions matching a not selectable configuration.
    """
    discrete = searchspace.discrete
    if not discrete.parameters:
        return 0
    eligible, _ = discrete.get_candidates()
    forbidden = discrete.exp_rep[~discrete.exp_rep.index.isin(eligible.index)]
    if forbidden.empty:
        return 0

    names = [p.name for p in discrete.parameters]
    forbidden_rows = set(forbidden[names].itertuples(index=False, name=None))
    return sum(
        row in forbidden_rows for row in df[names].itertuples(index=False, name=None)
    )


def parse_llm_response(
    response: str, /, searchspace: SearchSpace, batch_size: int
) -> pd.DataFrame:
    """Parse a language model response into a DataFrame of recommendations.

    Only the first ``batch_size`` suggestions are validated and returned, surplus
    suggestions are discarded.

    Args:
        response: The response from the language model.
        searchspace: The search space to validate recommendations against.
        batch_size: The number of recommendations to return.

    Returns:
        A DataFrame containing exactly ``batch_size`` distinct recommendations whose
        parameter values are valid and whose discrete part satisfies the discrete
        constraints of the search space.

    Raises:
        LLMMalformedResponseError: If the response cannot be parsed into the expected
            JSON structure.
        LLMBatchSizeError: If the response contains fewer than ``batch_size``
            suggestions.
        LLMUnknownParameterError: If a suggestion references parameters not in the
            search space.
        LLMMissingParameterError: If a suggestion omits required search space
            parameters.
        LLMNonNumericParameterError: If a suggestion gives non-numeric values for a
            numerical parameter.
        LLMInvalidParameterValueError: If a suggestion contains invalid parameter
            values, including inactive values of label-like parameters.
        LLMConstraintViolationError: If a suggestion violates a discrete constraint
            (including batch constraints) present in the search space.
        LLMIneligiblePointsError: If a suggestion corresponds to a discrete
            configuration that is currently not selectable.
        LLMDuplicateRecommendationsError: If the recommendations are not distinct.

    Warns:
        LLMResponseWarning: If the search space contains continuous constraints.
            Continuous constraints cannot be validated after the fact, so compliance
            of the LLM suggestions with such constraints is not guaranteed.
    """
    payload = _extract_json_array(response)
    if payload is None:
        raise LLMMalformedResponseError("Response contains no JSON array of objects.")
    try:
        suggestions = json.loads(payload)
    except JSONDecodeError as e:
        raise LLMMalformedResponseError(f"Error parsing JSON output: {e}.") from e

    # Check the number of suggestions before validating any of them
    if len(suggestions) < batch_size:
        raise LLMBatchSizeError(
            f"The language model returned {len(suggestions)} recommendation(s) "
            f"instead of the requested {batch_size}.",
            requested=batch_size,
            received=len(suggestions),
        )
    # IMPROVE: Instead of discarding surplus suggestions, all suggestions could be
    #   validated individually and the first valid ones selected, saving a recovery
    #   attempt when only some of them are invalid. This requires suggestion-wise
    #   validation, a selection that respects batch constraints, and a decision on
    #   which error to report if too few valid suggestions remain.
    suggestions = suggestions[:batch_size]

    recommendations = []
    for suggestion in suggestions:
        if _PARAMETERS_FIELD not in suggestion:
            raise LLMMalformedResponseError(
                f"Each suggestion must contain a '{_PARAMETERS_FIELD}' field."
            )

        if _EXPLANATION_FIELD not in suggestion:
            raise LLMMalformedResponseError(
                f"Each suggestion must contain an '{_EXPLANATION_FIELD}' field."
            )

        params = suggestion[_PARAMETERS_FIELD]
        if not isinstance(params, dict):
            raise LLMMalformedResponseError("Parameters must be a JSON object.")

        param_names = {p.name for p in searchspace.parameters}
        unknown = set(params.keys()) - param_names
        if unknown:
            raise LLMUnknownParameterError(
                f"Response contains unknown parameter names: {unknown}.",
                unknown_names=unknown,
                valid_names=param_names,
            )

        recommendations.append(params)

    df = pd.DataFrame(recommendations)

    # Detect missing columns up front so they surface as a distinct error.
    missing = {p.name for p in searchspace.parameters}.difference(df.columns)
    if missing:
        raise LLMMissingParameterError(
            f"Response is missing values for the following parameters: {missing}.",
            parameters=missing,
        )

    # Validate parameter columns as for measurement input. `validate_parameter_input`
    # raises `TypeError` for non-numeric entries and `ValueError` for other invalid
    # values, which we surface as distinct error types.
    try:
        validate_parameter_input(
            df,
            searchspace.parameters,
            numerical_measurements_must_be_within_tolerance=True,
        )
    except TypeError as e:
        raise LLMNonNumericParameterError(str(e), detail=str(e)) from e
    except ValueError as e:
        raise LLMInvalidParameterValueError(str(e), detail=str(e)) from e
    df = normalize_input_dtypes(df, searchspace.parameters)

    # Inactive values of label-like parameters (e.g. other tasks in transfer learning)
    # may appear in measurements but must not be recommended
    try:
        validate_dataframe_active_values(df, searchspace.parameters)
    except ExceptionGroup as e:
        detail = " ".join(str(ex) for ex in e.exceptions)
        raise LLMInvalidParameterValueError(detail, detail=detail) from e

    continuous_constraints = (
        *searchspace.continuous.constraints_lin_eq,
        *searchspace.continuous.constraints_lin_ineq,
        *searchspace.continuous.constraints_nonlin,
    )
    if continuous_constraints:
        names = ", ".join(f"'{type(c).__name__}'" for c in continuous_constraints)
        warnings.warn(
            f"The search space contains continuous constraints ({names}) that cannot "
            f"be validated. The LLM suggestions may violate these constraints.",
            LLMResponseWarning,
            stacklevel=2,
        )

    for constraint in searchspace.discrete.constraints:
        # Only filtering constraints expose row-level validity; batch constraints are
        # handled separately below.
        if not isinstance(constraint, DiscreteFilteringConstraint):
            continue
        invalid_idx = constraint.get_invalid(df)
        if not invalid_idx.empty:
            raise LLMConstraintViolationError(
                f"{len(invalid_idx)} suggestion(s) violate the "
                f"'{type(constraint).__name__}' constraint on parameters "
                f"{constraint.parameters}.",
                constraint_name=type(constraint).__name__,
                parameters=constraint.parameters,
            )

    for constraint in searchspace.discrete.constraints_batch:
        param_name = constraint.parameters[0]
        unique_values = df[param_name].unique()
        if len(unique_values) > 1:
            raise LLMConstraintViolationError(
                f"Suggestions violate the '{type(constraint).__name__}' constraint on "
                f"parameter '{param_name}': all suggestions in a batch must share the "
                f"same value, but received {list(unique_values)}.",
                constraint_name=type(constraint).__name__,
                parameters=constraint.parameters,
            )

    if (n_forbidden := _count_forbidden_configurations(df, searchspace)) > 0:
        raise LLMIneligiblePointsError(
            f"{n_forbidden} suggestion(s) correspond to configurations that are "
            f"currently not selectable.",
            n_ineligible=n_forbidden,
        )

    # TODO: We should at some point have a more explicit check against the actual search
    # space object. Currently, we can e.g. not verify search space that use permutation
    # constraints as the search space contains an arbitrary representation of a
    # permutation group. Also, search spaces that are constructed as an explicit list of
    # configurations (e.g. if generated from a DataFrame) can currently not be validated

    if (n_duplicates := int(df.duplicated().sum())) > 0:
        raise LLMDuplicateRecommendationsError(
            f"{n_duplicates} suggestion(s) duplicate another suggestion of the batch.",
            n_duplicates=n_duplicates,
        )
    return df
