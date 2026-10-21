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


def _count_forbidden_configurations(df: pd.DataFrame, searchspace: SearchSpace) -> int:
    """Count the recommendations that match a currently non-selectable configuration.

    These are the discrete configurations filtered out of the search space passed to
    the recommender, i.e. excluded ones and, depending on the campaign settings, those
    already measured, recommended or pending.

    Args:
        df: The recommendations.
        searchspace: The search space passed to the recommender.

    Returns:
        The number of recommendations matching a not selectable configuration.
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
    """Parse a language model response into a dataframe of recommendations.

    Only the first ``batch_size`` recommendations are validated and returned, surplus
    recommendations are discarded.

    Args:
        response: The response from the language model.
        searchspace: The search space to validate recommendations against.
        batch_size: The number of recommendations to return.

    Returns:
        A dataframe containing exactly ``batch_size`` distinct recommendations whose
        parameter values are valid and whose discrete part satisfies the discrete
        constraints of the search space.

    Raises:
        LLMMalformedResponseError: If the response cannot be parsed into the expected
            JSON structure.
        LLMBatchSizeError: If the response contains fewer than ``batch_size``
            recommendations.
        LLMUnknownParameterError: If a recommendation references parameters not in the
            search space.
        LLMMissingParameterError: If a recommendation omits required search space
            parameters.
        LLMNonNumericParameterError: If a recommendation gives non-numeric values for a
            numerical parameter.
        LLMInvalidParameterValueError: If a recommendation contains invalid parameter
            values, including inactive values of label-like parameters.
        LLMConstraintViolationError: If a recommendation violates a discrete constraint
            (including batch constraints) present in the search space.
        LLMIneligiblePointsError: If a recommendation corresponds to a discrete
            configuration that is currently not selectable.
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

    param_names = {p.name for p in searchspace.parameters}
    configurations = []
    for recommendation in recommendations:
        if _PARAMETERS_FIELD not in recommendation:
            raise LLMMalformedResponseError(
                f"Each recommendation must contain a '{_PARAMETERS_FIELD}' field."
            )

        if _EXPLANATION_FIELD not in recommendation:
            raise LLMMalformedResponseError(
                f"Each recommendation must contain an '{_EXPLANATION_FIELD}' field."
            )

        params = recommendation[_PARAMETERS_FIELD]
        if not isinstance(params, dict):
            raise LLMMalformedResponseError("Parameters must be a JSON object.")

        if unknown := set(params.keys()) - param_names:
            raise LLMUnknownParameterError(
                f"Response contains unknown parameter names: {unknown}.",
                unknown_names=unknown,
                valid_names=param_names,
            )

        configurations.append(params)

    df = pd.DataFrame(configurations)

    # Detect missing columns up front so they surface as a distinct error.
    if missing := {p.name for p in searchspace.parameters}.difference(df.columns):
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
            f"be validated. The LLM recommendations may violate these constraints.",
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
                f"{len(invalid_idx)} recommendation(s) violate the "
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
                f"Recommendations violate the '{type(constraint).__name__}' "
                f"constraint on parameter '{param_name}': all recommendations in a "
                f"batch must share the same value, but received "
                f"{list(unique_values)}.",
                constraint_name=type(constraint).__name__,
                parameters=constraint.parameters,
            )

    if (n_forbidden := _count_forbidden_configurations(df, searchspace)) > 0:
        raise LLMIneligiblePointsError(
            f"{n_forbidden} recommendation(s) correspond to configurations that are "
            f"currently not selectable.",
            n_ineligible=n_forbidden,
        )

    # TODO: We should at some point have a more explicit check against the actual search
    # space object. Currently, we can e.g. not verify search space that use permutation
    # constraints as the search space contains an arbitrary representation of a
    # permutation group. Also, search spaces that are constructed as an explicit list of
    # configurations (e.g. if generated from a dataframe) can currently not be validated

    if (n_duplicates := int(df.duplicated().sum())) > 0:
        raise LLMDuplicateRecommendationsError(
            f"{n_duplicates} recommendation(s) duplicate another recommendation of "
            f"the batch.",
            n_duplicates=n_duplicates,
        )
    return df
