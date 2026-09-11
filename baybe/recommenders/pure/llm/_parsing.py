"""Parsing and validation of language model responses."""

from __future__ import annotations

import json
import warnings
from json import JSONDecodeError

import pandas as pd

from baybe.constraints.base import DiscreteFilteringConstraint
from baybe.exceptions import (
    ConstraintViolationError,
    IneligiblePointsError,
    InvalidParameterValueError,
    LLMResponseWarning,
    MalformedLLMResponseError,
    MissingParameterError,
    NonNumericParameterError,
    UnknownParameterError,
)
from baybe.recommenders.pure.llm._schema import (
    _EXPLANATION_FIELD,
    _PARAMETERS_FIELD,
)
from baybe.searchspace import SearchSpace
from baybe.utils.dataframe import fuzzy_row_match, normalize_input_dtypes
from baybe.utils.validation import validate_parameter_input


def extract_json_array(response: str, /) -> str:
    """Extract the JSON array payload from a raw language model response.

    Models often wrap the array in Markdown fences or prose, or emit several blocks
    (e.g. after reconsidering). Return the *last* array that is a list of objects (the
    intended answer), ignoring prose and stray brackets like ``x[0]``. If none is found,
    fall back to the last complete array, else the original text.

    Args:
        response: The raw response text.

    Returns:
        The substring spanning the extracted JSON array, or the original text if none.
    """
    decoder = json.JSONDecoder()
    last_array: str | None = None
    last_object_array: str | None = None
    search_start = 0
    while (start := response.find("[", search_start)) != -1:
        try:
            value, end = decoder.raw_decode(response, start)
        except JSONDecodeError:
            search_start = start + 1  # not a valid array here; try the next "["
            continue
        last_array = response[start:end]
        is_object_list = (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(x, dict) for x in value)
        )
        if is_object_list:
            last_object_array = last_array
        search_start = end  # keep scanning; a later block wins
    return last_object_array or last_array or response


def parse_llm_response(response: str, /, searchspace: SearchSpace) -> pd.DataFrame:
    """Parse a language model response into a DataFrame of recommendations.

    Args:
        response: The response from the language model.
        searchspace: The search space to validate recommendations against.

    Returns:
        A DataFrame containing the parsed recommendations, restricted to the eligible
        candidate set of the search space (as returned by
        :meth:`baybe.searchspace.discrete.SubspaceDiscrete.get_candidates`).

    Raises:
        MalformedLLMResponseError: If the response cannot be parsed into the expected
            JSON structure.
        UnknownParameterError: If a suggestion references parameters not in the search
            space.
        MissingParameterError: If a suggestion omits required search space parameters.
        NonNumericParameterError: If a suggestion gives non-numeric values for a
            numerical parameter.
        InvalidParameterValueError: If a suggestion contains invalid parameter values.
        ConstraintViolationError: If a suggestion violates a discrete constraint
            (including batch constraints) present in the search space.
        IneligiblePointsError: If a suggestion does not correspond to an eligible
            candidate of the search space.

    Warns:
        LLMResponseWarning: If the search space contains continuous constraints.
            Continuous constraints cannot be validated after the fact, so compliance
            of the LLM suggestions with such constraints is not guaranteed.
    """
    payload = extract_json_array(response)
    try:
        suggestions = json.loads(payload)
    except (JSONDecodeError, TypeError) as e:
        raise MalformedLLMResponseError(f"Error parsing JSON output: {e}.") from e

    if not isinstance(suggestions, list):
        raise MalformedLLMResponseError("Response must be a JSON array.")

    if not suggestions:
        raise MalformedLLMResponseError(
            "Response contains an empty array with no suggestions."
        )

    recommendations = []
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            raise MalformedLLMResponseError("Each suggestion must be a JSON object.")

        if _PARAMETERS_FIELD not in suggestion:
            raise MalformedLLMResponseError(
                f"Each suggestion must contain a '{_PARAMETERS_FIELD}' field."
            )

        if _EXPLANATION_FIELD not in suggestion:
            raise MalformedLLMResponseError(
                f"Each suggestion must contain an '{_EXPLANATION_FIELD}' field."
            )

        params = suggestion[_PARAMETERS_FIELD]
        if not isinstance(params, dict):
            raise MalformedLLMResponseError("Parameters must be a JSON object.")

        param_names = {p.name for p in searchspace.parameters}
        unknown = set(params.keys()) - param_names
        if unknown:
            raise UnknownParameterError(
                f"Response contains unknown parameter names: {unknown}.",
                unknown_names=unknown,
                valid_names=param_names,
            )

        recommendations.append(params)

    df = pd.DataFrame(recommendations)

    # Detect missing columns up front so they surface as a distinct error.
    missing = {p.name for p in searchspace.parameters}.difference(df.columns)
    if missing:
        raise MissingParameterError(
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
        raise NonNumericParameterError(str(e), detail=str(e)) from e
    except ValueError as e:
        raise InvalidParameterValueError(str(e), detail=str(e)) from e
    df = normalize_input_dtypes(df, searchspace.parameters)

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
            raise ConstraintViolationError(
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
            raise ConstraintViolationError(
                f"Suggestions violate the '{type(constraint).__name__}' constraint on "
                f"parameter '{param_name}': all suggestions in a batch must share the "
                f"same value, but received {list(unique_values)}.",
                constraint_name=type(constraint).__name__,
                parameters=constraint.parameters,
            )

    # Recover the exp_rep index (for campaign metadata tracking) via the same fuzzy
    # matching used for measurement input: exact for categorical, nearest numerical.
    # Matching against get_candidates() (not exp_rep directly) ensures that suggestions
    # snap to eligible points only, respecting the allow_recommending_* filters applied
    # by the campaign via FilteredSubspaceDiscrete.
    discrete_params = searchspace.discrete.parameters
    if discrete_params:
        exp_rep, _ = searchspace.discrete.get_candidates()
        aligned_index = fuzzy_row_match(exp_rep, df, discrete_params)
        # `fuzzy_row_match` silently drops suggestions with no eligible candidate, so
        # detect the shortfall explicitly rather than let them vanish.
        n_ineligible = len(df) - len(aligned_index)
        if n_ineligible > 0:
            raise IneligiblePointsError(
                f"{n_ineligible} suggestion(s) do not correspond to eligible "
                f"candidates of the search space.",
                n_ineligible=n_ineligible,
            )
        continuous_param_names = [p.name for p in searchspace.continuous.parameters]
        if continuous_param_names:
            rec_disc = exp_rep.loc[aligned_index]
            rec_cont = df[continuous_param_names].set_axis(aligned_index)
            return pd.concat([rec_disc, rec_cont], axis=1)
        else:
            return exp_rep.loc[aligned_index]

    return df
