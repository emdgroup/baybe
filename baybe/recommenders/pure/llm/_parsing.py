"""Parsing and validation of language model responses."""

from __future__ import annotations

import json
import warnings
from json import JSONDecodeError

import pandas as pd

from baybe.exceptions import LLMResponseError, LLMResponseWarning
from baybe.searchspace import SearchSpace
from baybe.utils.dataframe import fuzzy_row_match, normalize_input_dtypes
from baybe.utils.validation import validate_parameter_input


def extract_json_array(response: str, /) -> str:
    """Extract the JSON array payload from a raw language model response.

    Language models frequently wrap their output in Markdown code fences or add
    surrounding prose despite instructions to the contrary. This helper isolates the
    outermost ``[...]`` block so that such responses can still be parsed.

    Args:
        response: The raw response text.

    Returns:
        The substring spanning the outermost JSON array, or the original text if no
        array delimiters are found.
    """
    start = response.find("[")
    end = response.rfind("]")
    if start != -1 and end != -1 and start < end:
        return response[start : end + 1]
    return response


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
        LLMResponseError: If the response cannot be parsed, contains invalid parameter
            values, or violates any discrete constraint (including batch constraints)
            present in the search space.

    Warns:
        LLMResponseWarning: If the search space contains continuous constraints.
            Continuous constraints cannot be validated after the fact, so compliance
            of the LLM suggestions with such constraints is not guaranteed.
    """
    payload = extract_json_array(response) if isinstance(response, str) else response
    try:
        suggestions = json.loads(payload)
    except (JSONDecodeError, TypeError) as e:
        raise LLMResponseError(f"Error parsing JSON output: {e}") from e

    if not isinstance(suggestions, list):
        raise LLMResponseError("Response must be a JSON array")

    if not suggestions:
        raise LLMResponseError("Response contains an empty array with no suggestions.")

    recommendations = []
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            raise LLMResponseError("Each suggestion must be a JSON object")

        if "parameters" not in suggestion:
            raise LLMResponseError("Each suggestion must contain a 'parameters' field")

        if "explanation" not in suggestion:
            raise LLMResponseError(
                "Each suggestion must contain an 'explanation' field"
            )

        params = suggestion["parameters"]
        if not isinstance(params, dict):
            raise LLMResponseError("Parameters must be a JSON object")

        param_names = {p.name for p in searchspace.parameters}
        unknown = set(params.keys()) - param_names
        if unknown:
            raise LLMResponseError(
                f"Response contains unknown parameter names: {unknown}"
            )

        recommendations.append(params)

    df = pd.DataFrame(recommendations)

    # Validate parameter columns as for measurement input. Called directly, not via
    # `preprocess_dataframe`, which is a no-op under `preprocess_dataframes=False`.
    # TODO: Sharpen the caught errors — different validation failures (e.g. missing
    #   columns vs. out-of-range values) will require different reactions (hard error
    #   vs. recovery vs. eligibility drop) rather than a single wrapped error.
    try:
        validate_parameter_input(
            df,
            searchspace.parameters,
            numerical_measurements_must_be_within_tolerance=True,
        )
    except (ValueError, TypeError) as e:
        raise LLMResponseError(str(e)) from e
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
        invalid_idx = constraint.get_invalid(df)
        if not invalid_idx.empty:
            raise LLMResponseError(
                f"{len(invalid_idx)} suggestion(s) violate the "
                f"'{type(constraint).__name__}' constraint on parameters "
                f"{constraint.parameters}."
            )

    for constraint in searchspace.discrete.constraints_batch:
        param_name = constraint.parameters[0]
        unique_values = df[param_name].unique()
        if len(unique_values) > 1:
            raise LLMResponseError(
                f"Suggestions violate 'DiscreteBatchConstraint' on parameter "
                f"'{param_name}': all suggestions in a batch must share the same "
                f"value, but received {list(unique_values)}."
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
        continuous_param_names = [p.name for p in searchspace.continuous.parameters]
        if continuous_param_names:
            rec_disc = exp_rep.loc[aligned_index]
            rec_cont = df[continuous_param_names].set_axis(aligned_index)
            return pd.concat([rec_disc, rec_cont], axis=1)
        else:
            return exp_rep.loc[aligned_index]

    return df
