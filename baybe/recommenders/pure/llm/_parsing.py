"""Parsing and validation of language model responses."""

from __future__ import annotations

import json
import math
import warnings
from json import JSONDecodeError

import pandas as pd

from baybe.exceptions import LLMResponseError, LLMResponseWarning
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.searchspace import SearchSpace


def extract_json_array(response: str) -> str:
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


def parse_llm_response(response: str, searchspace: SearchSpace) -> pd.DataFrame:
    """Parse a language model response into a DataFrame of recommendations.

    Args:
        response: The response from the language model.
        searchspace: The search space to validate recommendations against.

    Returns:
        A DataFrame containing the parsed recommendations.

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

    for param in searchspace.parameters:
        if param.name not in df.columns:
            raise LLMResponseError(f"Missing parameter: {param.name}")

        values = df[param.name]

        if isinstance(param, NumericalContinuousParameter):
            if not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in values
            ):
                raise LLMResponseError(
                    f"Non-finite or non-numeric values for continuous parameter: "
                    f"{param.name}"
                )
            bounds = param.bounds.to_tuple()
            min_val, max_val = bounds
            if not all(min_val <= v <= max_val for v in values):
                raise LLMResponseError(
                    f"Values for {param.name} outside bounds [{min_val}, {max_val}]"
                )

        elif isinstance(param, DiscreteParameter):
            allowed = list(param.values)
            if param.is_numerical:
                allowed_floats = [float(a) for a in allowed]
                invalid = []
                canonical = []
                for v in values:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        invalid.append(v)
                        canonical.append(v)
                        continue
                    if fv in allowed_floats:
                        canonical.append(allowed[allowed_floats.index(fv)])
                    else:
                        invalid.append(v)
                        canonical.append(v)
                if invalid:
                    raise LLMResponseError(
                        f"Invalid values {invalid} for parameter '{param.name}'. "
                        f"Allowed values are: {allowed}"
                    )
                df[param.name] = canonical
            else:
                invalid = [v for v in values if v not in allowed]
                if invalid:
                    raise LLMResponseError(
                        f"Invalid values {invalid} for parameter '{param.name}'. "
                        f"Allowed values are: {allowed}"
                    )
                # Categorical values from JSON are strings; cast to canonical
                allowed_map = {str(a): a for a in allowed}
                df[param.name] = [allowed_map.get(str(v), v) for v in values]

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

    # Recover exp_rep index so campaign metadata tracking works correctly,
    # analogous to the merge-based alignment in the Botorch hybrid recommender.
    discrete_param_names = [p.name for p in searchspace.discrete.parameters]
    if discrete_param_names:
        exp_rep = searchspace.discrete.exp_rep
        _IDX_COL = "__exp_rep_idx__"
        exp_lookup = exp_rep[discrete_param_names].copy()
        exp_lookup[_IDX_COL] = exp_rep.index
        aligned_index = pd.Index(
            df[discrete_param_names]
            .merge(exp_lookup, on=discrete_param_names, how="left")[_IDX_COL]
            .values
        )
        continuous_param_names = [p.name for p in searchspace.continuous.parameters]
        if continuous_param_names:
            rec_disc = exp_rep.loc[aligned_index]
            rec_cont = df[continuous_param_names].set_axis(aligned_index)
            return pd.concat([rec_disc, rec_cont], axis=1)
        else:
            return exp_rep.loc[aligned_index]

    return df
