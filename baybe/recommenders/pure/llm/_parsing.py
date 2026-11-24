"""Parsing and validation of language model responses."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import narwhals.stable.v2 as nw
import pandas as pd

from baybe._llm.exceptions import (
    LLMBatchConstraintValidationError,
    LLMBatchSizeError,
    LLMDuplicateRecommendationsError,
    LLMIneligiblePointsError,
    LLMUnknownParameterError,
)
from baybe._llm.extraction import extract_payload
from baybe.exceptions import LLMResponseWarning
from baybe.recommenders.pure.llm._schema import Response
from baybe.searchspace import SearchSpace
from baybe.settings import active_settings
from baybe.utils.dataframe import _df_with_backend, normalize_input_dtypes
from baybe.utils.numerical import closest_element

if TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame


def parse_llm_response(
    response: str, /, searchspace: SearchSpace, batch_size: int
) -> IntoDataFrame:
    """Parse a language model response into a dataframe of recommendations.

    The response must match the :class:`Response` schema and contain exactly
    ``batch_size`` recommendations. A mismatch is rejected before any recommendation is
    validated against the search space. The columns of the returned dataframe
    correspond to the parameters of the search space.

    Numerical discrete values accepted by their parameter's range check are snapped
    to the nearest grid value before configuration validation.

    Args:
        response: The response from the language model.
        searchspace: The search space to validate recommendations against.
        batch_size: The number of recommendations to return.

    Returns:
        A dataframe in the default dataframe backend containing exactly ``batch_size``
        distinct recommendations that belong to the search space and satisfy its batch
        constraints.

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
    recommendations = extract_payload(response, Response).recommendations

    # Check the number of recommendations before validating any of them
    if len(recommendations) != batch_size:
        raise LLMBatchSizeError(
            f"The language model returned {len(recommendations)} recommendation(s) "
            f"instead of the requested {batch_size}.",
            requested=batch_size,
            received=len(recommendations),
        )

    df = pd.DataFrame([rec.parameters for rec in recommendations])
    param_names = {p.name for p in searchspace.parameters}

    # Check for unknown parameter names
    if extra := set(df.columns) - param_names:
        raise LLMUnknownParameterError(
            f"Response contains unknown parameter names: {extra}.",
            unknown_names=extra,
            valid_names=param_names,
        )

    # The search space requires a column for every parameter to validate the
    # configurations
    if missing := param_names - set(df.columns):
        raise LLMIneligiblePointsError(
            f"The response is missing parameter columns: {sorted(missing)}."
        )

    # Numerical parameters must have numeric values
    # TODO: Check the expected column dtypes for all parameter types once parameters
    #   expose their expected representation
    if non_numeric := [
        p.name
        for p in searchspace.parameters
        if p.is_numerical
        and p.name in df
        and not pd.api.types.is_numeric_dtype(df[p.name])
    ]:
        raise LLMIneligiblePointsError(
            f"The response contains non-numeric values for parameters "
            f"{sorted(non_numeric)}."
        )

    # Snap eligible numerical discrete values before candidate and constraint checks
    for p in searchspace.discrete.parameters:
        if p.is_numerical:
            df[p.name] = df[p.name].map(
                lambda x: (
                    closest_element(p.values, cast(float, x)) if p.is_in_range(x) else x
                )
            )

    # The search space is responsible for validating the individual configurations
    invalid = nw.from_native(
        searchspace._get_invalid_configurations(df), eager_only=True
    )
    if not invalid.is_empty():
        raise LLMIneligiblePointsError(
            f"{len(invalid)} configuration(s) do not belong to the search space."
        )
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
    for constraint in searchspace.discrete.batch_constraints:
        param_name = constraint.parameters[0]
        unique_values = df[param_name].unique()
        if len(unique_values) > 1:
            raise LLMBatchConstraintValidationError(
                f"Recommendations violate the '{type(constraint).__name__}' "
                f"constraint on parameter '{param_name}': all recommendations in a "
                f"batch must share the same value, but received "
                f"{list(unique_values)}.",
                parameters=constraint.parameters,
            )

    if (n_duplicates := int(df.duplicated().sum())) > 0:
        raise LLMDuplicateRecommendationsError(
            f"{n_duplicates} recommendation(s) duplicate another recommendation of "
            f"the batch.",
            n_duplicates=n_duplicates,
        )
    return _df_with_backend(
        nw.from_native(
            df.loc[:, [p.name for p in searchspace.parameters]], eager_only=True
        ),
        active_settings.default_dataframe_backend,
    ).to_native()
