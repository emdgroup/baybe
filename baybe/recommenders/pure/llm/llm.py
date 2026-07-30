"""LLM-based recommender for experimental design."""

from __future__ import annotations

import gc
import json
import math
import warnings
from json import JSONDecodeError
from types import SimpleNamespace
from typing import Any, ClassVar

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of, min_len
from typing_extensions import override

from baybe.exceptions import (
    IncompatibilityError,
    LLMResponseError,
    LLMResponseWarning,
)
from baybe.objectives.base import Objective
from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders.pure.base import PureRecommender
from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType
from baybe.utils.conversion import to_string
from baybe.utils.validation import preprocess_dataframe

_PROMPT_TEMPLATE = """\
You are an expert experimental design assistant. Your task is to suggest new \
experimental conditions based on the following information:

EXPERIMENT DESCRIPTION:
{{ experiment_description }}

OPTIMIZATION OBJECTIVE:
{{ objective_description }}

{% if objective is not none %}
OPTIMIZATION TARGETS:
{{ objective }}
{% endif %}
PARAMETERS:
{% for param in parameters %}
Parameter: {{ param.name }}
{% if param.description is not none %}
Description: {{ param.description }}
{% endif %}
Type: {{ param.type }}
{% if param.type == 'continuous' %}
Bounds: [{{ param.bounds[0] }}, {{ param.bounds[1] }}]
{% else %}
Allowed values: {{ param.values }}
{% endif %}
{% if param.unit is not none %}
Unit: {{ param.unit }}
{% endif %}

{% endfor %}

{% if measurements is not none and not measurements.empty %}
PREVIOUS MEASUREMENTS:
{{ measurements.to_string(index=False) }}
{% endif %}

{% if pending_experiments is not none and not pending_experiments.empty %}
PENDING EXPERIMENTS:
The following experiments have already been proposed and are awaiting results.
Do not recommend these again.
{{ pending_experiments.to_string(index=False) }}
{% endif %}

Please suggest {{ batch_size }} new experimental conditions that are likely to \
improve the optimization objective.
For each suggestion, provide:
1. A brief explanation of why you chose these values
2. The values for each parameter

{% if format_instructions is not none %}
{{ format_instructions }}
{% else %}
Format your response as a JSON array of objects with the following structure \
(no backticks):
[
  {
    "explanation": "Brief explanation of the suggestion",
    "parameters": {
      "param1": value1,
      "param2": value2,
      ...
    }
  },
  ...
]
{% endif %}
"""

_RECOVERY_PROMPT_TEMPLATE = """\
The previous response was malformed and could not be parsed as JSON. Please \
correct the response to match the required format.

ERROR: {{ error }}

ORIGINAL RESPONSE:
{{ original_response }}

PARAMETERS:
{% for param in parameters %}
Parameter: {{ param.name }}
Type: {{ param.type }}
{% if param.type == 'continuous' %}
Bounds: [{{ param.bounds[0] }}, {{ param.bounds[1] }}]
{% else %}
Allowed values: {{ param.values }}
{% endif %}
{% endfor %}

Please provide a corrected JSON response that follows the required format:
{% if format_instructions is not none %}
{{ format_instructions }}
{% else %}
[
  {
    "explanation": "Brief explanation of the suggestion",
    "parameters": {
      "param1": value1,
      "param2": value2,
      ...
    }
  },
  ...
]
{% endif %}\
"""


_RESERVED_LITELLM_KEYS = frozenset({"model", "messages"})


def _extract_parameter_info(
    parameters: tuple[Parameter, ...],
) -> list[SimpleNamespace]:
    """Extract parameter information for prompt construction.

    Args:
        parameters: The parameters from the search space.

    Returns:
        A list of namespace objects containing parameter information.

    Raises:
        IncompatibilityError: If a parameter type is not supported.
    """
    infos = []
    for param in parameters:
        info: dict[str, Any] = {
            "name": param.name,
            "description": param.description,
            "unit": param.unit,
        }

        if isinstance(param, NumericalContinuousParameter):
            info["type"] = "continuous"
            info["bounds"] = param.bounds.to_tuple()
        elif isinstance(param, DiscreteParameter):
            info["type"] = "discrete_numeric" if param.is_numerical else "categorical"
            info["values"] = list(param.values)
        else:
            raise IncompatibilityError(
                f"Parameter '{param.name}' has unsupported type "
                f"'{type(param).__name__}' for "
                f"'{LLMRecommender.__name__}'. Only "
                f"'{NumericalContinuousParameter.__name__}' and "
                f"'{DiscreteParameter.__name__}' subclasses are supported."
            )

        infos.append(SimpleNamespace(**info))

    return infos


def _extract_json_array(response: str) -> str:
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


@define(slots=False)
class LLMRecommender(PureRecommender):
    """Recommender that uses a language model to suggest new experimental points.

    Unlike other pure recommenders, this recommender does not implement the
    ``_recommend_discrete``/``_recommend_continuous``/``_recommend_hybrid`` hooks.
    The language model returns a complete set of parameter configurations that already
    constitutes the recommendation, so there is no per-subspace selection step to
    delegate to those hooks. Instead, :meth:`recommend` is fully overridden and builds
    the result directly from the model response.
    """

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    model: str = field(validator=(instance_of(str), min_len(1)))
    """The LiteLLM model identifier to use for recommendations."""

    experiment_description: str = field(validator=(instance_of(str), min_len(1)))
    """Textual description of the experiment."""

    objective_description: str = field(validator=(instance_of(str), min_len(1)))
    """Textual description of the optimization objective."""

    format_instructions: str | None = field(default=None)
    """Optional custom instructions for formatting the LLM's response."""

    recovery_model: str | None = field(default=None)
    """Optional model to use for recovery attempts.

    If ``None``, uses the same model as the main recommendations.
    """

    litellm_args: dict[str, Any] = field(factory=dict, converter=dict)
    """Additional arguments to pass to LiteLLM."""

    recovery_litellm_args: dict[str, Any] | None = field(default=None)
    """Optional arguments to pass to LiteLLM during recovery attempts.

    If ``None``, uses the same arguments as the main recommendations.
    """

    @litellm_args.validator
    def _validate_litellm_args(self, attribute, value):  # noqa: DOC101, DOC103
        """Validate litellm_args does not contain reserved keys."""
        conflicts = _RESERVED_LITELLM_KEYS & set(value.keys())
        if conflicts:
            raise ValueError(
                f"'litellm_args' must not contain keys that are set explicitly: "
                f"{conflicts}. Use the dedicated class attributes instead."
            )

    @recovery_litellm_args.validator
    def _validate_recovery_litellm_args(self, attribute, value):  # noqa: DOC101, DOC103
        """Validate recovery_litellm_args does not contain reserved keys."""
        if value is None:
            return
        conflicts = _RESERVED_LITELLM_KEYS & set(value.keys())
        if conflicts:
            raise ValueError(
                f"'recovery_litellm_args' must not contain keys that are set "
                f"explicitly: {conflicts}. Use the dedicated class attributes instead."
            )

    def _construct_prompt(
        self,
        searchspace: SearchSpace,
        batch_size: int,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> str:
        """Construct the prompt for the language model.

        Args:
            searchspace: The search space to generate recommendations for.
            batch_size: The number of recommendations to generate.
            objective: Optional objective to include in the prompt.
            measurements: Optional measurements to include in the prompt.
            pending_experiments: Optional pending experiments to include in the prompt.

        Returns:
            The constructed prompt.
        """
        from baybe._optional.llm import Template

        parameters = _extract_parameter_info(searchspace.parameters)

        template = Template(
            _PROMPT_TEMPLATE,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(
            experiment_description=self.experiment_description,
            objective_description=self.objective_description,
            objective=objective,
            parameters=parameters,
            measurements=measurements,
            pending_experiments=pending_experiments,
            batch_size=batch_size,
            format_instructions=self.format_instructions,
        )

    def _attempt_recovery(
        self,
        error: Exception,
        original_response: str,
        searchspace: SearchSpace,
    ) -> pd.DataFrame:
        """Attempt to recover from a malformed LLM response by asking for correction.

        Args:
            error: The error that occurred during parsing.
            original_response: The original malformed response.
            searchspace: The search space to validate recommendations against.

        Returns:
            A DataFrame containing the corrected recommendations.

        Raises:
            LLMResponseError: If recovery fails.
        """
        from baybe._optional.llm import Template, completion

        parameters = _extract_parameter_info(searchspace.parameters)
        template = Template(
            _RECOVERY_PROMPT_TEMPLATE, trim_blocks=True, lstrip_blocks=True
        )
        recovery_prompt = template.render(
            error=str(error),
            original_response=original_response,
            parameters=parameters,
            format_instructions=self.format_instructions,
        )

        litellm_args = self.recovery_litellm_args or self.litellm_args
        try:
            response = completion(
                model=self.recovery_model or self.model,
                messages=[{"role": "user", "content": recovery_prompt}],
                **litellm_args,
            )
        except Exception as e:
            raise LLMResponseError(
                f"Recovery LLM call failed ({type(e).__name__}): {e}. "
                f"Original error: {error}"
            ) from e

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMResponseError(
                f"Recovery response had unexpected structure: {e}. "
                f"Original error: {error}"
            ) from e

        if content is None:
            raise LLMResponseError(
                f"Recovery returned empty content (None). Original error: {error}"
            )

        try:
            return self._parse_llm_response(content, searchspace)
        except LLMResponseError as e:
            raise LLMResponseError(
                f"Recovery produced another malformed response: {e}. "
                f"Original error: {error}"
            ) from e

    def _parse_llm_response(
        self, response: str, searchspace: SearchSpace
    ) -> pd.DataFrame:
        """Parse the LLM response into a DataFrame of recommendations.

        Args:
            response: The response from the language model.
            searchspace: The search space to validate recommendations against.

        Returns:
            A DataFrame containing the parsed recommendations.

        Raises:
            LLMResponseError: If the response cannot be parsed or
                contains invalid values.
        """
        payload = (
            _extract_json_array(response) if isinstance(response, str) else response
        )
        try:
            suggestions = json.loads(payload)
        except (JSONDecodeError, TypeError) as e:
            raise LLMResponseError(f"Error parsing JSON output: {e}") from e

        if not isinstance(suggestions, list):
            raise LLMResponseError("Response must be a JSON array")

        if not suggestions:
            raise LLMResponseError(
                "Response contains an empty array with no suggestions."
            )

        recommendations = []
        for suggestion in suggestions:
            if not isinstance(suggestion, dict):
                raise LLMResponseError("Each suggestion must be a JSON object")

            if "parameters" not in suggestion:
                raise LLMResponseError(
                    "Each suggestion must contain a 'parameters' field"
                )

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
                            f"Invalid values {invalid} for parameter "
                            f"'{param.name}'. "
                            f"Allowed values are: {allowed}"
                        )
                    df[param.name] = canonical
                else:
                    invalid = [v for v in values if v not in allowed]
                    if invalid:
                        raise LLMResponseError(
                            f"Invalid values {invalid} for parameter "
                            f"'{param.name}'. "
                            f"Allowed values are: {allowed}"
                        )
                    # Categorical values from JSON are strings; cast to canonical
                    allowed_map = {str(a): a for a in allowed}
                    df[param.name] = [allowed_map.get(str(v), v) for v in values]

        return df

    @override
    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate recommendations using the language model.

        Args:
            batch_size: The number of recommendations to generate.
            searchspace: The search space to generate recommendations for.
            objective: Optional objective to include in the prompt.
            measurements: Optional measurements to include in the prompt.
            pending_experiments: Optional pending experiments to include in the prompt.

        Returns:
            A DataFrame containing the recommendations as individual rows.

        Raises:
            LLMResponseError: If the call to the language model fails, or if its
                response cannot be parsed and recovery fails.
        """
        from baybe._optional.llm import completion

        if measurements is not None:
            measurements = preprocess_dataframe(
                measurements,
                searchspace,
                numerical_measurements_must_be_within_tolerance=False,
            )

        if pending_experiments is not None:
            pending_experiments = preprocess_dataframe(
                pending_experiments,
                searchspace,
                numerical_measurements_must_be_within_tolerance=False,
            )

        prompt = self._construct_prompt(
            searchspace, batch_size, objective, measurements, pending_experiments
        )
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **self.litellm_args,
            )
        except Exception as e:
            raise LLMResponseError(
                f"The call to the language model failed ({type(e).__name__}): {e}. "
                f"Check your API credentials, network connection, and the model "
                f"identifier '{self.model}'."
            ) from e

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMResponseError(
                f"LLM returned an unexpected response structure: {e}"
            ) from e

        if content is None:
            raise LLMResponseError("LLM returned empty content (None).")

        try:
            output = self._parse_llm_response(content, searchspace)
        except LLMResponseError as e:
            output = self._attempt_recovery(e, content, searchspace)

        if len(output) < batch_size:
            warnings.warn(
                f"LLM returned {len(output)} suggestions instead of the "
                f"requested {batch_size}.",
                LLMResponseWarning,
                stacklevel=2,
            )

        return output.head(batch_size)

    @override
    def __str__(self) -> str:
        fields = [
            to_string("Model", self.model, single_line=True),
            to_string("LiteLLM Args", self.litellm_args, single_line=True),
            to_string(
                "Experiment Description", self.experiment_description, single_line=True
            ),
            to_string(
                "Optimization Objective", self.objective_description, single_line=True
            ),
        ]
        return to_string(self.__class__.__name__, *fields)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
