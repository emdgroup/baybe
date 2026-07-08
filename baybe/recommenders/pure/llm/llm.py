"""LLM-based recommender for experimental design."""

from __future__ import annotations

import gc
import json
import warnings
from collections.abc import Callable
from json import JSONDecodeError
from types import SimpleNamespace
from typing import Any, ClassVar

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of, min_len
from typing_extensions import override

from baybe.exceptions import IncompatibilityError, LLMResponseError
from baybe.objectives.base import Objective
from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType
from baybe.utils.conversion import to_string

_PROMPT_TEMPLATE = """\
You are an expert experimental design assistant. Your task is to suggest new \
experimental conditions based on the following information:

EXPERIMENT DESCRIPTION:
{{ experiment_description }}

OPTIMIZATION OBJECTIVE:
{{ objective_description }}

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

{% if related_data is not none and not related_data.empty %}
RELATED DATA:
Here is data from other optimization campaigns.
It might be useful to learn from these experiments or not.
Use it as you see fit.
{{ related_data.to_string(index=False) }}
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


@define(slots=False)
class LLMRecommender(NonPredictiveRecommender):
    """Recommender that uses a language model to suggest new experimental points."""

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

    litellm_args: dict[str, Any] = field(factory=dict)
    """Additional arguments to pass to LiteLLM."""

    recovery_litellm_args: dict[str, Any] | None = field(default=None)
    """Optional arguments to pass to LiteLLM during recovery attempts.

    If ``None``, uses the same arguments as the main recommendations.
    """

    related_data: pd.DataFrame | None = field(default=None)
    """Optional DataFrame containing data from similar optimization campaigns.

    This data can be used to inform the recommendations by learning from
    similar experiments.
    """

    is_feasible_experiment: Callable[[pd.Series], bool] | None = field(default=None)
    """Optional function to check if an experiment is feasible.

    If provided, the recommender will request additional experiments and
    filter for feasibility. Only feasible experiments are returned.
    """

    overflow_experiments: int = field(default=0)
    """Number of additional experiments to request from the LLM.

    The LLM will be asked to generate ``batch_size + overflow_experiments``
    experiments. After filtering for feasibility, the first ``batch_size``
    feasible experiments will be returned.
    """

    def _construct_prompt(
        self,
        searchspace: SearchSpace,
        batch_size: int,
        measurements: pd.DataFrame | None = None,
    ) -> str:
        """Construct the prompt for the language model.

        Args:
            searchspace: The search space to generate recommendations for.
            batch_size: The number of recommendations to generate.
            measurements: Optional measurements to include in the prompt.

        Returns:
            The constructed prompt.
        """
        from baybe._optional.llm import Template

        total_experiments = batch_size + self.overflow_experiments
        parameters = _extract_parameter_info(searchspace.parameters)

        template = Template(
            _PROMPT_TEMPLATE,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(
            experiment_description=self.experiment_description,
            objective_description=self.objective_description,
            parameters=parameters,
            measurements=measurements,
            related_data=self.related_data,
            batch_size=total_experiments,
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
        try:
            suggestions = json.loads(response)
        except (JSONDecodeError, TypeError) as e:
            raise LLMResponseError(f"Error parsing JSON output: {e}") from e

        if not isinstance(suggestions, list):
            raise LLMResponseError("Response must be a JSON array")

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
                if not all(isinstance(v, (int, float)) for v in values):
                    raise LLMResponseError(
                        f"Non-numeric values for continuous parameter: {param.name}"
                    )
                bounds = param.bounds.to_tuple()
                min_val, max_val = bounds
                if not all(min_val <= v <= max_val for v in values):
                    raise LLMResponseError(
                        f"Values for {param.name} outside bounds [{min_val}, {max_val}]"
                    )

            elif isinstance(param, DiscreteParameter):
                allowed = list(param.values)
                invalid = [v for v in values if v not in allowed]
                if invalid:
                    raise LLMResponseError(
                        f"Invalid values {invalid} for parameter '{param.name}'. "
                        f"Allowed values are: {allowed}"
                    )

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
            objective: Optional objective to optimize for.
            measurements: Optional measurements to include in the prompt.
            pending_experiments: Optional pending experiments to consider.

        Returns:
            A DataFrame containing the recommendations as individual rows.

        Raises:
            LLMResponseError: If the LLM response cannot be parsed or
                recovery fails.
        """
        from baybe._optional.llm import completion

        prompt = self._construct_prompt(searchspace, batch_size, measurements)
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self.litellm_args,
        )

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

        if len(output) < batch_size + self.overflow_experiments:
            warnings.warn(
                f"LLM returned {len(output)} suggestions instead of the "
                f"requested {batch_size + self.overflow_experiments}.",
                stacklevel=2,
            )

        if self.is_feasible_experiment is not None:
            feasible_mask = output.apply(self.is_feasible_experiment, axis=1)
            feasible_experiments = output[feasible_mask]

            if len(feasible_experiments) < batch_size:
                warnings.warn(
                    f"Only {len(feasible_experiments)} of {batch_size} requested "
                    f"experiments passed the feasibility check. Consider increasing "
                    f"overflow_experiments (currently {self.overflow_experiments}).",
                    stacklevel=2,
                )

            return feasible_experiments.head(batch_size)

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
            to_string("Related Data", self.related_data, single_line=True),
            to_string(
                "Overflow Experiments", self.overflow_experiments, single_line=True
            ),
            to_string(
                "Feasibility Check",
                "Enabled" if self.is_feasible_experiment is not None else "Disabled",
                single_line=True,
            ),
        ]
        return to_string(self.__class__.__name__, *fields)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
