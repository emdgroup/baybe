"""LLM-based recommender for experimental design."""

import json
from enum import Enum
from json import JSONDecodeError

import pandas as pd
from attrs import define, field
from jinja2 import Template
from litellm import completion
from typing_extensions import override

from baybe.exceptions import LLMResponseError
from baybe.objectives.base import Objective
from baybe.recommenders.base import RecommenderProtocol
from baybe.searchspace import SearchSpace
from baybe.utils.conversion import to_string

PROMPT = Template(
    """You are an expert experimental design assistant. Your task is to suggest new experimental conditions based on the following information:

EXPERIMENT DESCRIPTION:
{{ experiment_description }}

OPTIMIZATION OBJECTIVE:
{{ objective_description }}

PARAMETERS:
{% for param in parameter_descriptions %}
Parameter: {{ param.name }}
Description: {{ param.description }}
Type: {{ param.type.value }}
{% if param.bounds is not none %}
{% if param.type.value == 'continuous' %}
Bounds: [{{ param.bounds[0] }}, {{ param.bounds[1] }}]
{% else %}
Allowed values: {{ param.bounds }}
{% endif %}
{% endif %}
{% if param.unit is not none %}
Unit: {{ param.unit }}
{% endif %}
{% if param.default_value is not none %}
Default value: {{ param.default_value }}
{% endif %}
{% if param.constraints is not none %}
Constraints: {{ param.constraints }}
{% endif %}

{% endfor %}

{% if measurements is not none and not measurements.empty %}
PREVIOUS MEASUREMENTS:
{{ measurements.to_string() }}
{% endif %}

Please suggest {{ batch_size }} new experimental conditions that are likely to improve the optimization objective.
For each suggestion, provide:
1. A brief explanation of why you chose these values
2. The values for each parameter

{% if format_instructions is not none %}
{{ format_instructions }}
{% else %}
Format your response as a JSON array of objects with the following structure (no backticks):
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
""",  # noqa: E501, W293
    trim_blocks=True,
    lstrip_blocks=True,
)

RECOVERY_PROMPT = Template(
    """The previous response was malformed and could not be parsed as JSON. Please correct the response to match the required format.

ERROR: {{ error }}

ORIGINAL RESPONSE:
{{ original_response }}

PARAMETERS:
{% for param in parameter_descriptions %}
Parameter: {{ param.name }}
Type: {{ param.type.value }}
{% if param.bounds is not none %}
{% if param.type.value == 'continuous' %}
Bounds: [{{ param.bounds[0] }}, {{ param.bounds[1] }}]
{% else %}
Allowed values: {{ param.bounds }}
{% endif %}
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
{% endif %}""",  # noqa: E501, W293
    trim_blocks=True,
    lstrip_blocks=True,
)


class ParameterType(Enum):
    """Types of parameters that can be optimized."""

    CONTINUOUS = "continuous"
    """Continuous parameter with numeric bounds."""

    DISCRETE_NUMERIC = "discrete_numeric"
    """Discrete parameter with numeric values."""

    DISCRETE_CATEGORICAL = "discrete_categorical"
    """Discrete parameter with categorical values."""

    BINARY = "binary"
    """Binary parameter (True/False)."""


@define
class ParameterDescription:
    """Description of a parameter for LLM-based optimization."""

    name: str = field()
    """Name of the parameter."""

    description: str = field()
    """Human-readable description of what the parameter represents."""

    type: ParameterType = field()
    """Type of the parameter."""

    bounds: tuple[float, float] | list[float | str] | None = field(default=None)
    """Bounds or allowed values for the parameter.

    For continuous parameters: tuple of (min, max)
    For discrete parameters: list of allowed values
    For binary parameters: None
    """

    unit: str | None = field(default=None)
    """Unit of measurement for the parameter (e.g., '°C', 'bar', 'mol/L')."""

    default_value: float | str | bool | None = field(default=None)
    """Default value for the parameter if known."""

    constraints: str | None = field(default=None)
    """Additional constraints or considerations for this parameter."""

    def __str__(self) -> str:
        """Return a string representation of the parameter description."""
        parts = [
            f"Name: {self.name}",
            f"Description: {self.description}",
            f"Type: {self.type.value}",
        ]

        if self.bounds is not None:
            if isinstance(self.bounds, tuple):
                parts.append(f"Bounds: [{self.bounds[0]}, {self.bounds[1]}]")
            else:
                parts.append(f"Allowed values: {self.bounds}")

        if self.unit is not None:
            parts.append(f"Unit: {self.unit}")

        if self.default_value is not None:
            parts.append(f"Default: {self.default_value}")

        if self.constraints is not None:
            parts.append(f"Constraints: {self.constraints}")

        return "\n".join(parts)


@define
class LLMRecommender(RecommenderProtocol):
    """Recommender that uses a language model to suggest new experimental points."""

    # Object variables
    model: str = field()
    """The LiteLLM model to use for recommendations."""

    experiment_description: str = field()
    """Textual description of the experiment."""

    objective_description: str = field()
    """Textual description of the optimization objective."""

    parameter_descriptions: list[ParameterDescription] = field(factory=list)
    """List of parameter descriptions."""

    format_instructions: str | None = field(default=None)
    """Optional custom instructions for formatting the LLM's response."""

    recovery_model: str | None = field(default=None)
    """Optional model to use for recovery attempts.

    If None, uses the same model as the main recommendations.
    """

    litellm_args: dict = field(factory=dict)
    """Additional arguments to pass to LiteLLM."""

    recovery_litellm_args: dict | None = field(default=None)
    """Optional arguments to pass to LiteLLM during recovery attempts.

    If None, uses the same arguments as the main recommendations.
    If provided, these arguments will override the main litellm_args
    for recovery attempts.
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
        return PROMPT.render(
            experiment_description=self.experiment_description,
            objective_description=self.objective_description,
            parameter_descriptions=self.parameter_descriptions,
            measurements=measurements,
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
        recovery_prompt = RECOVERY_PROMPT.render(
            error=str(error),
            original_response=original_response,
            parameter_descriptions=self.parameter_descriptions,
            format_instructions=self.format_instructions,
        )

        try:
            # Use recovery-specific args if provided, otherwise fall back to main args
            litellm_args = self.recovery_litellm_args or self.litellm_args
            response = completion(
                model=self.recovery_model or self.model,
                messages=[{"role": "user", "content": recovery_prompt}],
                **litellm_args,
            )
            return self._parse_llm_response(
                response.choices[0].message.content, searchspace
            )
        except Exception as e:
            raise LLMResponseError(
                f"Failed to recover from malformed response: {str(e)}"
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
        # Parse the JSON response
        try:
            suggestions = json.loads(response)
        except JSONDecodeError as e:
            raise LLMResponseError(f"Error parsing JSON output: {e}")
        if not isinstance(suggestions, list):
            raise LLMResponseError("Response must be a JSON array")

        # Extract parameter values from each suggestion
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

            # Extract parameter values
            params = suggestion["parameters"]
            if not isinstance(params, dict):
                raise LLMResponseError("Parameters must be a JSON object")

            # Validate parameter names
            param_names = {p.name for p in self.parameter_descriptions}
            if not all(name in param_names for name in params.keys()):
                raise LLMResponseError("Response contains unknown parameter names")

            # Add to recommendations
            recommendations.append(params)

        # Convert to DataFrame
        df = pd.DataFrame(recommendations)

        # Validate against search space
        for param in self.parameter_descriptions:
            if param.name not in df.columns:
                raise LLMResponseError(f"Missing parameter: {param.name}")

            values = df[param.name]

            # Validate based on parameter type
            if param.type == ParameterType.CONTINUOUS:
                if not all(isinstance(v, (int, float)) for v in values):
                    raise LLMResponseError(
                        f"Non-numeric values for continuous parameter: {param.name}"
                    )
                if param.bounds is not None:
                    min_val, max_val = param.bounds
                    if not all(min_val <= v <= max_val for v in values):
                        raise LLMResponseError(
                            f"Values for {param.name} outside bounds "
                            f"[{min_val}, {max_val}]"
                        )

            elif param.type == ParameterType.DISCRETE_NUMERIC:
                if not all(isinstance(v, (int, float)) for v in values):
                    raise LLMResponseError(
                        f"Non-numeric values for discrete numeric parameter:"
                        f" {param.name}"
                    )
                if param.bounds is not None:
                    if not all(v in param.bounds for v in values):
                        raise LLMResponseError(
                            f"Invalid values for discrete parameter: {param.name}"
                        )

            elif param.type == ParameterType.DISCRETE_CATEGORICAL:
                if param.bounds is not None:
                    if not all(v in param.bounds for v in values):
                        raise LLMResponseError(
                            f"Invalid values for categorical parameter: {param.name}"
                        )

            elif param.type == ParameterType.BINARY:
                if not all(isinstance(v, bool) for v in values):
                    raise LLMResponseError(
                        f"Non-boolean values for binary parameter: {param.name}"
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
        """
        prompt = self._construct_prompt(searchspace, batch_size, measurements)
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self.litellm_args,
        )
        try:
            output = self._parse_llm_response(
                response.choices[0].message.content, searchspace
            )
        except LLMResponseError as e:
            # Attempt to recover from malformed response
            output = self._attempt_recovery(
                e, response.choices[0].message.content, searchspace
            )
        return output

    def __str__(self) -> str:
        fields = [
            to_string("Model", self.model, single_line=True),
            to_string("LiteLLM Args", self.litellm_args, single_line=True),
            to_string(
                "Experiment Description", self.experiment_description, single_line=True
            ),
            to_string(
                "Optimization Objectives", self.objective_description, single_line=True
            ),
            to_string(
                "Parameter Descriptions", self.parameter_descriptions, single_line=True
            ),
        ]
        return to_string(self.__class__.__name__, *fields)
