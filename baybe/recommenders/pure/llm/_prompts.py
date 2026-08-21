"""Prompt construction for the LLM recommender."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from attrs import asdict as attrs_asdict

from baybe.exceptions import IncompatibilityError
from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.searchspace import SearchSpace

if TYPE_CHECKING:
    import pandas as pd

    from baybe.objectives.base import Objective

_PROMPT_TEMPLATE = """\
You are an expert experimental design assistant. Your task is to suggest new \
experimental conditions based on the following information:

EXPERIMENT DESCRIPTION:
{{ experiment_description }}

{% if objective is not none %}
{% if objective.metadata.description is not none %}
OPTIMIZATION OBJECTIVE:
{{ objective.metadata.description }}

{% endif %}
OPTIMIZATION TARGETS:
{% for target in objective.targets %}
Target: {{ target.name }}
{% if target.metadata.description is not none %}
Description: {{ target.metadata.description }}
{% endif %}
{% if target.metadata.unit is not none %}
Unit: {{ target.metadata.unit }}
{% endif %}

{% endfor %}
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
{% if param.misc %}
{% for key, value in param.misc.items() %}
{{ key }}: {{ value }}
{% endfor %}
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
]\
"""


def _extract_parameter_info(
    parameters: tuple[Parameter, ...], recommender_name: str
) -> list[SimpleNamespace]:
    """Extract parameter information for prompt construction.

    Args:
        parameters: The parameters from the search space.
        recommender_name: The name of the recommender, used in error messages.

    Returns:
        A list of namespace objects containing parameter information.

    Raises:
        IncompatibilityError: If a parameter type is not supported.
    """
    infos = []
    for param in parameters:
        info: dict[str, Any] = {
            "name": param.name,
            **attrs_asdict(param.metadata),
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
                f"'{recommender_name}'. Only "
                f"'{NumericalContinuousParameter.__name__}' and "
                f"'{DiscreteParameter.__name__}' subclasses are supported."
            )

        infos.append(SimpleNamespace(**info))

    return infos


def build_prompt(
    searchspace: SearchSpace,
    *,
    recommender_name: str,
    batch_size: int,
    experiment_description: str,
    objective: Objective | None,
    measurements: pd.DataFrame | None,
    pending_experiments: pd.DataFrame | None,
) -> str:
    """Construct the main prompt for the language model.

    Args:
        searchspace: The search space to generate recommendations for.
        recommender_name: The name of the recommender, used in error messages.
        batch_size: The number of recommendations to generate.
        experiment_description: Textual description of the experiment.
        objective: Optional objective to include in the prompt. Set
            :attr:`baybe.objectives.base.Objective.metadata` to provide the
            language model with a description of what to optimize and per-target
            context such as units and descriptions.
        measurements: Optional measurements to include in the prompt.
        pending_experiments: Optional pending experiments to include in the prompt.

    Returns:
        The constructed prompt.
    """
    from baybe._optional.llm import Template

    parameters = _extract_parameter_info(searchspace.parameters, recommender_name)
    template = Template(_PROMPT_TEMPLATE, trim_blocks=True, lstrip_blocks=True)
    return template.render(
        experiment_description=experiment_description,
        objective=objective,
        parameters=parameters,
        measurements=measurements,
        pending_experiments=pending_experiments,
        batch_size=batch_size,
    )


def build_recovery_prompt(
    searchspace: SearchSpace,
    *,
    recommender_name: str,
    error: Exception,
    original_response: str,
) -> str:
    """Construct the recovery prompt asking the model to correct a malformed response.

    Args:
        searchspace: The search space to generate recommendations for.
        recommender_name: The name of the recommender, used in error messages.
        error: The error that occurred during parsing.
        original_response: The original malformed response.

    Returns:
        The constructed recovery prompt.
    """
    from baybe._optional.llm import Template

    parameters = _extract_parameter_info(searchspace.parameters, recommender_name)
    template = Template(_RECOVERY_PROMPT_TEMPLATE, trim_blocks=True, lstrip_blocks=True)
    return template.render(
        error=str(error),
        original_response=original_response,
        parameters=parameters,
    )
