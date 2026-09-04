"""Prompt construction for the LLM recommender."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

from baybe.exceptions import IncompatibilityError, LLMResponseError
from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.parameters.substance import SubstanceParameter
from baybe.recommenders.pure.llm._schema import _response_format
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
Type: {{ param.kind }}
{{ param.domain }}
{% if param.unit is not none %}
Unit: {{ param.unit }}
{% endif %}
{% for key, value in param.misc %}
{{ key }}: {{ value }}
{% endfor %}

{% endfor %}

{% if measurements is not none %}
PREVIOUS MEASUREMENTS:
{{ measurements }}
{% endif %}

{% if pending_experiments is not none %}
PENDING EXPERIMENTS:
The following experiments have already been proposed and are awaiting results.
Do not recommend these again.
{{ pending_experiments }}
{% endif %}

Please suggest {{ batch_size }} new experimental conditions that are likely to \
improve the optimization objective.
For each suggestion, provide:
1. A brief explanation of why you chose these values
2. The values for each parameter

Format your response as a JSON array of objects with the following structure \
(no backticks):
{{ response_format }}
"""

_RECOVERY_PROMPT_TEMPLATE = """\
Your previous recommendation could not be used and needs to be corrected.

WHAT WENT WRONG:
{{ recovery_instruction }}

ORIGINAL RESPONSE:
{{ original_response }}

PARAMETERS:
{% for param in parameters %}
Parameter: {{ param.name }}
Type: {{ param.kind }}
{{ param.domain }}
{% endfor %}

Please provide a corrected JSON response that follows the required format:
{{ response_format }}\
"""


class _ParameterPromptInfo(TypedDict):
    """Typed, presentation-only view of a parameter for prompt rendering.

    Gives the template a single stable shape (instead of the previous
    dynamically-shaped ``SimpleNamespace``); the value domain is flattened into one
    ``domain`` string so no field is conditionally present.
    """

    name: str
    description: str | None
    kind: Literal["continuous", "discrete_numeric", "categorical", "substance"]
    domain: str
    unit: str | None
    misc: tuple[tuple[str, str], ...]


class _PromptContext(TypedDict):
    """Typed render context for the main prompt."""

    experiment_description: str
    objective: Objective | None
    parameters: tuple[_ParameterPromptInfo, ...]
    measurements: str | None
    pending_experiments: str | None
    batch_size: int
    response_format: str


class _RecoveryPromptContext(TypedDict):
    """Typed render context for the recovery prompt."""

    parameters: tuple[_ParameterPromptInfo, ...]
    recovery_instruction: str
    original_response: str
    response_format: str


def _parameter_prompt_info(parameter: Parameter) -> _ParameterPromptInfo:
    """Build the prompt view of a parameter.

    Args:
        parameter: The parameter to describe.

    Returns:
        A typed, presentation-only view consumed by the prompt template.

    Raises:
        IncompatibilityError: If the parameter type is not supported.
    """
    if isinstance(parameter, NumericalContinuousParameter):
        lower, upper = parameter.bounds.to_tuple()
        kind: Literal["continuous", "discrete_numeric", "categorical", "substance"] = (
            "continuous"
        )
        domain = f"Bounds: [{lower}, {upper}]"
    elif isinstance(parameter, SubstanceParameter):
        # Substances are chemical compounds: expose their SMILES so the model can
        # reason about structure, while still choosing by substance name.
        kind = "substance"
        substances = ", ".join(
            f"{name} ({smiles})" for name, smiles in parameter.data.items()
        )
        domain = f"Allowed values (choose by name; SMILES in parentheses): {substances}"
    elif isinstance(parameter, DiscreteParameter):
        kind = "discrete_numeric" if parameter.is_numerical else "categorical"
        domain = f"Allowed values: {list(parameter.values)}"
    else:
        raise IncompatibilityError(
            f"Parameter '{parameter.name}' has unsupported type "
            f"'{type(parameter).__name__}'. Only "
            f"'{NumericalContinuousParameter.__name__}' and "
            f"'{DiscreteParameter.__name__}' subclasses are supported."
        )
    return {
        "name": parameter.name,
        "description": parameter.description,
        "kind": kind,
        "domain": domain,
        "unit": parameter.unit,
        "misc": tuple(
            (key, str(value)) for key, value in parameter.metadata.misc.items()
        ),
    }


def make_prompt(
    searchspace: SearchSpace,
    *,
    batch_size: int,
    experiment_description: str,
    objective: Objective | None,
    measurements: pd.DataFrame | None,
    pending_experiments: pd.DataFrame | None,
) -> str:
    """Construct the main prompt for the language model.

    Args:
        searchspace: The search space to generate recommendations for.
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
    from baybe._optional.llm import StrictUndefined, Template

    measurements_text = (
        measurements.to_string(index=False)
        if measurements is not None and not measurements.empty
        else None
    )
    pending_text = (
        pending_experiments.to_string(index=False)
        if pending_experiments is not None and not pending_experiments.empty
        else None
    )
    context: _PromptContext = {
        "experiment_description": experiment_description,
        "objective": objective,
        "parameters": tuple(_parameter_prompt_info(p) for p in searchspace.parameters),
        "measurements": measurements_text,
        "pending_experiments": pending_text,
        "batch_size": batch_size,
        "response_format": _response_format(),
    }
    template = Template(
        _PROMPT_TEMPLATE,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    return template.render(context)


def make_recovery_prompt(
    searchspace: SearchSpace,
    *,
    error: LLMResponseError,
    original_response: str,
) -> str:
    """Construct the recovery prompt asking the model to correct its response.

    Args:
        searchspace: The search space to generate recommendations for.
        error: The error that occurred while processing the previous response. Its
            :attr:`~baybe.exceptions.LLMResponseError.recovery_instruction` provides the
            error-specific guidance embedded in the prompt.
        original_response: The original response that could not be used.

    Returns:
        The constructed recovery prompt.
    """
    from baybe._optional.llm import StrictUndefined, Template

    context: _RecoveryPromptContext = {
        "parameters": tuple(_parameter_prompt_info(p) for p in searchspace.parameters),
        "recovery_instruction": error.recovery_instruction,
        "original_response": original_response,
        "response_format": _response_format(),
    }
    template = Template(
        _RECOVERY_PROMPT_TEMPLATE,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    return template.render(context)
