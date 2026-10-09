"""Prompt construction for the LLM recommender."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

from baybe.exceptions import IncompatibilityError, LLMResponseError
from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.enum import Scalarizer
from baybe.objectives.pareto import ParetoObjective
from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.parameters.substance import SubstanceParameter
from baybe.recommenders.pure.llm._schema import _response_format
from baybe.searchspace import SearchSpace
from baybe.targets.binary import BinaryTarget
from baybe.targets.numerical import NumericalTarget
from baybe.transformations import IdentityTransformation

if TYPE_CHECKING:
    import pandas as pd

    from baybe.objectives.base import Objective
    from baybe.targets.base import Target

_FORBIDDEN_INSTRUCTIONS = """\
The following configurations are currently NOT selectable. You MUST NOT recommend any
of them. Treat this list as the single source of truth about what not to recommend.
Do not rely on your own memory of novelty. Before you output each suggestion, compare it
row-by-row against this list; if it matches a row, discard it and choose a genuinely
new configuration instead:\
"""

_PROMPT_TEMPLATE = """\
You are an expert experimental design assistant. Your task is to suggest new \
experimental conditions based on the following information:

EXPERIMENT DESCRIPTION:
{{ experiment_description }}

{% if objective is not none %}
OPTIMIZATION OBJECTIVE:
{% if objective.description is not none %}
{{ objective.description }}
{% endif %}
{% if objective.combination is not none %}
{{ objective.combination }}
{% endif %}

TARGETS:
{% for target in objective.targets %}
Target: {{ target.name }}
Goal: {{ target.goal }}
{% if target.transformation is not none %}
Transformation applied before optimizing: {{ target.transformation }}
{% endif %}
{% if target.description is not none %}
Description: {{ target.description }}
{% endif %}
{% if target.unit is not none %}
Unit: {{ target.unit }}
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
{%- if pending_experiments is not none %}

PENDING EXPERIMENTS:
The following experiments have already been proposed and are awaiting results.
Do not recommend these again.
{{ pending_experiments }}
{% endif %}
{%- if forbidden_configurations is not none %}

FORBIDDEN CONFIGURATIONS:
{{ forbidden_instructions }}
{{ forbidden_configurations }}
{% endif %}
{%- if recovery_instruction is not none %}

Your previous recommendation could not be used and needs to be corrected.

WHAT WENT WRONG:
{{ recovery_instruction }}

ORIGINAL RESPONSE:
{{ original_response }}

Please provide a corrected set of {{ batch_size }} experimental conditions that \
addresses the problem described above and improves the optimization objective.
{% else %}

Please suggest {{ batch_size }} new experimental conditions that are likely to \
improve the optimization objective.
{% endif %}
For each suggestion, provide:
1. A brief explanation of why you chose these values
2. The values for each parameter

Format your response as a JSON array of objects with the following structure \
(no backticks):
{{ response_format }}
"""


class _ParameterPromptInfo(TypedDict):
    """Typed, presentation-only view of a parameter for prompt rendering."""

    name: str
    description: str | None
    kind: Literal["continuous", "discrete_numeric", "categorical", "substance"]
    domain: str
    unit: str | None
    misc: tuple[tuple[str, str], ...]


class _TargetPromptInfo(TypedDict):
    """Typed, presentation-only view of an optimization target."""

    name: str
    goal: str
    transformation: str | None
    description: str | None
    unit: str | None


class _ObjectivePromptInfo(TypedDict):
    """Typed, presentation-only view of the optimization objective."""

    description: str | None
    combination: str | None
    targets: tuple[_TargetPromptInfo, ...]


class _PromptContext(TypedDict):
    """Typed render context for the prompt."""

    experiment_description: str
    objective: _ObjectivePromptInfo | None
    parameters: tuple[_ParameterPromptInfo, ...]
    measurements: str | None
    pending_experiments: str | None
    forbidden_configurations: str | None
    forbidden_instructions: str
    batch_size: int
    response_format: str
    recovery_instruction: str | None
    original_response: str | None


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
            f"{name} ({smiles})"
            for name, smiles in parameter.data.items()
            if name in parameter.active_values
        )
        domain = f"Allowed values (choose by name; SMILES in parentheses): {substances}"
    elif isinstance(parameter, DiscreteParameter):
        kind = "discrete_numeric" if parameter.is_numerical else "categorical"
        values = parameter.active_values
        if parameter.is_numerical:
            values = tuple(value.item() for value in values)  # avoid numpy-repr leak
        domain = f"Allowed values: {list(values)}"
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


def _forbidden_configurations(searchspace: SearchSpace) -> str | None:
    """Render discrete configurations that should not be recommended currently.

    Some candidates should not be recommended in the current iteration, since it might
    be forbidden to recommend points already recommended (e.g. via setting
    ``allow_recommending_already_recommended``) or because the points are currently
    pending. Hence, those points are dropped from the eligible candidate set the
    recommender receives. Surfacing them lets the model avoid proposing configurations
    that would be rejected as ineligible.

    Args:
        searchspace: The search space to recommend for.

    Returns:
        A rendered table of the forbidden discrete configurations, or ``None`` if the
        search space has no discrete part or nothing has been filtered out.
    """
    discrete = searchspace.discrete
    if not discrete.parameters:
        return None
    eligible, _ = discrete.get_candidates()
    forbidden = discrete.exp_rep[~discrete.exp_rep.index.isin(eligible.index)]
    if forbidden.empty:
        return None
    return forbidden.to_string(index=False)


def _target_prompt_info(target: Target) -> _TargetPromptInfo:
    """Build the prompt view of an optimization target.

    Args:
        target: The target to describe.

    Returns:
        A typed, presentation-only view of the target's optimization semantics.

    Raises:
        IncompatibilityError: If the target type is not supported.
    """
    transformation: str | None = None
    if isinstance(target, NumericalTarget):
        goal = "minimize" if target.minimize else "maximize"
        transformation = (
            None
            if isinstance(target.transformation, IdentityTransformation)
            else str(target.transformation)
        )
    elif isinstance(target, BinaryTarget):
        goal = (
            f"achieve the success value '{target.success_value}' "
            f"(as opposed to the failure value '{target.failure_value}')"
        )
    else:
        raise IncompatibilityError(
            f"Target '{target.name}' has unsupported type "
            f"'{type(target).__name__}'. Only '{NumericalTarget.__name__}' and "
            f"'{BinaryTarget.__name__}' are supported."
        )
    return {
        "name": target.name,
        "goal": goal,
        "transformation": transformation,
        "description": target.description,
        "unit": target.unit,
    }


def _objective_prompt_info(objective: Objective) -> _ObjectivePromptInfo:
    """Build the prompt view of the optimization objective.

    Args:
        objective: The objective to describe.

    Returns:
        A typed, presentation-only view: the objective description, how its targets are
        combined, and the per-target views.
    """
    targets = tuple(_target_prompt_info(t) for t in objective.targets)
    combination: str | None
    if isinstance(objective, DesirabilityObjective):
        aggregation = (
            "weighted geometric mean"
            if objective.scalarizer is Scalarizer.GEOM_MEAN
            else "weighted arithmetic mean"
        )
        weights = ", ".join(
            f"{t['name']}={w:.3g}"
            for t, w in zip(targets, objective.normalized_weights)
        )
        combination = (
            f"The targets are aggregated into a single desirability score via a "
            f"{aggregation} (weights: {weights}), which is then maximized."
        )
    elif isinstance(objective, ParetoObjective):
        combination = (
            "The targets are optimized jointly as a multi-objective (Pareto) problem: "
            "they are not combined into a single score; seek the best trade-offs, "
            "optimizing each target in its stated direction."
        )
    else:
        combination = None
    return {
        "description": objective.metadata.description,
        "combination": combination,
        "targets": targets,
    }


def make_prompt(
    batch_size: int,
    searchspace: SearchSpace,
    objective: Objective | None = None,
    measurements: pd.DataFrame | None = None,
    pending_experiments: pd.DataFrame | None = None,
    *,
    experiment_description: str,
    error: LLMResponseError | None = None,
    original_response: str | None = None,
) -> str:
    """Construct the prompt for the language model.

    The recommendation-context arguments follow the canonical order used by
    :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`; the LLM-specific
    ``experiment_description`` is keyword-only.

    When ``error`` and ``original_response`` are provided, the prompt is rendered in
    *recovery* mode: it carries the exact same context as the original query and appends
    a correction section asking the model to fix its previous response. Because each
    model call is stateless, carrying the full context here is what lets the recovery
    attempt reason as well as the original one.

    Args:
        batch_size: The number of recommendations to generate.
        searchspace: The search space to generate recommendations for.
        objective: Optional objective to include in the prompt. Set
            :attr:`baybe.objectives.base.Objective.metadata` to provide the
            language model with a description of what to optimize and per-target
            context such as units and descriptions.
        measurements: Optional measurements to include in the prompt.
        pending_experiments: Optional pending experiments to include in the prompt.
        experiment_description: Textual description of the experiment.
        error: If set, renders a recovery prompt using the error's
            :attr:`~baybe.exceptions.LLMResponseError.recovery_instruction`. Must be
            provided together with ``original_response``.
        original_response: The previous, rejected response to be corrected. Must be
            provided together with ``error``.

    Raises:
        ValueError: If exactly one of ``error`` and ``original_response`` is provided.

    Returns:
        The constructed prompt.
    """
    if (error is None) != (original_response is None):
        raise ValueError("'error' and 'original_response' must be provided together.")

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
        "objective": (
            _objective_prompt_info(objective) if objective is not None else None
        ),
        "parameters": tuple(_parameter_prompt_info(p) for p in searchspace.parameters),
        "measurements": measurements_text,
        "pending_experiments": pending_text,
        "forbidden_configurations": _forbidden_configurations(searchspace),
        "forbidden_instructions": _FORBIDDEN_INSTRUCTIONS,
        "batch_size": batch_size,
        "response_format": _response_format(batch_size),
        "recovery_instruction": (
            error.recovery_instruction if error is not None else None
        ),
        "original_response": original_response,
    }
    template = Template(
        _PROMPT_TEMPLATE,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    return template.render(context)
