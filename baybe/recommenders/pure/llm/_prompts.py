"""Prompt construction for the LLM recommender."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, TypedDict

import pandas as pd

from baybe._llm.exceptions import LLMResponseError
from baybe._llm.rendering import to_json_lines, to_prompt_dict
from baybe.recommenders.pure.llm._schema import _response_format
from baybe.searchspace import SearchSpace
from baybe.transformations.base import Transformation
from baybe.utils.basic import get_subclasses

if TYPE_CHECKING:
    from baybe.objectives.base import Objective

# The number of configurations sampled from the search space as examples
_N_EXAMPLE_CONFIGURATIONS = 3

_PROMPT_TEMPLATE = """\
You are an expert experimental design assistant. Your task is to recommend new \
experimental conditions based on the following information. Objects are given in \
JSON format, one object per line.

BATCH REQUIREMENTS:
Provide exactly {{ batch_size }} distinct recommendations, no more and no fewer.
{% if batch_size > 1 %}
Choose the recommendations jointly as a diversified batch, rather than \
independently selecting each experiment. Avoid redundant experiments while \
respecting all constraints.
{% endif %}

EXPERIMENT DESCRIPTION:
{{ experiment_description }}

{% if objective is not none %}
OPTIMIZATION OBJECTIVE:
Each target is optimized by maximizing its transformed value (or minimizing it if \
"minimize" is true). "constructor_info", if present, shows how the target was defined. \
Binary targets are optimized by obtaining their "success_value" as often as possible.
{{ objective }}
{% if transformations is not none %}

TRANSFORMATIONS:
The transformations used by the targets are defined as follows.
{{ transformations }}
{% endif %}

{% endif %}
PARAMETERS:
{{ parameters }}
{% if constraints is not none %}

CONSTRAINTS:
All recommendations must satisfy the following constraints.
{{ constraints }}
{% endif %}

EXAMPLE CONFIGURATIONS:
The following configurations are examples from the search space. Follow these \
examples when making new recommendations, in particular regarding the format and \
representation of the parameter values.
{{ example_configurations }}
{% if measurements is not none %}

PREVIOUS MEASUREMENTS:
{{ measurements }}
{% endif %}
{%- if pending_experiments is not none %}

PENDING EXPERIMENTS:
The following experiments have already been proposed and are awaiting results.
Take them into account when generating your recommendations.
{{ pending_experiments }}
{% endif %}
{%- if recovery_instruction is not none %}

Your previous recommendation could not be used and needs to be corrected.

WHAT WENT WRONG:
{{ recovery_instruction }}
{% if original_response is not none %}

ORIGINAL RESPONSE:
{{ original_response }}
{% endif %}

Provide a corrected set of exactly {{ batch_size }} distinct experimental conditions \
that addresses the problem described above and improves the optimization objective.
{% else %}

Recommend exactly {{ batch_size }} distinct new experimental conditions that are \
likely to improve the optimization objective.
{% endif %}
For each recommendation, provide:
1. A brief explanation of why you chose these values
2. The values for each parameter

Format your response as a JSON array of objects with the following structure \
(no backticks):
{{ response_format }}
"""


class _PromptContext(TypedDict):
    """Typed render context for the prompt."""

    batch_size: int
    constraints: str | None
    example_configurations: str
    experiment_description: str
    measurements: str | None
    objective: str | None
    original_response: str | None
    parameters: str
    pending_experiments: str | None
    recovery_instruction: str | None
    response_format: str
    transformations: str | None


def _describe_transformations(objective_json: str, /) -> str | None:
    """Describe the transformations used in an objective via their docstrings.

    Args:
        objective_json: The objective as rendered in the prompt.

    Returns:
        One description per used transformation type, or ``None`` if the objective
        uses no transformations.
    """
    # Matching the quoted type name avoids partial matches, e.g. of
    # "AffineTransformation" within "TwoSidedAffineTransformation"
    used = [
        t
        for t in get_subclasses(Transformation)
        if f'"type": "{t.__name__}"' in objective_json
    ]
    if not used:
        return None
    return "\n\n".join(f"{t.__name__}: {inspect.getdoc(t)}" for t in used)


def _sample_example_configurations(searchspace: SearchSpace, /) -> pd.DataFrame:
    """Sample configurations from the search space to serve as examples in the prompt.

    The discrete part is drawn from the currently selectable candidates and the
    continuous part uniformly from the continuous subspace, respecting its constraints.

    Args:
        searchspace: The search space to sample from.

    Returns:
        Up to :data:`_N_EXAMPLE_CONFIGURATIONS` configurations. Fewer are returned if
        fewer discrete candidates are selectable.
    """
    n = _N_EXAMPLE_CONFIGURATIONS
    if searchspace.discrete.is_empty:
        return searchspace.continuous.sample_uniform(n)

    candidates, _ = searchspace.discrete.get_candidates()
    discrete = candidates.sample(min(n, len(candidates))).reset_index(drop=True)
    continuous = searchspace.continuous.sample_uniform(len(discrete))
    return pd.concat([discrete, continuous], axis=1)


def make_prompt(
    batch_size: int,
    searchspace: SearchSpace,
    objective: Objective | None = None,
    measurements: pd.DataFrame | None = None,
    pending_experiments: pd.DataFrame | None = None,
    *,
    experiment_description: str,
    recovery: tuple[LLMResponseError, str | None] | None = None,
) -> str:
    """Construct the prompt for the language model.

    The recommendation-context arguments follow the canonical order used by
    :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`; the LLM-specific
    ``experiment_description`` is keyword-only.

    The objective, parameters and constraints are included in their serialized form,
    and the transformations used by the objective are described via their docstrings.
    Measurements and pending experiments are restricted to their parameter and target
    columns. In addition, a few configurations sampled from the search space are
    included as examples (see :func:`_sample_example_configurations`).

    When ``recovery`` is provided, the prompt is rendered in *recovery* mode: it
    carries the exact same context as the original query and appends a correction
    section asking the model to fix its previous response. Because each model call is
    stateless, carrying the full context here is what lets the recovery attempt reason
    as well as the original one.

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
        recovery: Optional ``(error, original_response)`` tuple to render a recovery
            prompt. The ``error`` provides the
            ``LLMResponseError.recovery_instruction``, and
            ``original_response`` is the previous, rejected response to be corrected
            (may be ``None`` if the model returned no content).

    Returns:
        The constructed prompt.
    """
    error, original_response = recovery if recovery is not None else (None, None)

    from baybe._optional.llm import StrictUndefined, Template

    parameter_names = [p.name for p in searchspace.parameters]
    target_names = [t.name for t in objective.targets] if objective is not None else []
    constraints = searchspace.constraints
    objective_json = (
        to_json_lines([to_prompt_dict(objective)]) if objective is not None else None
    )
    context: _PromptContext = {
        "experiment_description": experiment_description,
        "objective": objective_json,
        "transformations": (
            _describe_transformations(objective_json)
            if objective_json is not None
            else None
        ),
        "parameters": to_json_lines(to_prompt_dict(p) for p in searchspace.parameters),
        "constraints": (
            to_json_lines(to_prompt_dict(c) for c in constraints)
            if constraints
            else None
        ),
        "example_configurations": _sample_example_configurations(searchspace).to_string(
            index=False
        ),
        # NOTE: Measurements and pending experiments are the only information about
        #   earlier experiments in the prompt. Configurations that are currently not
        #   selectable for other reasons, e.g. excluded ones or ones recommended by
        #   another recommender but not yet measured, are not shown upfront. Proposing
        #   them is only caught by the search space validation of the response, which
        #   reports them to the model for the recovery attempt.
        "measurements": (
            measurements[parameter_names + target_names].to_string(index=False)
            if measurements is not None and not measurements.empty
            else None
        ),
        "pending_experiments": (
            pending_experiments[parameter_names].to_string(index=False)
            if pending_experiments is not None and not pending_experiments.empty
            else None
        ),
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
