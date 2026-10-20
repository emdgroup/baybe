"""Prompt construction for the LLM recommender."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypedDict

from attrs import fields

from baybe.exceptions import LLMResponseError
from baybe.parameters.base import _DiscreteLabelLikeParameter
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.custom import CustomDiscreteParameter
from baybe.parameters.substance import SubstanceParameter
from baybe.recommenders.pure.llm._schema import _response_format
from baybe.searchspace import SearchSpace

if TYPE_CHECKING:
    import pandas as pd

    from baybe.constraints.base import Constraint
    from baybe.objectives.base import Objective
    from baybe.parameters.base import Parameter

_FORBIDDEN_INSTRUCTIONS = """\
The following configurations are currently NOT selectable. You MUST NOT recommend any
of them. Treat this list as the single source of truth about what not to recommend.
Do not rely on your own memory of novelty. Before you output each recommendation,
compare it row-by-row against this list; if it matches a row, discard it and choose a
genuinely new configuration instead:\
"""

_PROMPT_TEMPLATE = """\
You are an expert experimental design assistant. Your task is to recommend new \
experimental conditions based on the following information. Objects are given in \
JSON format, one object per line.

EXPERIMENT DESCRIPTION:
{{ experiment_description }}

{% if objective is not none %}
OPTIMIZATION OBJECTIVE:
Each target is optimized by maximizing its transformed value (or minimizing it if \
"minimize" is true); "constructor_info", if present, shows how the target was defined, \
e.g. the "match_value" of a match target.
{{ objective }}

Transformations map a measured value x to the value being optimized:
- IdentityTransformation: x
- AffineTransformation: factor * x + shift
- ClampingTransformation: x limited to the range [cutoffs.lower, cutoffs.upper]
- AbsoluteTransformation: |x|
- LogarithmicTransformation: ln(x)
- ExponentialTransformation: exp(x)
- PowerTransformation: x ** exponent
- SigmoidTransformation: 1 / (1 + exp(-steepness * (center - x)))
- TwoSidedAffineTransformation: slope_left * (x - midpoint) if x < midpoint, \
else slope_right * (x - midpoint)
- BellTransformation: exp(-((x - center) / sigma) ** 2 / 2), i.e. highest at x = center
- TriangularTransformation: rising linearly from 0 at cutoffs.lower to 1 at peak, \
falling linearly back to 0 at cutoffs.upper, and 0 outside the cutoffs
- ChainedTransformation: the listed transformations applied in order, first one first
- AdditiveTransformation / MultiplicativeTransformation: sum / product of the listed \
transformations, each applied to x

{% endif %}
PARAMETERS:
{{ parameters }}
{% if constraints is not none %}

CONSTRAINTS:
All recommendations must satisfy the following constraints.
{{ constraints }}
{% endif %}
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
{%- if forbidden_configurations is not none %}

FORBIDDEN CONFIGURATIONS:
{{ forbidden_instructions }}
{{ forbidden_configurations }}
{% endif %}
{%- if recovery_instruction is not none %}

Your previous recommendation could not be used and needs to be corrected.

WHAT WENT WRONG:
{{ recovery_instruction }}
{% if original_response is not none %}

ORIGINAL RESPONSE:
{{ original_response }}
{% endif %}

Please provide a corrected set of {{ batch_size }} distinct experimental conditions \
that addresses the problem described above and improves the optimization objective.
{% else %}

Please recommend {{ batch_size }} distinct new experimental conditions that are likely \
to improve the optimization objective.
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
    experiment_description: str
    forbidden_configurations: str | None
    forbidden_instructions: str
    measurements: str | None
    objective: str | None
    original_response: str | None
    parameters: str
    pending_experiments: str | None
    recovery_instruction: str | None
    response_format: str


def _drop_empty(obj: Any, /) -> Any:
    """Recursively drop ``None`` values and empty dictionaries from dictionaries.

    Args:
        obj: The serialized object to clean up.

    Returns:
        The cleaned-up object.
    """
    if isinstance(obj, dict):
        cleaned = {key: _drop_empty(value) for key, value in obj.items()}
        return {k: v for k, v in cleaned.items() if v is not None and v != {}}
    if isinstance(obj, list):
        return [_drop_empty(value) for value in obj]
    return obj


def _to_prompt_dict(obj: Parameter | Constraint | Objective, /) -> dict[str, Any]:
    """Serialize an object into the representation shown to the language model.

    Args:
        obj: The object to serialize.

    Returns:
        The serialized object.
    """
    serialized = obj.to_dict()
    if isinstance(obj, _DiscreteLabelLikeParameter):
        active = list(obj.active_values)
        serialized.pop(fields(_DiscreteLabelLikeParameter)._active_values.alias)
        # For label-like parameters, ``values`` holds the active labels only.
        serialized[fields(CategoricalParameter)._values.alias] = active
        if isinstance(obj, SubstanceParameter):
            # Substance parameters keep the SMILES of their active labels.
            serialized[fields(SubstanceParameter).data.alias] = {
                label: smiles for label, smiles in obj.data.items() if label in active
            }
        elif isinstance(obj, CustomDiscreteParameter):
            # The descriptor table of custom parameters is omitted since it is only
            # relevant for the surrogate model.
            del serialized[fields(CustomDiscreteParameter).data.alias]
    # ``None`` values and empty dictionaries are dropped since they carry no
    # information for the language model.
    return _drop_empty(serialized)


def _to_json_lines(serialized_objects: Iterable[dict[str, Any]], /) -> str:
    """Render serialized objects as JSON, one object per line.

    Args:
        serialized_objects: The serialized objects to render.

    Returns:
        The rendered objects.
    """
    return "\n".join(json.dumps(obj, ensure_ascii=False) for obj in serialized_objects)


def _table_text(df: pd.DataFrame | None, columns: list[str], /) -> str | None:
    """Render the given columns of a dataframe as a table.

    Restricting the columns excludes non-experimental columns, such as the metadata
    columns added by :class:`~baybe.campaign.Campaign`.

    Args:
        df: The optional dataframe to render.
        columns: The columns to include.

    Returns:
        The rendered table, or ``None`` if no or an empty dataframe is provided.
    """
    if df is None or df.empty:
        return None
    return df[columns].to_string(index=False)


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

    The objective, parameters and constraints are included in their serialized form.
    Measurements and pending experiments are restricted to their parameter and target
    columns.

    When ``error`` is provided, the prompt is rendered in *recovery* mode: it carries
    the exact same context as the original query and appends a correction section
    asking the model to fix its previous response. Because each model call is
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
        error: If set, renders a recovery prompt using the error's
            :attr:`~baybe.exceptions.LLMResponseError.recovery_instruction`.
        original_response: The previous, rejected response to be corrected, if the
            model returned any content. Requires ``error``.

    Raises:
        ValueError: If ``original_response`` is provided without ``error``.

    Returns:
        The constructed prompt.
    """
    if original_response is not None and error is None:
        raise ValueError("'original_response' can only be provided with 'error'.")

    from baybe._optional.llm import StrictUndefined, Template

    parameter_names = [p.name for p in searchspace.parameters]
    target_names = [t.name for t in objective.targets] if objective is not None else []
    constraints = searchspace.constraints
    context: _PromptContext = {
        "experiment_description": experiment_description,
        "objective": (
            _to_json_lines([_to_prompt_dict(objective)])
            if objective is not None
            else None
        ),
        "parameters": _to_json_lines(
            _to_prompt_dict(p) for p in searchspace.parameters
        ),
        "constraints": (
            _to_json_lines(_to_prompt_dict(c) for c in constraints)
            if constraints
            else None
        ),
        "measurements": _table_text(measurements, parameter_names + target_names),
        "pending_experiments": _table_text(pending_experiments, parameter_names),
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
