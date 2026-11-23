"""Shared definitions for the LLM response wire contract.

The language model is asked to return a JSON object holding an array of
recommendations. The field names of those objects are derived from the classes
defined here, so that prompt construction and response parsing consume the same
definitions and cannot silently drift apart.
"""

import gc
from typing import Any

from attrs import define, field, fields


@define(frozen=True)
class ResponseItem:
    """A single recommendation in the language model response."""

    explanation: str = field()
    """Free-text explanation of the recommendation."""

    parameters: dict[str, Any] = field()
    """Parameter assignments for the recommendation."""


@define(frozen=True)
class Response:
    """The complete language model response."""

    recommendations: list[ResponseItem] = field()
    """The recommendations, in the order provided by the language model."""


_EXPLANATION_FIELD = fields(ResponseItem).explanation.name
"""JSON field holding a recommendation's free-text explanation."""

_PARAMETERS_FIELD = fields(ResponseItem).parameters.name
"""JSON field holding a recommendation's parameter assignments."""

_RECOMMENDATIONS_FIELD = fields(Response).recommendations.name
"""JSON field holding the array of recommendations."""


def _response_format(batch_size: int = 1) -> str:
    """Render the response-format example embedded in the prompts.

    Args:
        batch_size: The number of recommendations to illustrate in the example. The
            returned example contains exactly ``batch_size`` entries so the model
            can see the expected array length at a glance.

    Returns:
        A JSON example illustrating the expected response structure, using the field
        names defined in this module as the single source of truth.
    """
    entry = (
        f"    {{\n"
        f'      "{_EXPLANATION_FIELD}": "Brief explanation of the recommendation",\n'
        f'      "{_PARAMETERS_FIELD}": {{\n'
        f'        "<parameter-name>": <value>\n'
        f"      }}\n"
        f"    }}"
    )
    entries = ",\n".join(entry for _ in range(max(batch_size, 1)))
    return f'{{\n  "{_RECOMMENDATIONS_FIELD}": [\n{entries}\n  ]\n}}'


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
