"""Shared definitions for the LLM response wire contract.

The language model is asked to return a JSON array of suggestion objects. The field
names of those objects are defined here once, so that prompt construction and response
parsing consume the same definitions and cannot silently drift apart.
"""

_EXPLANATION_FIELD = "explanation"
"""JSON field holding a suggestion's free-text explanation."""

_PARAMETERS_FIELD = "parameters"
"""JSON field holding a suggestion's parameter assignments."""


def _response_format(batch_size: int = 1) -> str:
    """Render the response-format example embedded in the prompts.

    Args:
        batch_size: The number of suggestions to illustrate in the example. The
            returned example contains exactly ``batch_size`` entries so the model
            can see the expected array length at a glance.

    Returns:
        A JSON example illustrating the expected response structure, using the field
        names defined in this module as the single source of truth.
    """
    entry = (
        f"  {{\n"
        f'    "{_EXPLANATION_FIELD}": "Brief explanation of the suggestion",\n'
        f'    "{_PARAMETERS_FIELD}": {{\n'
        f'      "<parameter-name>": <value>\n'
        f"    }}\n"
        f"  }}"
    )
    entries = ",\n".join(entry for _ in range(max(batch_size, 1)))
    return f"[\n{entries}\n]"
