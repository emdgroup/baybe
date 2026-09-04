"""Shared definitions for the LLM response wire contract.

The language model is asked to return a JSON array of suggestion objects. The field
names of those objects are defined here once, so that prompt construction and response
parsing consume the same definitions and cannot silently drift apart.
"""

_EXPLANATION_FIELD = "explanation"
"""JSON field holding a suggestion's free-text explanation."""

_PARAMETERS_FIELD = "parameters"
"""JSON field holding a suggestion's parameter assignments."""


def _response_format() -> str:
    """Render the response-format example embedded in the prompts.

    Returns:
        A JSON example illustrating the expected response structure, using the field
        names defined in this module as the single source of truth.
    """
    return f"""\
[
  {{
    "{_EXPLANATION_FIELD}": "Brief explanation of the suggestion",
    "{_PARAMETERS_FIELD}": {{
      "<parameter-name>": <value>
    }}
  }}
]"""
