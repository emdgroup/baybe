"""Optional LLM imports."""

from baybe.exceptions import OptionalImportError

try:
    from jinja2 import Template
    from litellm import completion

except ModuleNotFoundError as ex:
    raise OptionalImportError(name="litellm", group="llm") from ex

__all__ = [
    "Template",
    "completion",
]
