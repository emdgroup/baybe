"""Optional LLM imports."""

from baybe.exceptions import OptionalImportError

try:
    from jinja2 import StrictUndefined, Template
except ModuleNotFoundError as ex:
    raise OptionalImportError(name="jinja2", group="llm") from ex

try:
    from litellm import completion
except ModuleNotFoundError as ex:
    raise OptionalImportError(name="litellm", group="llm") from ex

__all__ = [
    "StrictUndefined",
    "Template",
    "completion",
]
