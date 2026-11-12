"""LLM-specific exceptions.

The public base error and the public warning are defined in ``baybe.exceptions`` so
that they are part of the documented public surface; this module imports the error to
build the internal subclass hierarchy.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from typing_extensions import override

from baybe.exceptions import LLMError


class LLMCallError(LLMError):
    """The call to the language model failed before producing a response.

    Raised for failures such as network errors, rate limiting, timeouts, or an
    unknown model identifier, where no response is returned to process.
    """


class LLMAuthenticationError(LLMCallError):
    """Authentication with the language model provider failed."""


class LLMResponseError(LLMError):
    """An error occurred while processing a language model response."""

    @property
    def recovery_instruction(self) -> str:
        """Guidance for the language model on how to correct its response."""
        return (
            "Your previous response could not be used. Provide a new response that "
            "follows all instructions in the prompt."
        )


class LLMMalformedResponseError(LLMResponseError):
    """A language model response could not be parsed into the expected structure."""

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            "Your previous response did not contain valid JSON in the required "
            "format. Return JSON that follows the required response format, and "
            "nothing else."
        )


class LLMUnknownParameterError(LLMResponseError):
    """A language model response referenced parameters outside the search space."""

    def __init__(
        self,
        *args: Any,
        unknown_names: Collection[str],
        valid_names: Collection[str],
    ):
        super().__init__(*args)
        self.unknown_names = unknown_names
        self.valid_names = valid_names

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"Your previous response used unknown parameter names "
            f"{sorted(self.unknown_names)}. Use only the following parameters: "
            f"{sorted(self.valid_names)}."
        )


class LLMBatchConstraintValidationError(LLMResponseError):
    """A language model response violated a batch constraint of the search space."""

    def __init__(
        self,
        *args: Any,
        constraint_name: str,
        parameters: Collection[str],
    ):
        super().__init__(*args)
        self.constraint_name = constraint_name
        self.parameters = parameters

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"Your previous response violated the '{self.constraint_name}' constraint "
            f"on parameters {list(self.parameters)}. Ensure all recommendations "
            f"satisfy this constraint."
        )


class LLMIneligiblePointsError(LLMResponseError):
    """A language model response proposed points outside the search space."""

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"{self}\nPropose different configurations that belong to the search "
            f"space instead."
        )


class LLMDuplicateRecommendationsError(LLMResponseError):
    """A language model response contained identical recommendations."""

    def __init__(self, *args: Any, n_duplicates: int):
        super().__init__(*args)
        self.n_duplicates = n_duplicates

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"{self.n_duplicates} of your proposed configurations duplicate another "
            f"configuration of the same batch. All recommendations within a batch must "
            f"be distinct."
        )


class LLMBatchSizeError(LLMResponseError):
    """A language model response contained the wrong number of recommendations."""

    def __init__(self, *args: Any, requested: int, received: int):
        super().__init__(*args)
        self.requested = requested
        self.received = received

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"Your previous response provided {self.received} recommendation(s), "
            f"but exactly {self.requested} are required. Provide {self.requested} "
            f"distinct, valid recommendations."
        )
