"""LLM-specific exceptions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection

from typing_extensions import override

from baybe.constraints.discrete import DiscreteBatchConstraint
from baybe.exceptions import LLMError


class LLMResponseError(LLMError, ABC):
    """An error occurred while processing a language model response."""

    @property
    @abstractmethod
    def recovery_instruction(self) -> str:
        """Guidance for the language model on how to correct its response."""


class LLMMalformedResponseError(LLMResponseError):
    """A language model response could not be parsed into the expected structure."""

    def __init__(self, message: str, *, problems: Collection[str] = ()) -> None:
        super().__init__(message)
        self.problems = tuple(problems)

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            "Your previous response did not contain valid JSON in the required "
            "format. Return JSON that follows the required response format, and "
            "nothing else." + "".join(f"\n- {p}" for p in self.problems)
        )


class LLMUnknownParameterError(LLMResponseError):
    """A language model response referenced parameters outside the search space."""

    def __init__(
        self,
        message: str,
        *,
        unknown_names: Collection[str],
        valid_names: Collection[str],
    ) -> None:
        super().__init__(message)
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

    def __init__(self, message: str, *, parameters: Collection[str]) -> None:
        super().__init__(message)
        self.parameters = parameters

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"Your previous response violated the "
            f"'{DiscreteBatchConstraint.__name__}' constraint on parameters "
            f"{list(self.parameters)}. Ensure all recommendations satisfy this "
            f"constraint."
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

    def __init__(self, message: str, *, n_duplicates: int) -> None:
        super().__init__(message)
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

    def __init__(self, message: str, *, requested: int, received: int) -> None:
        super().__init__(message)
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
