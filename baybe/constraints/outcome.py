"""Functionality for outcome constraints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pandas as pd
import torch
from attrs import define, field
from attrs.validators import in_, instance_of

from baybe.serialization.mixin import SerialMixin
from baybe.targets.base import Target


@define(frozen=True, slots=False)
class OutcomeConstraint(SerialMixin):
    """A constraint applied to target outcomes in the output space.

    Outcome constraints restrict the feasible region based on target predictions,
    different from parameter constraints which restrict the input space.
    """

    target: Target = field(validator=instance_of(Target))
    """The target to be constrained."""

    operator: Literal["<=", ">=", "=="] = field(validator=in_(["<=", ">=", "=="]))
    """The constraint operator."""

    threshold: float = field(validator=instance_of((int, float)), converter=float)
    """The constraint threshold value in experimental units."""

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.target.name} {self.operator} {self.threshold}"

    def get_computational_threshold(self) -> float:
        """Convert experimental threshold to computational units.

        Returns:
            The threshold value in computational units.
        """
        # Create dummy series with threshold value in experimental units
        experimental_series = pd.Series([self.threshold], name=self.target.name)

        # Apply the same transformations as the target
        computational_series = self.target.transform(experimental_series)

        return computational_series.iloc[0]

    def to_botorch_constraint_func(
        self, target_idx: int
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Create a botorch-compatible constraint function.

        Args:
            target_idx: Index of the target in model output.

        Returns:
            A constraint function that returns <= 0 for feasible region.
        """
        computational_threshold = self.get_computational_threshold()

        def constraint_func(samples: torch.Tensor) -> torch.Tensor:
            """Constraint function operating on computational level.

            Args:
                samples: Model output samples in computational units.

            Returns:
                Constraint values where <= 0 indicates feasible region.

            Raises:
                ValueError: If the constraint operator is not supported.
            """
            if self.operator == "<=":
                return samples[..., target_idx] - computational_threshold
            elif self.operator == ">=":
                return computational_threshold - samples[..., target_idx]
            elif self.operator == "==":
                # Equality constraint with small tolerance
                return (
                    torch.abs(samples[..., target_idx] - computational_threshold) - 1e-6
                )
            else:
                raise ValueError(f"Unsupported constraint operator: {self.operator}")

        return constraint_func
