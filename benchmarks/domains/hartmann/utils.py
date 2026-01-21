"""Utilities for the Hartmann domain."""

import numpy as np
import torch
from botorch.test_functions.synthetic import Hartmann


class CustomHartmann(Hartmann):
    """Custom Hartmann function with support for shifting input dimensions.

    Extends the Hartmann test function to support shifting input dimensions
    (and adjusting bounds accordingly). Other parameters like bounds, dim,
    noise_std, and negate are passed directly to the parent Hartmann class.

    Args:
        shift: Amount to shift individual dimension coordinates by.
            E.g. [0.2, 0, 0] would shift dimension 0 by 0.2.
            If None, no shifting is applied.
        **kwargs: Keyword arguments passed to parent Hartmann class.

    Raises:
        ValueError: If bounds shape is invalid (i.e., not list of (min,max) tuples).
        ValueError: If shift shape does not match the used dimensions.
    """

    def __init__(
        self,
        shift: list[float] | None = None,
        **kwargs,
    ) -> None:
        # Get botorch defaults if not specified
        bounds = np.array(kwargs.get("bounds", Hartmann(**kwargs).bounds.T))
        if bounds.shape[1] != 2:
            raise ValueError(
                "Bounds shape is invalid, it should be tuple (min,max) per dim."
            )

        # Process the shifts
        if shift is not None and len(shift) != bounds.shape[0]:
            raise ValueError("Shift shape does not match used dimensions.")
        self.shift = shift if shift is not None else [0.0] * bounds.shape[0]

        # Shift the bounds
        bounds = list(
            map(
                tuple,
                np.array(
                    [
                        [
                            low if shift >= 0 else low + shift
                            for low, shift in zip(bounds[:, 0], self.shift)
                        ],
                        [
                            high if shift <= 0 else high + shift
                            for high, shift in zip(bounds[:, 1], self.shift)
                        ],
                    ]
                ).T,
            )
        )

        kwargs["bounds"] = bounds

        super().__init__(**kwargs)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the shifted Hartmann function.

        Args:
            x: Input tensor of shape (n_samples, n_dimensions).

        Returns:
            Output tensor of shape (n_samples,).
        """
        # Create a copy to avoid modifying input
        x_shifted = x.clone()
        for dim, shift in enumerate(self.shift):
            x_shifted[:, dim] = x_shifted[:, dim] + shift
        return super().__call__(x_shifted)
