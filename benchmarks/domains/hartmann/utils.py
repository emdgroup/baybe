"""Utilities for the Hartmann domain."""

import numpy as np
import torch
from botorch.test_functions.synthetic import Hartmann
from typing_extensions import override


class ShiftedHartmann(Hartmann):
    """Hartmann function with support for shifting input dimensions.

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

        # Process the shifts
        if shift is not None and len(shift) != bounds.shape[0]:
            raise ValueError("Shift shape does not match used dimensions.")
        self.shift = shift if shift is not None else [0.0] * bounds.shape[0]

        # Extend the bounds
        # The original Hartmann function throws an error if it is called outside of its
        # bounds ([0,1] by default). However, it is technically feasible to evaluate it
        # outside of the default unit interval. To enable passing of the validation for
        # our shifted variant, we simply expand the bounds by an appropriate amount
        # that depends on ``shift``. The bounds used in the benchmark search space
        # remain unchanged.
        # Note: We can not only shift the upper and lower bounds as that leads to the
        # optimal value (``_optimizer``) being excluded from bounds. As this is
        # hard-coded in the Hartmann class init, we can not override it before it is
        # evaluated.
        shifted_bounds = bounds + np.array(self.shift)[:, None]
        bounds_extended = list(
            map(
                tuple,
                np.stack(
                    [
                        np.minimum(bounds[:, 0], shifted_bounds[:, 0]),
                        np.maximum(bounds[:, 1], shifted_bounds[:, 1]),
                    ],
                    axis=1,
                ),
            )
        )

        kwargs["bounds"] = bounds_extended

        super().__init__(**kwargs)
        # Remove optimizers as they are incorrect if shift is used
        # They are anyway only set for 3D and 6D hartman in the parent class
        if self._optimizers is not None:
            self._optimizers = None
            self.optimizers = None

    @override
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
