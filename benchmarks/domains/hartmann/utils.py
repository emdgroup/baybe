"""Utilities for the Hartmann domain."""

import numpy as np
import torch
from botorch.test_functions.synthetic import Hartmann


def get_shifted_hartmann(
    shifts: list[float], bounds: np.ndarray | None = None, **kwargs
) -> tuple[callable, np.ndarray]:
    """Create a wrapper around Hartmann function that shifts individual dimensions.

    Args:
        shifts: Amount to shift individual dimension coordinates by.
            E.g. [0, 0.2, 0] would mean shifting dimension 1 by 0.2
            and 0 for dimensions 0 and 2.
        bounds: Bounds as would be used for the Hartmann function without shifting.
            Should be of shape (2, n_dimensions) with first row being lower bounds
            and second row being upper bounds.
            They will be extended internally to account for the shifts.
            If None uses bounds of [0, 1] for each dimension.
        **kwargs: Additional keyword arguments to pass to the Hartmann function.

    Returns:
        Callable that wraps Hartmann with shifted dimensions and shifted bounds.
          The callable expects as input a tensor of shape (n_samples, n_dimensions).
          The bounds are transformed so that they can be directly used in 
          further Hartmann function calls.
    """
    # Make sure input dimensions are consistent
    dim = kwargs.get("dim", None)
    if dim is None:
        dim = len(shifts)
        kwargs["dim"] = dim
    if bounds is None:
        bounds = np.array([[0.0] * dim, [1.0] * dim])
    assert dim == len(shifts) == bounds.shape[1]
    assert bounds.shape[0] == 2

    # Make shifted bounds
    bounds_shifted = np.array(
        [
            [
                low if shift >= 0 else low + shift
                for low, shift in zip(bounds[0], shifts)
            ],
            [
                high if shift <= 0 else high + shift
                for high, shift in zip(bounds[1], shifts)
            ],
        ]
    ).transpose()
    kwargs["bounds"] = bounds_shifted

    base_func = Hartmann(**kwargs)

    def shifted_hartmann(x: torch.Tensor) -> torch.Tensor:
        # Create a copy to avoid modifying input
        x_shifted = x.clone()
        for dim, shift in enumerate(shifts):
            x_shifted[:, dim] = x_shifted[:, dim] + shift
        return base_func(x_shifted)

    return shifted_hartmann, bounds_shifted
