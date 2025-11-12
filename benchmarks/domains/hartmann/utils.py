"""Utilities for the Hartmann domain."""

import numpy as np
import torch
from attr import define, field
from botorch.test_functions.synthetic import Hartmann


@define
class HartmannShifted:
    """Wrapper around the Hartmann function that shifts its input dimensions.

    Besides shifting the input, also the bounds will be shifted.
    """

    shifts: list[float] = field()
    """Amount to shift individual dimension coordinates by.

    E.g. [0, 0.2, 0] would mean shifting dimension 1 by 0.2
    and 0 for dimensions 0 and 2.
    """

    kwargs_dict: dict = field(factory=dict)
    """Additional keyword arguments for the Hartmann function.

    If bounds are not specified, uses bounds of [0, 1] for each dimension.
    If dim is not specified, it is inferred based on shifts.
    """

    base_func: Hartmann = field(init=False)
    """The underlying Hartmann function instance."""

    def __attrs_post_init__(self) -> None:
        """Initialize the shifted Hartmann function."""
        # Make sure input dimensions are consistent
        dim = self.kwargs_dict.get("dim", None)
        if dim is None:
            dim = len(self.shifts)
            self.kwargs_dict["dim"] = dim
        if "bounds" not in self.kwargs_dict:
            self.kwargs_dict["bounds"] = np.array([[0.0] * dim, [1.0] * dim]).T
        bounds_original = self.kwargs_dict["bounds"]
        assert dim == len(self.shifts) == bounds_original.shape[0]
        assert bounds_original.shape[1] == 2

        # Make shifted bounds
        bounds_shifted = np.array(
            [
                [
                    low if shift >= 0 else low + shift
                    for low, shift in zip(bounds_original[:, 0], self.shifts)
                ],
                [
                    high if shift <= 0 else high + shift
                    for high, shift in zip(bounds_original[:, 1], self.shifts)
                ],
            ]
        ).T
        self.kwargs_dict["bounds"] = bounds_shifted

        self.base_func = Hartmann(**self.kwargs_dict)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the shifted Hartmann function.

        Args:
            x: Input tensor of shape (n_samples, n_dimensions).

        Returns:
            Output tensor of shape (n_samples,).
        """
        # Create a copy to avoid modifying input
        x_shifted = x.clone()
        for dim, shift in enumerate(self.shifts):
            x_shifted[:, dim] = x_shifted[:, dim] + shift
        return self.base_func(x_shifted)

    @property
    def dim(self) -> int:
        """Dimension of the input space."""
        return self.base_func.dim

    @property
    def _bounds(self) -> np.ndarray:
        """Bounds of the shifted function."""
        return self.base_func._bounds
