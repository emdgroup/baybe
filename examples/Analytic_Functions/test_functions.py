"""A file containing functions that can be used for analytical tests."""
# pylint: disable=invalid-name
from abc import ABC, abstractmethod

import numpy as np
from attrs import define
from baybe.utils.serialization import SerialMixin


@define
class AnalyticTestFunction(ABC, SerialMixin):
    """Abstract parent class for analytical test functions."""

    dimension: int

    def __call__(self, x: np.array) -> float:
        raise NotImplementedError("A __call__ method needs to be implemented.")

    @property
    @abstractmethod
    def bounds(self) -> np.ndarray:
        """The bounds of the individual variables"""

    @property
    @abstractmethod
    def optimum(self) -> float:
        """The optimum value of the function."""

    @property
    @abstractmethod
    def argmin(self) -> np.ndarray:
        """The input for which the optimum value is obtained."""


@define
class Hartmann3(AnalyticTestFunction):
    """Three-dimensional Hartmann function.
    It has 4 local minima."""

    dimension: int = 3

    def __call__(self, x1: float, x2: float, x3: float) -> float:
        x = np.array([x1, x2, x3])

        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [3, 10, 30],
                [0.1, 10, 35],
                [3, 10, 30],
                [0.1, 10, 35],
            ]
        )
        P = 1e-4 * np.array(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        )

        outer = 0
        for i in range(4):
            inner = sum(A[i, j] * (x[j] - P[i, j]) ** 2 for j in range(3))
            outer = outer + alpha[i] * np.exp(-inner)

        y = -outer
        return y

    @property
    def bounds(self) -> np.ndarray:
        return np.array(((0, 1),) * 3)

    @property
    def optimum(self) -> float:
        return -3.86278

    @property
    def argmin(self) -> np.ndarray:
        return np.array([0.114614, 0.555649, 0.852547])


@define
class Rastrigin(AnalyticTestFunction):
    """
    General form of the rastrigin test function.
    This function has a lot of local minima, but only one global minimum which is at
    x=(0,...,0).
    """

    def __call__(self, *X: float, A: int = 10) -> float:
        delta = [x**2 - A * np.cos(2 * np.pi * x) for x in X]
        y = A * len(X) + np.sum(delta)
        return y

    @property
    def bounds(self) -> list:
        return [
            (-5.12, 5.12),
        ] * self.dimension

    @property
    def optimum(self) -> float:
        return 0

    @property
    def argmin(self) -> np.array:
        return np.array((0,) * self.dimension)
