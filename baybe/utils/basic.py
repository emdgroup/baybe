"""Collection of small basic utilities."""

import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, TypeVar

import numpy as np
import torch

_T = TypeVar("_T")


@dataclass(frozen=True, repr=False)
class Dummy:
    """Placeholder element for array-like data types.

    Useful e.g. for detecting duplicates in constraints.
    """

    def __repr__(self):
        """Return a representation of the placeholder."""
        return "<dummy>"


def get_subclasses(cls: _T, recursive: bool = True, abstract: bool = False) -> List[_T]:
    """Return a list of subclasses for the given class.

    Args:
        cls: The base class to retrieve subclasses for.
        recursive: If ``True``, indirect subclasses (i.e. subclasses of subclasses)
            are included.
        abstract: If ``True``, abstract subclasses are included.

    Returns:
        A list of subclasses for the given class.
    """
    from baybe.utils import is_abstract

    subclasses = []
    for subclass in cls.__subclasses__():
        # Append direct subclass only if it is not abstract
        if abstract or not is_abstract(subclass):
            subclasses.append(subclass)

        # If requested, add indirect subclasses
        if recursive:
            subclasses.extend(get_subclasses(subclass, abstract=abstract))

    return subclasses


def set_random_seed(seed: int):
    """Set the global random seed.

    Args:
        seed: The chosen global random seed.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def hilberts_factory(factory: Callable[..., _T]) -> Iterable[_T]:
    """Provide an infinite stream of the factory's products."""
    while True:
        yield factory()
