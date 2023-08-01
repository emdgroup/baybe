"""Collection of small utilities."""

import random
from dataclasses import dataclass
from typing import List, TypeVar

import numpy as np
import torch


_T = TypeVar("_T")


@dataclass(frozen=True, repr=False)
class Dummy:
    """
    Placeholder element for array-like data types. Useful e.g. for detecting
    duplicates in constraints.
    """

    def __repr__(self):
        return "<dummy>"


def get_subclasses(cls: _T, recursive: bool = True, abstract: bool = False) -> List[_T]:
    """
    Returns a list of subclasses for the given class.

    Parameters
    ----------
    cls
        The base class to retrieve subclasses for.
    recursive : bool
        If True, indirect subclasses (i.e. subclasses of subclasses) are included.
    abstract : bool
        If True, abstract subclasses are included.

    Returns
    -------
    list
        A list of subclasses for the given class.
    """
    from baybe.utils import isabstract  # pylint: disable=import-outside-toplevel

    subclasses = []
    for subclass in cls.__subclasses__():

        # Append direct subclass only if it is not abstract
        if abstract or not isabstract(subclass):
            subclasses.append(subclass)

        # If requested, add indirect subclasses
        if recursive:
            subclasses.extend(get_subclasses(subclass, abstract=abstract))

    return subclasses


def set_random_seed(seed: int) -> None:
    """Sets the global random seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
