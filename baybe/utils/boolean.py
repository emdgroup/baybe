"""Functions implementing boolean checks."""

from abc import ABC
from typing import Any

from attr import cmp_using
from rdkit import Chem


# Used for comparing pandas dataframes in attrs classes
eq_dataframe = cmp_using(lambda x, y: x.equals(y))


def is_abstract(cls: Any) -> bool:
    """
    Determines if a given class is abstract in a more general sense than
    `inspect.abstract`, which only verifies if a class has abstract methods. The
    latter can be problematic when the class has no abstract methods but is
    nevertheless not directly usable, for example, because it has uninitialized
    members, which are only covered in its non-"abstract" subclasses. By contrast,
    this method simply checks if the class derives from `abc.ABC`.

    Parameters
    ----------
    cls : Any
        The class to be inspected.

    Returns
    -------
    bool
        True if the class is "abstract" (see definition above), False else.
    """
    return ABC in cls.__bases__


def is_valid_smiles(smiles: str) -> bool:
    """
    Tests if a SMILES string is valid according to RDKit.

    Parameters
    ----------
    smiles : str
        SMILES string to be tested.

    Returns
    -------
    bool
        True if smiles is valid, False else.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def strtobool(val: str) -> bool:
    """
    Convert a string representation of truth to True or False. Adapted from distutils.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    Parameters
    ----------
    val: str
        String to be checked.

    Returns
    -------
    bool
    """
    if val.lower() in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val.lower() in ("n", "no", "f", "false", "off", "0"):
        return False

    raise ValueError(f"Invalid truth value: {val}")


def check_if_in(element: Any, allowed: list) -> None:
    """
    Checks if an element is in a given list of elements and raises a
    context-specific exception if it is not.
    """
    if element not in allowed:
        raise ValueError(
            f"The value '{element}' is not allowed. Must be one of {allowed}."
        )
