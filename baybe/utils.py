"""
Collection of small utilities
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

if TYPE_CHECKING:
    from .core import BayBE


def is_valid_smiles(smiles: str) -> bool:
    """
    Test if a SMILEs string is valid according to RDKit.

    Parameters
    ----------
    smiles: str
        SMILES string
    Returns
    -------
        bool, True if smiles is valid, False else
    """
    raise NotImplementedError("This function is not implemented yet")


def to_tensor(*dfs: pd.DataFrame) -> Union[Tensor, Iterable[Tensor]]:
    """Converts a given set of dataframes into tensors (dropping all indices)."""
    out = (torch.from_numpy(df.values).to(torch.float32) for df in dfs)
    if len(dfs) == 1:
        out = next(out)
    return out


def check_if_in(element: Any, allowed: list):
    """
    Checks if an element is in a given list of elements and raises a
    context-specific exception if it is not.
    """
    if element not in allowed:
        raise ValueError(
            f"The value '{element}' is not allowed. Must be one of {allowed}."
        )


def add_fake_results(
    data: pd.DataFrame,
    obj: BayBE,
    good_reference_values: Optional[Dict[str, list]] = None,
    good_intervals: Optional[Tuple] = None,
    bad_intervals: Optional[Tuple] = None,
) -> None:
    """
    Add fake results to a dataframe which was the result of the BayBE recommendation
    action. It is possible to identify "good" values, which will be given a better
    target value. With this the algorithm can be driven towards certain optimal values
    whilst still being random. Useful for testing.

    Parameters
    ----------
    data : pandas dataframe
           Output of the recommend function of a BayBE object
    obj : BayBE class instance
          The baybe object which provides configuration, targets, etc.
    good_reference_values : dictionary
                  A dictionaries which defines parameters and respective values
                  which identify what will be considered good values.
                  Example {'parameter1': [1,4,42]}
    good_intervals : 2-tuple
                     Good entries will get a random value in the range defined by this
                     tuple
    bad_intervals : 2-tuple
                    Bad entries will get a random value in the range defined by this
                    tuple

    Returns
    -------
    Nothing since it operated directly on the data
    """
    # TODO Add support for multiple targets

    # Sanity checks for good_bad_ratio
    if good_intervals is None:
        if obj.targets[0].mode == "MAX":
            good_intervals = (66, 100)
        elif obj.targets[0].mode == "MIN":
            good_intervals = (0, 33)
        elif obj.targets[0].mode == "MATCH":
            good_intervals = tuple(*obj.targets[0].bounds)
        else:
            raise ValueError("Unrecognized target mode when trying to add fake values.")
    if bad_intervals is None:
        if obj.targets[0].mode == "MAX":
            bad_intervals = (0, 33)
        elif obj.targets[0].mode == "MIN":
            bad_intervals = (66, 100)
        elif obj.targets[0].mode == "MATCH":
            bad_intervals = (
                0.05 * obj.targets[0].bounds[0],
                0.3 * obj.targets[0].bounds[0],
            )
        else:
            raise ValueError("Unrecognized target mode when trying to add fake values.")
    if not isinstance(good_intervals, Tuple) or (len(good_intervals) != 2):
        raise TypeError("Parameter good_intervals must be a 2-tuple")
    if not isinstance(bad_intervals, Tuple) or (len(bad_intervals) != 2):
        raise TypeError("Parameter bad_intervals must be a 2-tuple")

    # Sanity check for good_values. Assure we only consider columns that are in the data
    if good_reference_values is None:
        good_reference_values = {}

    size = len(data)
    for target in obj.targets:
        # add bad values
        data[target.name] = np.random.randint(bad_intervals[0], bad_intervals[1], size)

        # add good values
        masks = []

        if len(good_reference_values) > 0:
            for param, vals in good_reference_values.items():
                if param not in data.columns:
                    raise ValueError(
                        f"When adding fake results you specified good "
                        f"values for the parameter '{param}' but this "
                        f"parameter is not in the dataframe."
                    )
                if not isinstance(vals, list):
                    raise TypeError(
                        f"Entries in parameter good_reference_values "
                        f"(which is a dictionary) must be lists, but you "
                        f"provided {vals}"
                    )
                mask = data[param].apply(partial(lambda x, v: x in v, v=vals))
                masks.append(mask)

            # Good values will be added where the parameters of the
            # corresponding datapoints match the ones defined in good_reference_values
            for k, mask in enumerate(masks):
                if k == 0:
                    final_mask = mask

                final_mask &= mask

            data.loc[final_mask, target.name] = np.random.randint(
                good_intervals[0], good_intervals[1], final_mask.sum()
            )


def add_noise(
    data: pd.DataFrame,
    obj: BayBE,
    noise_type: str = "absolute",
    noise_level: float = 1.0,
):
    """
    Adds uniform noise to parameter values of a recommendation frame. Simulates
    experimental noise and inputting numerical values that are slightly different
    than the recommendations coming from the search space.

    Parameters
    ----------
    data : pandas dataframe
           output of the recommend function of a BayBE object
    obj : BayBE class instance
          the baybe object which provides configuration, targets, etc.
    noise_type : str
        Defines whether the noise should be additive
    noise_level : float
        Level/magnitude of the noise, numerical value for type absolute and percentage
        for type relative_percent

    Returns
    -------
        Nothing
    """
    for param in obj.parameters:
        if "NUM" in param.type:
            if noise_type == "relative_percent":
                data[param.name] *= np.random.uniform(
                    1.0 - noise_level / 100.0, 1.0 + noise_level / 100.0, len(data)
                )
            elif noise_type == "absolute":
                data[param.name] += np.random.uniform(
                    -noise_level, noise_level, len(data)
                )
            else:
                raise ValueError(
                    f"Parameter noise_type was {noise_type} but must be either "
                    f'"absolute" or "relative_percent"'
                )
