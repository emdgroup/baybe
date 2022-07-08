"""
Collection of small utilities
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from baybe.core import BayBE


def is_valid_smiles(smiles: str) -> bool:
    """
    Test if a SMILEs string is valid. Currently NOT IMPLEMENTED
    :param smiles: SMILES string to test
    :return: True if the SMILES is valid according to RDKit
    """
    return "C" in smiles  # temporary until implemented


def add_fake_results(
    data: pd.DataFrame,
    obj: BayBE,
    good_reference_values: List = None,
    good_intervals: Tuple = None,
    bad_intervals: Tuple = None,
) -> None:
    """
    Add fake results to a dataframe which was the result of the BayBE recommendation
    action. It is possible to identify "good" values, which will be given a better
    target value. With this the algorithm can be driven towards certain optimal values
    whilst still being random. Useful for testing.

    Parameters
    ----------
    data : pandas dataframe
           output of the recommend function of a BayBE object
    obj : BayBE class instance
          the baybe object which provides configuration, targets, etc.
    good_reference_values : list
                  A list of dictionaries which define parameters and respective values
                  which identify what will be considered good values
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
    # Sanity checks for good_bad_ratio
    if good_intervals is None:
        good_intervals = (66, 100)
    if bad_intervals is None:
        bad_intervals = (0, 33)
    if not isinstance(good_intervals, Tuple) or (len(good_intervals) != 2):
        raise TypeError("Parameter good_intervals must be a 2-tuple")
    if not isinstance(bad_intervals, Tuple) or (len(bad_intervals) != 2):
        raise TypeError("Parameter bad_intervals must be a 2-tuple")

    # Sanity check for good_values
    if good_reference_values is None:
        good_reference_values = []

    size = len(data)
    for target in obj.targets:
        # add bad values
        # ToDo consider target direction (min, max, match) here too.
        #  Will be done after multi-target mode has been implemented
        data[target.name] = np.random.randint(bad_intervals[0], bad_intervals[1], size)

        # add good values
        masks = []

        if len(good_reference_values) > 0:
            for pair in good_reference_values:
                if (
                    not isinstance(pair, Dict)
                    or ("Parameter" not in pair.keys())
                    or ("Value" not in pair.keys())
                ):
                    raise TypeError(
                        "Entries in parameter good_values (which is a list) must be"
                        " dictionaries that provides the keys Parameter and Value"
                    )
                mask = data[pair["Parameter"]] == pair["Value"]
                masks.append(mask)

            # ToDo consider target direction (min, max, match) here too.
            #  Will be done after multi-target mode has been implemented
            # Good values will be added where the parameters of the
            # corresponding datapoints

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
