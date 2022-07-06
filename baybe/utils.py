"""
Collection of small utilities
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from baybe.core import BayBE


def is_valid_smiles(smiles: str):
    """
    Test if a SMILEs string is valid. Currently NOT IMPLEMENTED
    :param smiles: SMILES string to test
    :return: True if the SMILES is valid according to RDKit
    """
    return smiles


def add_fake_results(
    data: pd.DataFrame,
    obj: BayBE,
    good_bad_ratio: Tuple = None,
    good_values: List = None,
):
    """
    Adds fake results to a dataframe based on the targets in a BayBE object

    Currently does not consider the target direction (min, max or match)
    """
    # Sanity checks for good_bad_ratio
    if good_bad_ratio is None:
        good_bad_ratio = (0.2, 0.8)
    if not isinstance(good_bad_ratio, Tuple):
        raise TypeError("Parameter good_bad_ratio must be a tuple")
    if len(good_bad_ratio) != 2:
        raise AssertionError("Parameter good_bad_ratio")

    # Sanity check for good_values
    if good_values is None:
        good_values = []

    size = len(data)
    for target in obj.targets:
        print(target.name)

        # add bad values
        # ToDo consider target direction (min, max, match) here too.
        #  Will be done after multi-target mode has been implemented
        data[target.name] = np.random.randint(0, 100 * good_bad_ratio[0], size)

        # add good values
        for pair in good_values:
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

            # ToDo consider target direction (min, max, match) here too.
            #  Will be done after multi-target mode has been implemented
            data.loc[mask, target.name] = np.random.randint(
                100 * good_bad_ratio[1], 100, mask.sum()
            )
