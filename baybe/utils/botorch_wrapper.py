"""Wrapper functionality synthetic BoTorch test functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import pandas as pd
import torch
from botorch.test_functions import SyntheticTestFunction

if TYPE_CHECKING:
    pass


def add_dataframe_layer(
    test_function: SyntheticTestFunction,
    output_columns: Iterable[str],
):
    """Wrap a BoTorch test function into a format suitable for lookups in simulations.

    The wrapper layer removes the dataframe indices, passes the raw numeric values
    as tensor to the test function, and turns the response into a dataframe with the
    given column names. The See :mod:`baybe.simulation.lookup` for details on the
    lookup is used.

    Args:
        test_function: A synthetic test function from BoTorch.
            See https://botorch.org/api/test_functions.html.
        output_columns: The targets for which the test function yields values.

    Returns:
        A wrapped version of the provided function.
    """

    def wrapper(df: pd.DataFrame) -> pd.DataFrame:
        # Call the function with the raw numeric values in the form of a tensor
        response = test_function(torch.from_numpy(df.values))

        # It's not very clear from the BoTorch API how multiple return values would
        # look like, so better catch this case until we have a concrete example.
        if response.ndim > 1:
            raise NotImplementedError(
                "The test function is expected to return values for one target only."
            )

        # Attach the target labels
        return pd.DataFrame(response, columns=output_columns, index=df.index)

    return wrapper
