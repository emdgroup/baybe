"""Utilities related to data augmentation."""

from collections.abc import Sequence
from itertools import permutations

import pandas as pd


def df_apply_permutation_augmentation(
    df: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Bla."""
    new_rows: list[pd.DataFrame] = []
    for index, row in df.iterrows():
        # Extract the values from the specified columns
        original_values = row[columns].tolist()

        # Generate all permutations of these values
        all_perms = list(permutations(original_values))

        # For each permutation, create a new row if it's not already in the DataFrame
        for perm in all_perms:
            # Create a new row dictionary with the permuted values
            new_row = row.copy().to_frame().T
            new_row[columns] = perm
            new_rows.append(new_row)

    augmented_df = pd.concat([df] + new_rows)

    # Drop duplicates if any created inadvertently
    augmented_df.drop_duplicates(inplace=True)

    return augmented_df
