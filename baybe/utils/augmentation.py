"""Utilities related to data augmentation."""

from collections.abc import Sequence
from itertools import permutations

import pandas as pd


def _row_in_df(row: pd.Series | pd.DataFrame, df: pd.DataFrame) -> bool:
    """Bla."""
    if isinstance(row, pd.DataFrame):
        if len(row) != 1:
            raise ValueError(
                f"{_row_in_df.__name__} can only be called with pd.Series or "
                f"pd.DataFrame's that have exactly one row."
            )
        row = row.iloc[0]

    return (df == row).all(axis=1).any()


def df_apply_permutation_augmentation(
    df: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Bla."""
    new_rows: list[pd.DataFrame] = []
    for _, row in df.iterrows():
        # Extract the values from the specified columns
        original_values = row[columns].tolist()

        # Generate all permutations of these values
        all_perms = list(permutations(original_values))

        # For each permutation, create a new row if it's not already in the DataFrame
        for perm in all_perms:
            # Create a new row dictionary with the permuted values
            new_row = row.copy().to_frame().T
            new_row[columns] = perm
            if not _row_in_df(new_row, df):
                new_rows.append(new_row)

    augmented_df = pd.concat([df] + new_rows)

    return augmented_df


def df_apply_dependency_augmentation(
    df: pd.DataFrame,
    causing: tuple[str, Sequence],
    affected: Sequence[tuple[str, Sequence]],
) -> pd.DataFrame:
    """Bla."""
    new_rows: list[pd.DataFrame] = []

    # Iterate through all rows that have an invariance-causing value in the respective
    # column
    col_causing, vals_causing = causing
    df_filtered = df.loc[df[col_causing].isin(vals_causing), :]
    for _, row in df_filtered.iterrows():
        # Augment the  specific row by growing a dataframe iteratively going through
        # the affected columns. In each iteration augmented rows with that column
        # changed to all possible values are added.
        original_row = row.to_frame().T

        current_augmented = original_row.copy()
        for col_affected, vals_affected in affected:
            to_add = []
            for _, temp_row in current_augmented.iterrows():
                to_add += [
                    new_row
                    for val in vals_affected
                    if not _row_in_df(
                        new_row := temp_row.to_frame().T.assign(**{col_affected: val}),
                        current_augmented,
                    )
                ]
            current_augmented = pd.concat([current_augmented] + to_add)

        # Drop first entry because it's the original row
        current_augmented = current_augmented.iloc[1:, :]
        new_rows.append(current_augmented)

    augmented_df = pd.concat([df] + new_rows)

    return augmented_df
