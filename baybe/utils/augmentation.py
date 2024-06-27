"""Utilities related to data augmentation."""

from collections.abc import Sequence
from itertools import permutations

import pandas as pd


def _row_in_df(row: pd.Series | pd.DataFrame, df: pd.DataFrame) -> bool:
    """Check whether a row is fully contained in a dataframe.

    Args:
        row: The row to be checked.
        df: The dataframe to be checked.

    Returns:
        Boolean result.

    Raises:
        ValueError: If `row` is a dataframe that contains more than one row.
    """
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
    """Augment a dataframe if permutation invariant columns are present.

    Indices are preserved so that each augmented row will have the same index as its
    original.

    Args:
        df: The dataframe that should be augmented.
        columns: Sequence indicating the permutation invariant columns.

    Returns:
        The augmented dataframe containing the original one.
    """
    new_rows: list[pd.DataFrame] = []
    for _, row in df.iterrows():
        # Extract the values from the specified columns
        original_values = row[columns].tolist()  # type: ignore[call-overload]

        # Generate all permutations of these values
        all_perms = list(permutations(original_values))

        # For each permutation, create a new row if it's not already in the dataframe
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
    """Augment a dataframe if dependency invariant columns are present.

    This works with the concept of column-values pairs for causing and affected column.
    Any row present where the specified causing column has one of the provided values
    will trigger an augmentation on the affected columns. The latter are augmented by
    going through all their invariant values and adding respective new rows.

    Args:
        df: The dataframe that should be augmented.
        causing: Causing column name and its causing values.
        affected: List of affected columns and their invariant values.

    Returns:
        The augmented dataframe containing the original one.
    """
    new_rows: list[pd.DataFrame] = []

    # Iterate through all rows that have a causing value in the respective column.
    col_causing, vals_causing = causing
    df_filtered = df.loc[df[col_causing].isin(vals_causing), :]
    for _, row in df_filtered.iterrows():
        # Augment the  specific row by growing a dataframe iteratively going through
        # the affected columns. In each iteration augmented rows with that column
        # changed to all possible values are added. If there is more than one affected
        # column, it is important to include the augmented rows stemming from the
        # preceding columns as well.
        original_row = row.to_frame().T

        currently_added = original_row.copy()  # Start with the original row
        for col_affected, vals_invariant in affected:
            to_add = []

            # Go through all previously added rows + the original row
            for _, temp_row in currently_added.iterrows():
                to_add += [
                    new_row
                    for val in vals_invariant
                    if not _row_in_df(
                        new_row := temp_row.to_frame().T.assign(
                            **{col_affected: val}
                        ),  # this takes the current row and replaces the affected value
                        currently_added,
                    )
                ]
            # Update the currently added rows
            currently_added = pd.concat([currently_added] + to_add)

        # Drop first entry because it's the original row and store added rows
        currently_added = currently_added.iloc[1:, :]
        new_rows.append(currently_added)

    augmented_df = pd.concat([df] + new_rows)

    return augmented_df
