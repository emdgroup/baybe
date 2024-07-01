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
    df: pd.DataFrame,
    columns: Sequence[str],
    dependents: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Augment a dataframe if permutation invariant columns are present.

    Indices are preserved so that each augmented row will have the same index as its
    original. `dependent` columns are augmented in the same order as the `columns`.

    *   Original

        +---+---+---+---+
        | A | B | C | D |
        +===+===+===+===+
        | a | b | x | y |
        +---+---+---+---+
        | b | a | x | z |
        +---+---+---+---+

    *   Result with ``columns = ["A", "B"]``

        +---+---+---+---+
        | A | B | C | D |
        +===+===+===+===+
        | a | b | x | y |
        +---+---+---+---+
        | b | a | x | z |
        +---+---+---+---+
        | b | a | x | y |
        +---+---+---+---+
        | a | b | x | z |
        +---+---+---+---+

    *   Result with ``columns = ["A", "B"]``, ``dependents = ["C", "D"]``

        +---+---+---+---+
        | A | B | C | D |
        +===+===+===+===+
        | a | b | x | y |
        +---+---+---+---+
        | b | a | x | z |
        +---+---+---+---+
        | b | a | y | x |
        +---+---+---+---+
        | a | b | z | x |
        +---+---+---+---+

    Args:
        df: The dataframe that should be augmented.
        columns: The permutation invariant columns.
        dependents: Columns that are connected to `columns` and should be permuted in
            the same manner.

    Returns:
        The augmented dataframe containing the original one.

    Raises:
        ValueError: If `dependents` has length incompatible with `columns`.
    """
    dependents = dependents or []
    new_rows: list[pd.DataFrame] = []

    if dependents and len(columns) != len(dependents):
        raise ValueError(
            "When augmenting permutation invariance with dependent columns, there must "
            "be exactly the same amount of 'dependents' as there are 'columns'."
        )

    for _, row in df.iterrows():
        # Extract the values from the specified columns
        original_values = row[columns].tolist()  # type: ignore[call-overload]
        dependent_values = row[dependents].tolist() if dependents else None  # type: ignore[call-overload]

        # Generate all permutations of these values
        column_perms = list(permutations(original_values))
        dependent_perms = (
            list(permutations(dependent_values)) if dependent_values else None
        )

        # For each permutation, create a new row if it's not already in the dataframe
        for k, perm in enumerate(column_perms):
            # Create a new row dictionary with the permuted values
            new_row = pd.DataFrame([row])
            new_row[columns] = perm
            if dependent_perms:
                new_row[dependents] = dependent_perms[k]

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

    *   Original

        +---+---+---+---+
        | A | B | C | D |
        +===+===+===+===+
        | 0 | 2 | 5 | y |
        +---+---+---+---+
        | 1 | 3 | 5 | z |
        +---+---+---+---+

    *   Result with ``causing = ("A", [0])``, ``affected = [("B", [2,3,4])]``

        +---+---+---+---+
        | A | B | C | D |
        +===+===+===+===+
        | 0 | 2 | 5 | y |
        +---+---+---+---+
        | 1 | 3 | 5 | z |
        +---+---+---+---+
        | 0 | 3 | 5 | y |
        +---+---+---+---+
        | 0 | 4 | 5 | y |
        +---+---+---+---+

    *   Result with ``causing = ("A", [0, 1])`, `affected = [("B", [2,3])]``

        +---+---+---+---+
        | A | B | C | D |
        +===+===+===+===+
        | 0 | 2 | 5 | y |
        +---+---+---+---+
        | 1 | 3 | 5 | z |
        +---+---+---+---+
        | 0 | 3 | 5 | y |
        +---+---+---+---+
        | 1 | 2 | 5 | z |
        +---+---+---+---+

    *   Result with ``causing = ("A", [0])`, `affected = [("B", [2,3]), ("C", [5, 6])]``

        +---+---+---+---+
        | A | B | C | D |
        +===+===+===+===+
        | 0 | 2 | 5 | y |
        +---+---+---+---+
        | 1 | 3 | 5 | z |
        +---+---+---+---+
        | 0 | 3 | 5 | y |
        +---+---+---+---+
        | 0 | 2 | 6 | y |
        +---+---+---+---+
        | 0 | 3 | 6 | y |
        +---+---+---+---+

    Args:
        df: The dataframe that should be augmented.
        causing: Causing column name and its causing values.
        affected: Affected columns and their invariant values.

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
        original_row = pd.DataFrame([row])

        currently_added = original_row.copy()  # Start with the original row
        for col_affected, vals_invariant in affected:
            to_add = []

            # Go through all previously added rows + the original row
            for _, temp_row in currently_added.iterrows():
                to_add += [
                    new_row
                    for val in vals_invariant
                    if not _row_in_df(
                        new_row := temp_row.pipe(lambda x: pd.DataFrame([x])).assign(
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
