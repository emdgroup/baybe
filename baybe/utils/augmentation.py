"""Utilities related to data augmentation."""

from collections.abc import Sequence
from itertools import permutations, product

import pandas as pd


def _row_in_df(row: pd.Series | pd.DataFrame, df: pd.DataFrame) -> bool:
    """Check whether a row is fully contained in a dataframe.

    Args:
        row: The row to be checked.
        df: The dataframe to be checked.

    Returns:
        Boolean result.

    Raises:
        ValueError: If ``row`` is a dataframe that contains more than one row.
    """
    if isinstance(row, pd.DataFrame):
        if len(row) != 1:
            raise ValueError(
                f"{_row_in_df.__name__} can only be called with pd.Series or "
                f"pd.DataFrames that have exactly one row."
            )
        row = row.iloc[0]

    row = row.reindex(df.columns)
    return (df == row).all(axis=1).any()


def df_apply_permutation_augmentation(
    df: pd.DataFrame,
    columns: Sequence[Sequence[str]],
) -> pd.DataFrame:
    """Augment a dataframe if permutation invariant columns are present.

    *   Original

        +----+----+----+----+
        | A1 | A2 | B1 | B2 |
        +====+====+====+====+
        | a  | b  | x  | y  |
        +----+----+----+----+
        | b  | a  | x  | z  |
        +----+----+----+----+

    *   Result with ``columns = [["A1"], ["A2"]]``

        +----+----+----+----+
        | A1 | A2 | B1 | B2 |
        +====+====+====+====+
        | a  | b  | x  | y  |
        +----+----+----+----+
        | b  | a  | x  | z  |
        +----+----+----+----+
        | b  | a  | x  | y  |
        +----+----+----+----+
        | a  | b  | x  | z  |
        +----+----+----+----+

    *   Result with ``columns = [["A1", "B1"], ["A2", "B2"]]``

        +----+----+----+----+
        | A1 | A2 | B1 | B2 |
        +====+====+====+====+
        | a  | b  | x  | y  |
        +----+----+----+----+
        | b  | a  | x  | z  |
        +----+----+----+----+
        | b  | a  | y  | x  |
        +----+----+----+----+
        | a  | b  | z  | x  |
        +----+----+----+----+

    Args:
        df: The dataframe that should be augmented.
        columns: Sequences of permutation invariant columns. The n'th column in each
            sequence will be permuted together with each n'th column in the other
            sequences.

    Returns:
        The augmented dataframe containing the original one.

    Raises:
        ValueError: If ``dependents`` has length incompatible with ``columns``.
        ValueError: If entries in ``dependents`` are not of same length.
    """
    # Validation
    if len(columns) < 2:
        raise ValueError(
            "When augmenting permutation invariance, at least two column sequences "
            "must be given."
        )
    if len({len(seq) for seq in columns}) != 1 or len(columns[0]) < 1:
        raise ValueError(
            "Permutation augmentation can only work if the amount of columns un each "
            "sequence is the same and the sequences are not empty."
        )

    # Augmentation Loop
    new_rows: list[pd.DataFrame] = []
    idx_permutation = list(permutations(range(len(columns))))
    for _, row in df.iterrows():
        to_add = []
        for _, perm in enumerate(idx_permutation):
            new_row = row.copy()

            # Permute columns
            for deps in map(list, zip(*columns)):
                new_row[deps] = row[[deps[k] for k in perm]]

            # Check whether the new row is an existing permutation
            if not _row_in_df(new_row, df):
                to_add.append(new_row)

        new_rows.append(pd.DataFrame(to_add))
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

    *   Result with ``causing = ("A", [0, 1])``, ``affected = [("B", [2,3])]``

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

    *   Result with ``causing = ("A", [0])``,
        ``affected = [("B", [2,3]), ("C", [5, 6])]``

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
    col_causing, vals_causing = causing
    df_filtered = df.loc[df[col_causing].isin(vals_causing), :]
    affected_cols, affected_inv_vals = zip(*affected)
    affected_inv_vals_combinations = list(product(*affected_inv_vals))

    # Iterate through all rows that have a causing value in the respective column.
    for _, r in df_filtered.iterrows():
        # Create augmented rows
        to_add = [
            pd.Series({**r.to_dict(), **dict(zip(affected_cols, values))})
            for values in affected_inv_vals_combinations
        ]

        # Do not include rows that were present in the original
        to_add = [r2 for r2 in to_add if not _row_in_df(r2, df_filtered)]
        new_rows.append(pd.DataFrame(to_add))

    augmented_df = pd.concat([df] + new_rows)

    return augmented_df
