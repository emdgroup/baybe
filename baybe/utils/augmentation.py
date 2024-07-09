"""Utilities related to data augmentation."""

from collections.abc import Collection, Sequence
from itertools import permutations, product

import pandas as pd


def df_apply_permutation_augmentation(
    df: pd.DataFrame,
    column_groups: Sequence[Sequence[str]],
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

    *   Result with ``column_groups = [["A1"], ["A2"]]``

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

    *   Result with ``column_groups = [["A1", "B1"], ["A2", "B2"]]``

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
        column_groups: Sequences of permutation invariant columns. The n'th column in
            each group will be permuted together with each n'th column in the other
            groups.

    Returns:
        The augmented dataframe containing the original one. Augmented row indices are
        identical with the index of their original row.

    Raises:
        ValueError: If less than two column groups are given.
        ValueError: If a column group is empty.
        ValueError: If the column groups have differing amounts of entries.
    """
    # Validation
    if len(column_groups) < 2:
        raise ValueError(
            "When augmenting permutation invariance, at least two column sequences "
            "must be given."
        )

    if len({len(seq) for seq in column_groups}) != 1:
        raise ValueError(
            "Permutation augmentation can only work if the amount of columns in each "
            "sequence is the same."
        )
    elif len(column_groups[0]) < 1:
        raise ValueError(
            "Permutation augmentation can only work if each column group has at "
            "least one entry."
        )

    # Augmentation Loop
    new_rows: list[pd.DataFrame] = []
    idx_permutation = list(permutations(range(len(column_groups))))
    for _, row in df.iterrows():
        # For each row in the original df, collect all its permutations
        to_add = []
        for perm in idx_permutation:
            new_row = row.copy()

            # Permute columns, this is done separately for each tuple of columns that
            # belong together
            for deps in map(list, zip(*column_groups)):
                new_row[deps] = row[[deps[k] for k in perm]]

            to_add.append(new_row)

        # Store augmented rows, keeping the index of their original row
        new_rows.append(
            pd.DataFrame(to_add, columns=df.columns, index=[row.name] * len(to_add))
        )

    return pd.concat(new_rows)


def df_apply_dependency_augmentation(
    df: pd.DataFrame,
    causing: tuple[str, Sequence],
    affected: Collection[tuple[str, Sequence]],
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
        The augmented dataframe containing the original one. Augmented row indices are
        identical with the index of their original row.
    """
    new_rows: list[pd.DataFrame] = []
    col_causing, vals_causing = causing
    affected_cols, affected_inv_vals = zip(*affected)
    affected_inv_vals_combinations = list(product(*affected_inv_vals))

    # Iterate through all rows that have a causing value in the respective column.
    for _, row in df.iterrows():
        to_add = []
        if row[col_causing] not in vals_causing:
            # Just keep unaffected rows without augmentation
            to_add.append(row)
        else:
            # Create augmented rows by assigning the affected columns all possible
            # values
            to_add += [
                pd.Series({**row.to_dict(), **dict(zip(affected_cols, values))})
                for values in affected_inv_vals_combinations
            ]

        # Store augmented rows, keeping the index of their original row
        new_rows.append(
            pd.DataFrame(to_add, columns=df.columns, index=[row.name] * len(to_add))
        )

    return pd.concat(new_rows)
