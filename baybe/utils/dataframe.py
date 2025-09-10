"""Dataframe utilities."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np
import pandas as pd

from baybe.exceptions import InputDataTypeWarning, SearchSpaceMatchWarning
from baybe.parameters.base import Parameter
from baybe.utils.numerical import DTypeFloatNumpy

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.targets.base import Target

    _T = TypeVar("_T", bound=Parameter | Target)
    _ArrayLike = TypeVar("_ArrayLike", np.ndarray, Tensor)

_ConvertibleToTensor = int | float | np.ndarray | pd.Series | pd.DataFrame


@overload
def to_tensor(x: _ConvertibleToTensor, /) -> Tensor: ...


@overload
def to_tensor(*x: _ConvertibleToTensor) -> tuple[Tensor, ...]: ...


def to_tensor(*x: _ConvertibleToTensor) -> Tensor | tuple[Tensor, ...]:
    """Convert ints, floats, numpy arrays and pandas series/dataframes to tensors.

    Args:
        *x: The int(s)/float(s)/array(s)/series/dataframe(s) to be converted.

    Returns:
        The provided inputs represented as tensor(s).
    """
    # FIXME This function seems to trigger a problem when some columns in either of
    #  the dfs have a dtype other than int or float (e.g. object, bool). This can
    #  weirdly happen, even if all values are numeric, e.g. when a target column is
    #  looked up from a df in simulation, it can have dtype object even if it's all
    #  floats. As a simple fix (this seems to be the most reasonable place to take
    #  care of this) df.values has been changed to df.values.astype(float),
    #  even though this seems like double casting here.
    import torch

    from baybe.utils.torch import DTypeFloatTorch

    out = tuple(
        torch.tensor(xi, dtype=DTypeFloatTorch)
        if isinstance(xi, (int, float))
        else torch.from_numpy(
            (xi.values if isinstance(xi, (pd.Series, pd.DataFrame)) else xi).astype(
                DTypeFloatNumpy
            )
        ).to(DTypeFloatTorch)
        for xi in x
    )
    if len(x) == 1:
        out = out[0]
    return out


def add_fake_measurements(
    data: pd.DataFrame, targets: Iterable[Target]
) -> pd.DataFrame:
    """Add in-place fake target values to a given dataframe.

    Args:
        data: The dataframe which to augment with fake target values.
        targets: The targets for which fake results should be added.

    Returns:
        The modified dataframe.
    """
    from baybe.targets.binary import BinaryTarget
    from baybe.targets.numerical import NumericalTarget

    for target in targets:
        if isinstance(target, BinaryTarget):
            data[target.name] = np.random.choice(
                [target.failure_value, target.success_value], size=len(data)
            )
        elif isinstance(target, NumericalTarget):
            data[target.name] = np.random.uniform(-100, 100, size=len(data))

    return data


def add_parameter_noise(
    data: pd.DataFrame,
    parameters: Iterable[Parameter],
    noise_type: Literal["absolute", "relative_percent"] = "absolute",
    noise_level: float = 1.0,
) -> pd.DataFrame:
    """Apply uniform noise to the parameter values of a recommendation frame.

    The noise can be additive or multiplicative.
    This can be used to simulate experimental noise or imperfect user input containing
    numerical parameter values that differ from the recommendations. Note that the
    dataframe is changed in-place and also returned.

    Args:
        data: Output of the ``recommend`` function of a ``Campaign`` object, see
            :func:`baybe.campaign.Campaign.recommend`.
        parameters: The parameters for which the values shall be corrupted.
        noise_type: Defines whether the noise should be additive or multiplicative.
        noise_level: Level/magnitude of the noise. Must be provided as numerical value
            for noise type ``absolute`` and as percentage for noise type
            ``relative_percent``.

    Returns:
        The modified dataframe.

    Raises:
        ValueError: If ``noise_type`` is neither ``absolute`` nor
            ``relative_percent``.
    """
    # Validate input
    if noise_type not in ("relative_percent", "absolute"):
        raise ValueError(
            f"Parameter 'noise_type' was {noise_type} but must be either "
            "'absolute' or 'relative_percent'."
        )

    for param in (p for p in parameters if p.is_numerical):
        # Add selected noise type
        if noise_type == "relative_percent":
            data[param.name] *= np.random.uniform(
                1.0 - noise_level / 100.0, 1.0 + noise_level / 100.0, len(data)
            )
        elif noise_type == "absolute":
            data[param.name] += np.random.uniform(-noise_level, noise_level, len(data))

        # Respect continuous intervals
        if param.is_continuous:
            data[param.name] = data[param.name].clip(
                param.bounds.lower, param.bounds.upper
            )

    return data


def create_fake_input(
    parameters: Iterable[Parameter],
    targets: Iterable[Target],
    n_rows: int = 1,
    **kwargs: Any,
) -> pd.DataFrame:
    """Create fake valid input for :meth:`baybe.campaign.Campaign.add_measurements`.

    If noisy parameter values are desired, it is recommended to apply
    :func:`baybe.utils.dataframe.add_parameter_noise` to the output of this function.

    Args:
        parameters: The parameters.
        targets: The targets.
        n_rows: Number of desired rows.
        **kwargs: Additional arguments to be passed to
            :func:`baybe.utils.dataframe.add_fake_measurements`.

    Returns:
        Dataframe corresponding to fake measurement input.

    Raises:
        ValueError: If less than one row was requested.
    """
    # Assert at least one fake entry is being generated
    if n_rows < 1:
        raise ValueError(
            f"'{create_fake_input.__name__}' must at least create one row, but the "
            f"requested number was: {n_rows}."
        )

    # Create fake parameter values from their definitions
    content = {}
    for p in parameters:
        if p.is_discrete:
            vals = np.random.choice(p.values, n_rows, replace=True)
        else:
            vals = np.random.uniform(p.bounds.lower, p.bounds.upper, n_rows)

        content[p.name] = vals

    data = pd.DataFrame.from_dict(content)

    # Add fake target values
    add_fake_measurements(data, targets, **kwargs)

    return data


def df_drop_single_value_columns(
    df: pd.DataFrame, lst_exclude: list = None
) -> pd.DataFrame:
    """Drop dataframe columns with zero variance.

    Args:
        df: The dataframe to be cleaned.
        lst_exclude: List of column names that are excluded from this filter.

    Returns:
        The cleaned dataframe.
    """
    if lst_exclude is None:
        lst_exclude = []

    to_keep = []
    for col in df.columns:
        if (col in lst_exclude) or (df[col].nunique() > 1):
            to_keep.append(col)

    return df[to_keep]


def df_drop_string_columns(
    df: pd.DataFrame, ignore_list: list[str] | None = None
) -> pd.DataFrame:
    """Drop dataframe columns with string values.

    Args:
        df: The dataframe to be cleaned.
        ignore_list: List of columns that should not be dropped, even if they include
            string values.

    Returns:
        The cleaned dataframe.
    """
    ignore_list = ignore_list or []
    no_string = ~df.applymap(lambda x: isinstance(x, str)).any()
    no_string = no_string[no_string].index
    to_keep = set(no_string).union(set(ignore_list))
    ordered_cols = [col for col in df if col in to_keep]
    return df[ordered_cols]


def df_uncorrelated_features(
    df: pd.DataFrame, exclude_list: list[str] | None = None, threshold: float = 0.7
):
    """Return an uncorrelated set of features.

    Adapted from edbo (https://github.com/b-shields/edbo ,
    https://doi.org/10.1038/s41586-021-03213-y).

    Args:
        df: The dataframe to be cleaned
        exclude_list: If provided this defines the columns that should be ignored
        threshold: Threshold for column-column correlation above which columns should
            be dropped

    Returns:
        A new dataframe
    """
    # TODO: revise or replace with VRE method

    if exclude_list is None:
        data = df.copy()
    else:
        data = df.drop(columns=exclude_list)

    corr = data.corr().abs()
    to_keep = []
    for i in range(len(corr.iloc[:, 0])):
        above = corr.iloc[:i, i]
        if len(to_keep) > 0:
            above = above[to_keep]
        if len(above[above < threshold]) == len(above):
            to_keep.append(corr.columns.values[i])

    data = data[to_keep]

    if exclude_list is not None:
        data[exclude_list] = df.loc[:, exclude_list]

    return data


def add_noise_to_perturb_degenerate_rows(
    df: pd.DataFrame, noise_ratio: float = 0.001
) -> pd.DataFrame:
    """Add noise to degenerate rows to make them numerically distinguishable.

    Note that the dataframe is changed in-place and also returned. The dataframe is
    left untouched if no rows are degenerate.

    Args:
        df: The dataframe to be modified.
        noise_ratio: The magnitude of generated uniform noise relative to the
            min-max range of values for each column.

    Returns:
        The modified dataframe.

    Raises:
        TypeError: If the provided dataframe has non-numerical content.
    """
    # Find degenerate rows, exit if there are none
    degen_rows = df.duplicated(keep=False)
    if not degen_rows.any():
        return df

    # Assert that the input is purely numerical
    if any(df[col].dtype.kind not in "iufb" for col in df.columns):
        raise TypeError(
            f"'{add_noise_to_perturb_degenerate_rows.__name__}' only supports purely "
            f"numerical dataframes."
        )

    # Find the min-max range for each column. Constant columns will be assigned a range
    # of 1 as fallback as otherwise they would be left untouched
    column_ranges = df.max() - df.min()
    column_ranges = column_ranges.replace(0, 1)

    # Generate noise
    noise = np.random.uniform(
        -noise_ratio, noise_ratio, size=(degen_rows.sum(), df.shape[1])
    )
    noise_df = pd.DataFrame(noise, columns=df.columns, index=df.index[degen_rows])

    # Scale noise by column ranges and add it to the original dataframe
    noise_df = noise_df * column_ranges
    df.loc[degen_rows] += noise_df

    return df


def fuzzy_row_match(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    parameters: Sequence[Parameter],
) -> pd.Index:
    """Match rows of the right dataframe to rows of the left dataframe.

    This is useful for matching measurements to entries in the search space, e.g. to
    detect which ones have been measured. For categorical parameters, there needs to be
    an exact match with any of the allowed values. For numerical parameters, the points
    with the smallest deviation are considered.

    Note:
        This function assumes that the dataframes contain only allowed values as
        specified in the parameter objects. No further validation to assert this is
        done.

    Args:
        left_df: The data that serves as lookup reference.
        right_df: The data that is checked for matching rows in the left dataframe.
        parameters: Parameter objects that identify the relevant column names and how
            matching is performed.

    Returns:
        The index of the matching rows in ``left_df``.

    Raises:
        ValueError: If either ``left_df`` or ``right_df`` does not contain columns for
            each entry in parameters.
    """
    # Separate columns types
    cat_cols = {p.name for p in parameters if (not p.is_numerical and p.is_discrete)}
    num_cols = {p.name for p in parameters if (p.is_numerical and p.is_discrete)}
    non_discrete_cols = {p.name for p in parameters if not p.is_discrete}

    # Assert that all parameters appear in the given dataframes
    if diff := (cat_cols | num_cols).difference(left_df.columns):
        raise ValueError(
            f"For fuzzy row matching, all discrete parameters need to have a "
            f"corresponding column in the left dataframe. Parameters not found: {diff})"
        )
    if diff := (cat_cols | num_cols).difference(right_df.columns):
        raise ValueError(
            f"For fuzzy row matching, all discrete parameters need to have a "
            f"corresponding column in the right dataframe. Parameters not found: "
            f"{diff})"
        )

    provided_cols = {p.name for p in parameters}
    allowed_cols = cat_cols | num_cols | non_discrete_cols
    assert allowed_cols == provided_cols, (
        f"There are parameter types that would be silently ignored: "
        f"{provided_cols.difference(allowed_cols)}"
    )

    # Ensure the dtype consistency in the dataframes
    left_df = normalize_input_dtypes(left_df, parameters)
    right_df = normalize_input_dtypes(right_df, parameters)

    # Initialize the match matrix. We will later filter it down using other
    # matrices (representing the matches for individual parameters) via logical 'and'.
    match_matrix = pd.DataFrame(
        True, index=right_df.index, columns=left_df.index, dtype=bool
    )

    # Match categorical parameters
    for col in cat_cols:
        # Per categorical parameter, this identifies matches between all elements of
        # left and right and stores them in a matrix.
        match_matrix &= right_df[col].values[:, None] == left_df[col].values[None, :]

    # Match numerical parameters
    for col in num_cols:
        # Compute absolute differences and find the minimum difference
        abs_diff = np.abs(right_df[col].values[:, None] - left_df[col].values[None, :])
        min_diff = abs_diff.min(axis=1, keepdims=True)
        match_matrix &= abs_diff == min_diff

    # Find the matching indices. If a right row is not matched to any of the rows in
    # left, idxmax would return the first index of left_df. Hence, we remember these
    # cases and drop them explicitly.
    matched_indices = pd.Index(match_matrix.idxmax(axis=1).values)
    mask_no_match = ~match_matrix.any(axis=1)
    matched_indices = matched_indices[~mask_no_match]

    mask_multiple_matches = match_matrix.sum(axis=1) > 1
    if multiple_match_indices := right_df.index[mask_multiple_matches].tolist():
        w = SearchSpaceMatchWarning(
            f"Some input rows have multiple matches with the search space. "
            f"Matching only first occurrence for these rows. Indices with multiple "
            f"matches: {multiple_match_indices}",
            right_df.loc[multiple_match_indices],
        )
        warnings.warn(w)

    return matched_indices


def pretty_print_df(
    df: pd.DataFrame,
    max_rows: int | None = 6,
    max_columns: int | None = 4,
    max_colwidth: int | None = 16,
    precision: int = 3,
) -> str:
    """Convert a dataframe into a pretty/readable format.

    This function returns a customized str representation of the dataframe.
    In case the dataframe is empty, it returns a corresponding statement.

    Args:
        df: The dataframe to be printed.
        max_rows: Maximum number of rows to display.
        max_columns: Maximum number of columns to display.
        max_colwidth: Maximum width of an individual column.
        precision: Number of digits to which numbers should be rounded.

    Returns:
        The values to be printed as a str table.
    """
    # Get custom str representation via pandas option_context
    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        max_columns,
        "display.max_colwidth",
        max_colwidth,
        "display.precision",
        precision,
        "expand_frame_repr",
        False,
    ):
        # Pandas does not truncate the names of columns with long names, which makes
        # computational representations barely readable in some of the examples. Hence,
        # we truncate them manually. For details, see
        # https://stackoverflow.com/questions/64976267/pandas-truncate-column-names)
        str_df = df.rename(
            columns=lambda x: x[:max_colwidth],
        )
        str_df = str(str_df)
    return str_df


def get_transform_objects(
    df: pd.DataFrame,
    objects: Sequence[_T],
    /,
    *,
    allow_missing: bool = False,
    allow_extra: bool = False,
) -> list[_T]:
    """Extract the objects relevant for transforming a given dataframe.

    The passed objects are assumed to have corresponding columns in the given dataframe,
    identified through their name attribute. The function returns the subset of objects
    that have a corresponding column in the dataframe and thus provide the necessary
    information for transforming the dataframe.

    Args:
        df: The dataframe to be searched for corresponding columns.
        objects: A collection of objects to be considered for transformation (provided
            they have a match in the given dataframe).
        allow_missing: Flag controlling if objects are allowed to have no corresponding
            columns in the dataframe.
        allow_extra: Flag controlling if the dataframe is allowed to have columns
            that have no corresponding objects.

    Raises:
        ValueError: If the given objects and dataframe are not compatible
            under the specified values for the Boolean flags.

    Returns:
        The (subset of) objects that need to be considered for the transformation.
    """
    names = [p.name for p in objects]

    if (not allow_missing) and (missing := set(names) - set(df)):  # type: ignore[arg-type]
        raise ValueError(
            f"The object(s) named {missing} cannot be matched against "
            f"the provided dataframe. If you want to transform a subset of "
            f"columns, explicitly set `allow_missing=True`."
        )

    if (not allow_extra) and (extra := set(df) - set(names)):
        raise ValueError(
            f"The provided dataframe column(s) {extra} cannot be matched against "
            f"the given objects. If you want to transform a dataframe "
            f"with additional columns, explicitly set `allow_extra=True'."
        )

    return [p for p in objects if p.name in df]


def transform_target_columns(
    df: pd.DataFrame,
    targets: Sequence[Target],
    /,
    *,
    allow_missing: bool = False,
    allow_extra: bool = False,
) -> pd.DataFrame:
    """Transform the columns of a dataframe that correspond to objects of type :class:`~baybe.targets.base.Target`.

    For more details, see :func:`baybe.utils.dataframe.get_transform_objects`.
    """  # noqa: E501
    # Extract the relevant part of the dataframe
    targets = get_transform_objects(
        df, targets, allow_missing=allow_missing, allow_extra=allow_extra
    )
    transformed = df[[t.name for t in targets]].copy()

    # Transform all targets individually
    for target in targets:
        transformed[target.name] = target.transform(df[target.name])

    return transformed


def filter_df(
    df: pd.DataFrame, /, to_keep: pd.DataFrame, complement: bool = False
) -> pd.DataFrame:
    """Filter a dataframe based on a second dataframe defining filtering conditions.

    Filtering is done via a join (see ``complement`` argument for details) between the
    input dataframe and the filter dataframe.

    Args:
        df: The dataframe to be filtered.
        to_keep: The dataframe defining the filtering conditions. By default
            (see ``complement`` argument), it defines the rows to be kept in the sense
            of an inner join.
        complement: If ``False``, the filter dataframe determines the rows to be kept
            (i.e. selection via inner join). If ``True``, the filtering mechanism is
            inverted so that the complement set of rows is kept (i.e. selection
            via anti-join).

    Returns:
        A new dataframe containing the result of the filtering process.

    Examples:
        >>> df = pd.DataFrame(
        ...         [[0, "a"], [0, "b"], [1, "a"], [1, "b"]],
        ...         columns=["num", "cat"]
        ... )
        >>> df
           num cat
        0    0   a
        1    0   b
        2    1   a
        3    1   b

        >>> filter_df(df, pd.DataFrame([0], columns=["num"]), complement=False)
           num cat
        0    0   a
        1    0   b

        >>> filter_df(df, pd.DataFrame([0], columns=["num"]), complement=True)
           num cat
        2    1   a
        3    1   b

        >>> filter_df(df, pd.DataFrame(), complement=True)
           num cat
        0    0   a
        1    0   b
        2    1   a
        3    1   b

        >>> filter_df(df, pd.DataFrame(), complement=False)
        Empty DataFrame
        Columns: [num, cat]
        Index: []

    """
    # Handle special case of empty filter
    if to_keep.empty:
        return df if complement else pd.DataFrame(columns=df.columns)

    # Remember original index name
    index_name = df.index.name

    # Identify rows to be dropped
    out = pd.merge(
        df.reset_index(names="_df_index"), to_keep, how="left", indicator=True
    ).set_index("_df_index")
    to_drop = out["_merge"] == ("both" if complement else "left_only")

    # Drop the points
    out.drop(index=out[to_drop].index, inplace=True)
    out.drop("_merge", axis=1, inplace=True)

    # Restore original index name
    out.index.name = index_name

    return out


def arrays_to_dataframes(
    input_labels: Sequence[str],
    output_labels: Sequence[str],
    /,
    use_torch: bool = False,
) -> Callable[
    [Callable[[_ArrayLike], _ArrayLike]], Callable[[pd.DataFrame], pd.DataFrame]
]:
    """Make a decorator for labeling the input/output columns of array-based callables.

    Useful for creating parameter-to-target lookups from array-based logic.
    The decorator transforms a callable designed to work with unlabelled arrays such
    that it can operate with dataframes instead. The original callable is expected to
    accept and return two-dimensional arrays. When decorated, the callable accepts and
    returns dataframes whose columns are mapped to the corresponding arrays based on the
    specified label sequences.

    Args:
        input_labels: The sequence of labels for the input columns.
        output_labels: The sequence of labels for the output columns.
        use_torch: Flag indicating if the callable is to be called with a numpy array
            or with a torch tensor.

    Returns:
        The decorator for the given input and output labels.
    """

    def decorator(
        fn: Callable[[_ArrayLike], _ArrayLike], /
    ) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """Turn an array-based callable into a dataframe-based callable."""

        @functools.wraps(fn)
        def wrapper(df: pd.DataFrame, /) -> pd.DataFrame:
            """Translate to/from an array-based callable using dataframes."""
            array_in = df[list(input_labels)].to_numpy()
            if use_torch:
                import torch

                with torch.no_grad():
                    array_out = fn(torch.from_numpy(array_in)).numpy()
            else:
                array_out = fn(array_in)
            return pd.DataFrame(array_out, columns=list(output_labels), index=df.index)

        return wrapper

    return decorator


class _ValidatedDataFrame(pd.DataFrame):
    """Wrapper indicating the underlying experimental data was already validated."""


def handle_missing_values(
    data: pd.DataFrame, columns: Collection[str], drop: bool = False
) -> pd.DataFrame:
    """Handle missing inputs by dropping corresponding rows or raising an error.

    Args:
        data: Data to be checked.
        columns: The column names to check.
        drop: Whether to drop the corresponding rows instead of raising an error.

    Raises:
        ValueError: If any row contains NaN in the target columns. Only relevant
            if ``drop=False``.

    Returns:
        A new dataframe with the rows containing NaN dropped.
    """
    mask = data[columns].isna().any(axis=1)

    if (not drop) and mask.any():
        raise ValueError(
            f"Incomplete measurements identified by NaN were found in the input, "
            f"but are not supported. Bad input in the rows with these "
            f"indices: {data.index[mask].tolist()}"
        )

    return data.loc[~mask]


def normalize_input_dtypes(df: pd.DataFrame, objects: Iterable[_T], /) -> pd.DataFrame:
    """Ensure that the input dataframe has the expected dtypes for all columns.

    Args:
        df: The input dataframe to be checked.
        objects: The objects for which to check the corresponding column dtypes.

    Returns:
        The original dataframe if there are no dtype issues. Otherwise, a copy with
        columns converted to expected dtypes.
    """

    def needs_float_dtype(obj) -> bool:
        """Check if the object requires float dtype column representation."""
        from baybe.targets.numerical import NumericalTarget

        return (isinstance(obj, Parameter) and obj.is_numerical) or isinstance(
            obj, NumericalTarget
        )

    # Find columns that are not of float dtype but should be
    wrong_cols = [
        o.name
        for o in objects
        if needs_float_dtype(o)
        and o.name in df.columns
        and not pd.api.types.is_float_dtype(df[o.name])
    ]

    # If there are no issues, return the original
    if not wrong_cols:
        return df

    # Make a copy of the dataframe and convert problematic column data types
    warnings.warn(
        f"The following columns have unexpected data types: {wrong_cols}. "
        f"Converting to float internally.",
        InputDataTypeWarning,
    )
    df = df.copy()
    for col in wrong_cols:
        df[col] = df[col].astype(DTypeFloatNumpy)
    return df
