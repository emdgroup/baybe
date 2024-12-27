"""Dataframe utilities."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import numpy as np
import pandas as pd

from baybe.targets.base import Target
from baybe.targets.binary import BinaryTarget
from baybe.targets.enum import TargetMode
from baybe.utils.numerical import DTypeFloatNumpy

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.parameters.base import Parameter
    from baybe.targets.base import Target

    _T = TypeVar("_T", bound=Parameter | Target)
    _ArrayLike = TypeVar("_ArrayLike", np.ndarray, Tensor)


# Logging
_logger = logging.getLogger(__name__)


@overload
def to_tensor(x: np.ndarray | pd.DataFrame, /) -> Tensor: ...


@overload
def to_tensor(*x: np.ndarray | pd.DataFrame) -> tuple[Tensor, ...]: ...


def to_tensor(*x: np.ndarray | pd.DataFrame) -> Tensor | tuple[Tensor, ...]:
    """Convert numpy arrays and pandas dataframes to tensors.

    Args:
        *x: The array(s)/dataframe(s) to be converted.

    Returns:
        The provided array(s)/dataframe(s) represented as tensor(s).
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
        torch.from_numpy(
            (xi.values if isinstance(xi, pd.DataFrame) else xi).astype(DTypeFloatNumpy)
        ).to(DTypeFloatTorch)
        for xi in x
    )
    if len(x) == 1:
        out = out[0]
    return out


def add_fake_measurements(
    data: pd.DataFrame,
    targets: Collection[Target],
    good_reference_values: dict[str, list] | None = None,
    good_intervals: dict[str, tuple[float, float]] | None = None,
    bad_intervals: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Add fake measurements to a dataframe which was the result of a recommendation.

    It is possible to specify "good" values, which will be given a better
    target value. With this, the algorithm can be driven towards certain optimal values
    whilst still being random. Useful for testing. Note that the dataframe is changed
    in-place and also returned.

    Args:
        data: A dataframe containing parameter configurations in experimental
            representation, for instance, created via
            :func:`baybe.campaign.Campaign.recommend`.
        targets: The targets for which fake results should be added to the dataframe.
        good_reference_values: A dictionary containing parameter names (= dict keys) and
            respective parameter values (= dict values) that specify what will be
            considered good parameter settings. Conditions for different parameters are
            connected via "and" logic, i.e. the targets will only get good values when
            all parameters have good reference values.
        good_intervals: A dictionary containing target names (= dict keys) and
            respective "good" target value ranges (= dict values) in the form of
            2-tuples. Each target will be assigned a random value in its respective
            target range whenever the corresponding parameters meet the conditions
            specified through ``good_reference_values``.
        bad_intervals: Analogous to ``good_intervals`` but covering the cases where
            the parameters lie outside the conditions specified through
            ``good_reference_values``.

    Returns:
        The modified dataframe.

    Raises:
        ValueError: If good values for a parameter were specified, but this parameter
            is not part of the dataframe.
        ValueError: If the target mode is unrecognized when trying to add fake values.
        TypeError: If the entries in ``good_reference_values`` are not lists.
    """
    # Per default, there are no reference values for good parameters
    if good_reference_values is None:
        good_reference_values = {}

    # Validate input
    for param, vals in good_reference_values.items():
        if param not in data.columns:
            raise ValueError(
                f"When adding fake results you specified good "
                f"values for the parameter '{param}' but this "
                f"parameter is not in the dataframe."
            )
        if not isinstance(vals, list):
            raise TypeError(
                f"Entries in parameter 'good_reference_values' "
                f"(which is a dictionary) must be lists, but you "
                f"provided {vals}."
            )

    # Set defaults for good intervals
    if good_intervals is None:
        good_intervals = {}
        for target in targets:
            if isinstance(target, BinaryTarget):
                continue
            if target.mode is TargetMode.MAX:
                lbound = target.bounds.lower if np.isfinite(target.bounds.lower) else 66
                ubound = (
                    target.bounds.upper if np.isfinite(target.bounds.upper) else 100
                )
                interv = (lbound, ubound)
            elif target.mode is TargetMode.MIN:
                lbound = target.bounds.lower if np.isfinite(target.bounds.lower) else 0
                ubound = target.bounds.upper if np.isfinite(target.bounds.upper) else 33
                interv = (lbound, ubound)
            elif target.mode is TargetMode.MATCH:
                lbound = target.bounds.lower if np.isfinite(target.bounds.lower) else 0
                ubound = (
                    target.bounds.upper if np.isfinite(target.bounds.upper) else 100
                )
                interv = (
                    lbound + 0.4 * (ubound - lbound),
                    lbound + 0.6 * (ubound - lbound),
                )
            else:
                raise ValueError(
                    "Unrecognized target mode when trying to add fake values."
                )
            good_intervals[target.name] = interv

    # Set defaults for bad intervals
    if bad_intervals is None:
        bad_intervals = {}
        for target in targets:
            if isinstance(target, BinaryTarget):
                continue
            if target.mode is TargetMode.MAX:
                lbound = target.bounds.lower if np.isfinite(target.bounds.lower) else 0
                ubound = target.bounds.upper if np.isfinite(target.bounds.upper) else 33
                interv = (lbound, ubound)
            elif target.mode is TargetMode.MIN:
                lbound = target.bounds.lower if np.isfinite(target.bounds.lower) else 66
                ubound = (
                    target.bounds.upper if np.isfinite(target.bounds.upper) else 100
                )
                interv = (lbound, ubound)
            elif target.mode is TargetMode.MATCH:
                lbound = target.bounds.lower if np.isfinite(target.bounds.lower) else 0
                ubound = (
                    target.bounds.upper if np.isfinite(target.bounds.upper) else 100
                )
                interv = (
                    # Take as bad values the interval above the good interval
                    lbound + 0.6 * (ubound - lbound),
                    lbound + 1.2 * (ubound - lbound),
                )
            else:
                raise ValueError(
                    "Unrecognized target mode when trying to add fake values."
                )
            bad_intervals[target.name] = interv

    # Add the fake data for each target
    for target in targets:
        if isinstance(target, BinaryTarget):
            # TODO: When refactoring, take into account good and bad intervals
            data[target.name] = np.random.choice(
                [target.failure_value, target.success_value], size=len(data)
            )
            continue

        # Add bad values
        data[target.name] = np.random.uniform(
            bad_intervals[target.name][0], bad_intervals[target.name][1], len(data)
        )

        # Create masks that identify locations where to place good values
        masks = []
        for param, vals in good_reference_values.items():
            mask = data[param].isin(vals)
            masks.append(mask)

        # Overwrite bad values with good ones using the computed masks
        if len(masks) > 0:
            final_mask = pd.concat(masks, axis=1).all(axis=1)
            data.loc[final_mask, target.name] = np.random.uniform(
                good_intervals[target.name][0],
                good_intervals[target.name][1],
                final_mask.sum(),
            )

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
    numerical_measurements_must_be_within_tolerance: bool,
) -> pd.Index:
    """Match row of the right dataframe to the rows of the left dataframe.

    This is useful for validity checks and to automatically match measurements to
    entries in the search space, e.g. to detect which ones have been measured.
    For categorical parameters, there needs to be an exact match with any of the
    allowed values. For numerical parameters, the user can decide via a flag
    whether values outside the tolerance should be accepted.

    Args:
        left_df: The data that serves as lookup reference.
        right_df: The data that should be checked for matching rows in the left
            dataframe.
        parameters: List of baybe parameter objects that are needed to identify
            potential tolerances.
        numerical_measurements_must_be_within_tolerance: If ``True``, numerical
            parameters are matched with the search space elements only if there is a
            match within the parameter tolerance. If ``False``, the closest match is
            considered, irrespective of the distance.

    Returns:
        The index of the matching rows in ``left_df``.

    Raises:
        ValueError: If some rows are present in the right but not in the left dataframe.
        ValueError: If the input data has invalid values.
    """
    # Assert that all parameters appear in the given dataframe
    if not all(col in right_df.columns for col in left_df.columns):
        raise ValueError(
            "For fuzzy row matching all rows of the right dataframe need to be present"
            " in the left dataframe."
        )

    inds_matched = []

    # Iterate over all input rows
    for ind, row in right_df.iterrows():
        # Check if the row represents a valid input
        valid = True
        for param in parameters:
            if param.is_numerical:
                if numerical_measurements_must_be_within_tolerance:
                    valid &= param.is_in_range(row[param.name])
            else:
                valid &= param.is_in_range(row[param.name])
            if not valid:
                raise ValueError(
                    f"Input data on row with the index {row.name} has invalid "
                    f"values in parameter '{param.name}'. "
                    f"For categorical parameters, values need to exactly match a "
                    f"valid choice defined in your config. "
                    f"For numerical parameters, a match is accepted only if "
                    f"the input value is within the specified tolerance/range. Set "
                    f"the flag 'numerical_measurements_must_be_within_tolerance' "
                    f"to 'False' to disable this behavior."
                )

        # Differentiate category-like and discrete numerical parameters
        cat_cols = [p.name for p in parameters if not p.is_numerical]
        num_cols = [p.name for p in parameters if (p.is_numerical and p.is_discrete)]

        # Discrete parameters must match exactly
        match = left_df[cat_cols].eq(row[cat_cols]).all(axis=1, skipna=False)

        # For numeric parameters, match the entry with the smallest deviation
        # TODO: allow alternative distance metrics
        for col in num_cols:
            abs_diff = (left_df[col] - row[col]).abs()
            match &= abs_diff == abs_diff.min()

        # We expect exactly one match. If that's not the case, print a warning.
        inds_found = left_df.index[match].to_list()
        if len(inds_found) == 0 and len(num_cols) > 0:
            _logger.warning(
                "Input row with index %s could not be matched to the search space. "
                "This could indicate that something went wrong.",
                ind,
            )
        elif len(inds_found) > 1:
            _logger.warning(
                "Input row with index %s has multiple matches with "
                "the search space. This could indicate that something went wrong. "
                "Matching only first occurrence.",
                ind,
            )
            inds_matched.append(inds_found[0])
        else:
            inds_matched.extend(inds_found)

    return pd.Index(inds_matched)


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
            f"The provided dataframe column(s) {extra} cannot be matched against"
            f"the given objects. If you want to transform a dataframe "
            f"with additional columns, explicitly set `allow_extra=True'."
        )

    return [p for p in objects if p.name in df]


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
