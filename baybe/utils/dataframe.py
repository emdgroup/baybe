"""Dataframe utilities."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    overload,
)

import narwhals.stable.v2 as nw
import numpy as np
import numpy.typing as npt
import pandas as pd
from narwhals.stable.v2.dependencies import is_into_dataframe, is_into_series
from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries
from narwhals.testing import assert_frame_equal

from baybe.constraints.base import DiscreteFilteringConstraint
from baybe.exceptions import IncompatibleArgumentError, InputDataTypeWarning
from baybe.settings import active_settings

if TYPE_CHECKING:
    from narwhals.stable.v2.typing import (
        IntoFrame,
        IntoFrameT,
    )
    from narwhals.typing import IntoBackend
    from torch import Tensor

    from baybe.parameters.base import Parameter
    from baybe.targets.base import Target

    _T = TypeVar("_T", bound=Parameter | Target)
    _ArrayLike = TypeVar("_ArrayLike", np.ndarray, Tensor)

    _IntoTensor: TypeAlias = (
        int | float | np.ndarray | nw.Series | nw.DataFrame | IntoSeries | IntoDataFrame
    )

Filter: TypeAlias = (
    nw.Expr
    | IntoSeries
    | Sequence[bool]
    | npt.NDArray[np.bool_]
    | DiscreteFilteringConstraint
    | IntoDataFrame
)
"""A criterion for filtering a dataframe.

* :class:`narwhals.stable.v2.Expr`: A Boolean expression.
* A Boolean mask (:class:`~narwhals.stable.v2.typing.IntoSeries`, a plain
  ``Sequence[bool]``, or a Boolean numpy array): Selects rows positionally.
* :class:`~baybe.constraints.base.DiscreteFilteringConstraint`: Applies the
  constraint's validity criterion.
* A dataframe (:class:`~narwhals.stable.v2.typing.IntoDataFrame`): Selects
  rows whose values match a row of the given dataframe in all of its
  columns (semi-join semantics). Must use the same dataframe backend as
  the filtered frame.
"""

_SeriesOrFrameT = TypeVar("_SeriesOrFrameT", nw.Series, nw.DataFrame)


def _is_filter(x: Any, /) -> bool:
    """Determine whether an object can be used as a dataframe filter."""
    return (
        isinstance(x, nw.Expr | DiscreteFilteringConstraint)
        or is_into_dataframe(x)
        or _is_boolean_mask(x)
    )


@overload
def to_tensor(x: _IntoTensor, /) -> Tensor: ...


@overload
def to_tensor(*x: _IntoTensor) -> tuple[Tensor, ...]: ...


def to_tensor(*x: _IntoTensor) -> Tensor | tuple[Tensor, ...]:
    """Convert ints, floats, numpy arrays and series/dataframes to tensors.

    Args:
        *x: The int(s)/float(s)/array(s)/series/dataframe(s) to be converted.

    Returns:
        The provided inputs represented as tensor(s).

        If possible, returns views of the original data, otherwise creates copies.
        The latter will be the case if the input data is not contiguous in memory, not
        of the configured Torch dtype, or uses negative strides.
    """
    import torch

    from baybe.utils.torch import torch_to_numpy_dtype_mapping

    torch_dtype = active_settings.DTypeFloatTorch
    nw_dtype = nw.Float32 if torch_dtype == torch.float32 else nw.Float64
    numpy_dtype = torch_to_numpy_dtype_mapping[torch_dtype]

    def _convert(x: _IntoTensor, /) -> Tensor:
        match x:
            case int() | float():
                return torch.tensor(x, dtype=active_settings.DTypeFloatTorch)
            case np.ndarray():
                # tensors with negative strides are not supported by PyTorch
                fix_strides = any(s < 0 for s in x.strides)
                x = x.astype(numpy_dtype, copy=fix_strides)
                # Copy if read-only (possible under pandas 3 Copy-on-Write)
                if not x.flags.writeable:
                    x = x.copy()
                tensor = torch.from_numpy(x)
            case nw.DataFrame():
                return _convert(x.select(nw.all().cast(nw_dtype)).to_numpy())
            case nw.Series():
                return _convert(x.cast(nw_dtype).to_numpy())
            case _:
                return _convert(nw.from_native(x, allow_series=True))

        # The `contiguous` call brings us closest to getting reproducible
        # results downstream in the torch ecosystem
        # https://github.com/meta-pytorch/botorch/issues/3046
        return tensor.contiguous()

    converted = tuple(_convert(xi) for xi in x)

    if len(x) == 1:
        return converted[0]

    return converted


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
    from baybe.parameters.numerical import NumericalContinuousParameter

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
        if isinstance(param, NumericalContinuousParameter):
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
    from baybe.parameters.base import DiscreteParameter
    from baybe.parameters.numerical import NumericalContinuousParameter

    # Assert at least one fake entry is being generated
    if n_rows < 1:
        raise ValueError(
            f"'{create_fake_input.__name__}' must at least create one row, but the "
            f"requested number was: {n_rows}."
        )

    # Create fake parameter values from their definitions
    content = {}
    for p in parameters:
        if isinstance(p, DiscreteParameter):
            vals = np.random.choice(p.values, n_rows, replace=True)
        elif isinstance(p, NumericalContinuousParameter):
            vals = np.random.uniform(p.bounds.lower, p.bounds.upper, n_rows)
        else:
            raise NotImplementedError(f"Unsupported parameter type: {type(p).__name__}")

        content[p.name] = vals

    data = pd.DataFrame.from_dict(content)

    # Add fake target values
    add_fake_measurements(data, targets, **kwargs)

    return data


def df_drop_single_value_columns(
    df: pd.DataFrame, lst_exclude: list | None = None
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
    no_string = ~df.map(lambda x: isinstance(x, str)).any()
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


def _fuzzy_anti_join(
    left: IntoFrame, right: IntoDataFrame, tolerances: dict[str, float], /
) -> nw.LazyFrame:
    """Return the rows of that do not fuzzily match any row of the second dataframe.

    Columns for which a tolerance is specified are considered a match if the absolute
    deviation is within that tolerance. All other columns require an exact match.

    Args:
        left: The data to be filtered.
        right: The data to be matched against.
        tolerances: A mapping from column name to the absolute tolerance within
            which a deviation in that column is still considered a match.

    Returns:
        The rows of the first dataframe that have no matching row in the second.

    Raises:
        ValueError: If the two dataframes share no columns at all.
        ValueError: If a tolerance is specified for a column that is not shared by both
            dataframes.
    """
    left_lf = nw.from_native(left).lazy()
    right_eager = nw.from_native(right, eager_only=True)

    left_cols = set(left_lf.collect_schema().names())
    right_cols = set(right_eager.columns)
    common_cols = left_cols & right_cols

    if not common_cols:
        raise ValueError(
            "The two dataframes have no columns in common, so no matching is possible."
        )
    if missing := set(tolerances) - common_cols:
        raise ValueError(
            f"A tolerance was specified for the following column(s), but they are "
            f"not shared by both dataframes: {missing}"
        )

    exact_cols = common_cols - set(tolerances)
    fuzzy_cols = common_cols & set(tolerances)

    if right_eager.is_empty():
        return left_lf

    if not fuzzy_cols:
        return left_lf.join(right_eager.lazy(), how="anti", on=list(exact_cols))

    def tolerance_match(other: Mapping[str, Any], /) -> nw.Expr:
        """Build an expression checking if all fuzzy columns match ``other``."""
        return nw.all_horizontal(
            *(
                (nw.col(col) - other[col]).abs() <= tolerances[col]
                for col in fuzzy_cols
            ),
            ignore_nulls=True,
        )

    if exact_cols:
        joined = left_lf.join(right_eager.lazy(), how="inner", on=list(exact_cols))
        joined = joined.filter(
            tolerance_match({col: nw.col(f"{col}_right") for col in fuzzy_cols})
        )
        matched = joined.select(list(left_cols)).unique()
        return left_lf.join(matched, how="anti", on=list(left_cols))

    row_exprs = [tolerance_match(row) for row in right_eager.iter_rows(named=True)]
    return left_lf.filter(~nw.any_horizontal(*row_exprs, ignore_nulls=True))


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
    df: IntoDataFrame,
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
    columns = set(df.columns)
    names = [p.name for p in objects]

    if (not allow_missing) and (missing := set(names) - columns):
        raise ValueError(
            f"The object(s) named {missing} cannot be matched against "
            f"the provided dataframe. If you want to transform a subset of "
            f"columns, explicitly set 'allow_missing=True'."
        )

    if (not allow_extra) and (extra := columns - set(names)):
        raise ValueError(
            f"The provided dataframe column(s) {extra} cannot be matched against "
            f"the given objects. If you want to transform a dataframe "
            f"with additional columns, explicitly set 'allow_extra=True'."
        )

    return [p for p in objects if p.name in columns]


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


def _is_boolean_mask(x: Any, /) -> bool:
    """Determine whether a given object is a Boolean mask.

    Args:
        x: The object to inspect.

    Returns:
        ``True`` if the object is a series (narwhals or native) with Boolean dtype,
        a Boolean numpy array, or a plain sequence of ``bool``-like values.
        ``False`` otherwise.
    """
    if is_into_series(x):
        return nw.from_native(x, series_only=True).dtype == nw.Boolean
    if isinstance(x, np.ndarray):
        return x.dtype == bool
    if isinstance(x, Sequence):
        return all(isinstance(v, bool | np.bool_) for v in x)
    return False


def _apply_filter(lf: nw.LazyFrame, filter_: Filter, /) -> nw.LazyFrame:
    """Narrow a lazyframe down to the rows matching a filter criterion.

    Args:
        lf: The lazyframe to be filtered.
        filter_: The filter criterion.

    Returns:
        The filtered lazyframe.

    Raises:
        IncompatibleArgumentError: If the filter criterion is a dataframe using a
            different dataframe backend than the frame to be filtered.
        TypeError: If the filter criterion cannot be interpreted as a filter.
    """
    from baybe.constraints.base import DiscreteFilteringConstraint

    match filter_:
        case nw.Expr():
            return lf.filter(filter_)

        case DiscreteFilteringConstraint():
            # TODO: Simplify once constraints expose a narwhals-native expression
            #   (replacing the current pandas-index/Polars-expression duality).
            eager = lf.collect()
            native = eager.to_pandas()
            valid = native.loc[filter_.get_valid(native)]
            return _df_with_backend(
                nw.from_native(valid, eager_only=True), nw.get_native_namespace(eager)
            ).lazy()

        case _ if is_into_dataframe(filter_):
            other = nw.from_native(filter_, eager_only=True)
            lf_backend = nw.get_native_namespace(lf)
            other_backend = nw.get_native_namespace(other)
            if other_backend != lf_backend:
                raise IncompatibleArgumentError(
                    f"The dataframe passed as a filter uses the "
                    f"'{other_backend.__name__}' backend, but the filtered frame "
                    f"uses '{lf_backend.__name__}'. Convert the filter dataframe "
                    f"to the matching backend before passing it."
                )
            return lf.join(other.lazy(), on=other.columns, how="semi")

        case _ if _is_boolean_mask(filter_):
            mask = (
                nw.from_native(filter_, series_only=True).to_list()
                if is_into_series(filter_)
                else [bool(v) for v in filter_]
            )
            return lf.collect().filter(mask).lazy()

        case _:
            raise TypeError(
                f"'{filter_!r}' of type '{type(filter_).__name__}' cannot be used as "
                f"a filter. See '{Filter}' for the supported types."
            )


def _filter_dataframe(
    lf: nw.LazyFrame, /, *filters: Filter, complement: bool = False
) -> nw.LazyFrame:
    """Filter a lazyframe based on one or more filter criteria.

    Multiple filters are combined conjunctively (logical AND), mirroring
    :meth:`narwhals.stable.v2.LazyFrame.filter`.

    Args:
        lf: The lazyframe to be filtered.
        *filters: One or more filter criteria.
        complement: If ``False``, the rows matching all given filters are kept.
            If ``True``, the complement set is kept instead.

    Returns:
        The filtered lazyframe.
    """
    matched = functools.reduce(_apply_filter, filters, lf)
    if not complement:
        return matched
    return lf.join(matched, on=lf.columns, how="anti")


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

                # Copy if read-only (possible under pandas 3 Copy-on-Write)
                if not array_in.flags.writeable:
                    array_in = array_in.copy()
                with torch.no_grad():
                    array_out = fn(torch.from_numpy(array_in)).numpy()
            else:
                array_out = fn(array_in)
            return pd.DataFrame(array_out, columns=list(output_labels), index=df.index)

        return wrapper

    return decorator


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


def normalize_input_dtypes(
    df: IntoFrameT, objects: Iterable[Parameter | Target], /
) -> IntoFrameT:
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
        from baybe.parameters.base import Parameter
        from baybe.targets.numerical import NumericalTarget

        return (isinstance(obj, Parameter) and obj.is_numerical) or isinstance(
            obj, NumericalTarget
        )

    df_nw = nw.from_native(df)
    schema = df_nw.collect_schema()

    # Find columns that need conversion to float
    cols_to_convert = [
        o.name
        for o in objects
        if needs_float_dtype(o) and o.name in schema and not schema[o.name].is_float()
    ]

    if not cols_to_convert:
        return df

    # Warn only about truly problematic types (not integers)
    warn_cols = [col for col in cols_to_convert if not schema[col].is_integer()]
    if warn_cols:
        warnings.warn(
            f"The following columns have unexpected data types: {warn_cols}. "
            f"Converting to float internally.",
            InputDataTypeWarning,
        )

    # TODO: `DTypeFloatNumpy` is a numpy-specific setting but is used here to control
    #   the float precision of a potentially non-pandas/non-numpy-backed dataframe.
    #   Revisit once a backend-agnostic float precision setting exists.
    float_dtype = (
        nw.Float32 if active_settings.DTypeFloatNumpy is np.float32 else nw.Float64
    )
    return df_nw.cast({col: float_dtype for col in cols_to_convert}).to_native()


def _infer_backend(*frames: IntoFrame | None) -> IntoBackend:
    """Infer the dataframe backend from the first non-``None`` frame.

    Falls back to :attr:`~baybe.settings.Settings.default_dataframe_backend` if all
    provided frames are ``None``.

    Args:
        *frames: The frames to inspect. Accepts both eager and lazy frames.
            Any of them may be ``None``.

    Returns:
        The inferred backend.
    """
    for frame in frames:
        if frame is not None:
            return nw.get_native_namespace(frame)
    return active_settings.default_dataframe_backend


def _copy_index(
    output: _SeriesOrFrameT, source: nw.DataFrame | nw.Series, /
) -> _SeriesOrFrameT:
    """Copy the pandas index from one series/dataframe to another.

    For non-pandas backends this is a no-op, since they have no index concept.

    Args:
        output: The narwhals Series or DataFrame to copy the index onto.
        source: The narwhals Series or DataFrame whose index is to be copied.

    Returns:
        The output with the index copied from the source.
    """
    # TODO: Replace once built-in solution is available
    # https://github.com/narwhals-dev/narwhals/issues/3693
    # https://github.com/narwhals-dev/narwhals/issues/3864
    if (index := nw.maybe_get_index(source)) is not None:
        return nw.maybe_set_index(
            output, index=nw.from_native(pd.Series(index), series_only=True)
        )
    return output


def _df_with_backend(obj: _SeriesOrFrameT, backend: IntoBackend, /) -> _SeriesOrFrameT:
    """Convert a narwhals Series/DataFrame to a different native backend.

    Args:
        obj: The narwhals Series/DataFrame to convert.
        backend: The target backend to convert to.

    Returns:
        The input object converted to the specified backend.
    """
    # TODO: Replace once built-in solution is available
    # https://github.com/narwhals-dev/narwhals/issues/3812

    incoming = nw.Implementation.from_backend(nw.get_native_namespace(obj))
    target = nw.Implementation.from_backend(backend)
    if incoming == target:
        return obj

    if isinstance(obj, nw.Series):
        name = obj.name
        return nw.from_dict(obj.to_frame().to_dict(), backend=backend)[name]  # type: ignore[return-value]
    return nw.from_dict(obj.to_dict(), backend=backend)  # type: ignore[return-value]


def _df_equals(
    df1: nw.DataFrame | nw.LazyFrame, df2: nw.DataFrame | nw.LazyFrame, /
) -> bool:
    """Check if two dataframes are equal."""
    if isinstance(df1, nw.LazyFrame) or isinstance(df2, nw.LazyFrame):
        # TODO: This is a temporary workaround to avoid runtime errors while working on
        #   the candidates framework but needs to be adressed in order to re-enable
        #   recommendation caching (without having to materialize the lazy frames.)
        return False

    # https://github.com/narwhals-dev/narwhals/issues/3715
    try:
        assert_frame_equal(df1, df2)
        return True
    except AssertionError:
        return False
