"""Functionality for different experimental parameter types."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import (
    Any,
    cast,
    ClassVar,
    Dict,
    get_args,
    get_type_hints,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
)

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import deep_iterable, gt, instance_of, lt, min_len
from scipy.spatial.distance import pdist

from baybe.exceptions import EmptySearchSpaceError
from baybe.utils import (
    convert_bounds,
    df_drop_single_value_columns,
    df_uncorrelated_features,
    eq_dataframe,
    get_base_unstructure_hook,
    InfiniteIntervalError,
    Interval,
    SerialMixin,
    unstructure_base,
)
from baybe.utils.chemistry import _MORDRED_INSTALLED, _RDKIT_INSTALLED

if _RDKIT_INSTALLED:
    from baybe.utils import (
        is_valid_smiles,
        smiles_to_fp_features,
        smiles_to_rdkit_features,
    )

    if _MORDRED_INSTALLED:
        from baybe.utils import smiles_to_mordred_features

# TODO[12356]: There should be a better way than registering with the global converter.
# TODO: Think about what is the best approach to handle field unions. That is, when
#  serializing e.g. a field of typy Union[int, float], it must be ensured that the
#  deserialized type is correctly recovered, i.e. that a 1.0 is recovered as a float
#  and not as an int. Potential options:
#   1)  Adding explicit hooks like the ones below (probably registered with a custom
#       converter, though)
#   2)  Adding a converter to the field that resolves the types and ensures that the
#       object always carries a specific type, removing the need for unions in the
#       first place.
#   3)  Waiting for the upcoming "attrs built-in approach". Quote from doc:
#       "In the future, cattrs will gain additional tools to make union handling even
#       easier and automate generating these hooks."
#       https://catt.rs/en/stable/unions.html
cattrs.register_structure_hook(Union[int, float], lambda x, _: float(x))
cattrs.register_structure_hook(Union[bool, float], lambda x, _: x)

# TODO: Introduce encoding enums


def _validate_decorrelation(obj: Any, attribute: Any, value: float) -> None:
    """Validate the decorrelation."""
    instance_of((bool, float))(obj, attribute, value)
    if isinstance(value, float):
        gt(0.0)(obj, attribute, value)
        lt(1.0)(obj, attribute, value)


def _validate_unique_values(  # noqa: DOC101, DOC103
    obj: Any, _: Any, value: list
) -> None:
    """Validate that there are no duplicates in ```value```.

    Raises:
        ValueError: If there are duplicates in ```value```.
    """
    if len(set(value)) != len(value):
        raise ValueError(
            f"Cannot assign the following values containing duplicates to "
            f"parameter {obj.name}: {value}."
        )


@define(frozen=True, slots=False)
class Parameter(ABC, SerialMixin):
    """Abstract base class for all parameters.

    Stores information about the type, range, constraints, etc. and handles in-range
    checks, transformations etc.

    Args:
        name: The name of the parameter
    """

    # class variables
    is_numeric: ClassVar[bool]
    """Class variable encoding whether this parameter is numeric."""
    is_discrete: ClassVar[bool]
    """Class variable encoding whether this parameter is discrete."""

    # object variables
    name: str = field()

    @abstractmethod
    def is_in_range(self, item: Any) -> bool:
        """Return whether an item is within the parameter range.

        Args:
            item: The item to be checked.

        Returns:
            ```True``` if the item is within the parameter range, ```False``` otherwise.
        """


@define(frozen=True, slots=False)
class DiscreteParameter(Parameter, ABC):
    """Abstract class for discrete parameters.

    Args:
        encoding: The encoding of the parameter.
    """

    # TODO [15280]: needs to be refactored

    # class variables
    is_discrete: ClassVar[bool] = True

    # object variables
    encoding: ClassVar[Optional[str]] = None

    @property
    @abstractmethod
    def values(self) -> list:
        """The values the parameter can take."""

    @cached_property
    @abstractmethod
    def comp_df(self) -> pd.DataFrame:
        """Return the computational representation of the parameter."""

    def is_in_range(self, item: Any) -> bool:  # noqa: D102
        # See base class.
        return item in self.values

    def transform_rep_exp2comp(self, data: pd.Series) -> pd.DataFrame:
        """Transform data from experimental to computational representation.

        Args:
            data: Data to be transformed.

        Returns:
            The transformed version of the data.
        """
        if self.encoding:
            # replace each label with the corresponding encoding
            transformed = pd.merge(
                left=data.rename("Labels").to_frame(),
                left_on="Labels",
                right=self.comp_df,
                right_index=True,
                how="left",
            ).drop(columns="Labels")
        else:
            transformed = data.to_frame()

        return transformed


@define(frozen=True, slots=False)
class CategoricalParameter(DiscreteParameter):
    """Parameter class for categorical parameters.

    Args:
        encoding: The encoding of the parameter.
    """

    # TODO: Since object variables are not inherited, we need to include it here again.
    # Due to this, changes in such variables are not auto-documented by Sphinx, making
    # it necessary to change them manually.
    # This might change when moving towards html based documentation.

    # class variables
    is_numeric: ClassVar[bool] = False

    # object variables
    _values: list = field(
        converter=list, validator=[min_len(2), _validate_unique_values]
    )
    encoding: Literal["OHE", "INT"] = field(default="OHE")

    @property
    def values(self) -> list:
        """The values of the parameter."""
        return self._values

    @cached_property
    def comp_df(self) -> pd.DataFrame:  # noqa: D102
        # See base class.
        if self.encoding == "OHE":
            cols = [f"{self.name}_{val}" for val in self.values]
            comp_df = pd.DataFrame(np.eye(len(self.values), dtype=int), columns=cols)
        elif self.encoding == "INT":
            comp_df = pd.DataFrame(range(len(self.values)), columns=[self.name])
        comp_df.index = pd.Index(self.values)

        return comp_df


@define(frozen=True, slots=False)
class NumericalDiscreteParameter(DiscreteParameter):
    """Parameter class for discrete numerical parameters (a.k.a. setpoints).

    Args:
        tolerance: The absolute tolerance used for deciding whether a value is in range.
            A tolerance larger than half the minimum distance between parameter values
            is not allowed because that could cause ambiguity when inputting data points
            later.
    """

    # class variables
    is_numeric: ClassVar[bool] = True

    # object variables
    _values: List[Union[int, float]] = field(
        converter=list,
        validator=[
            deep_iterable(instance_of((int, float)), instance_of(list)),
            min_len(2),
            _validate_unique_values,
        ],
    )
    tolerance: float = field(default=0.0)

    @tolerance.validator
    def _validate_tolerance(  # noqa: DOC101, DOC103
        self, _: Any, tolerance: float
    ) -> None:
        """Validate that the given tolerance is safe.

        The tolerance is the allowed experimental uncertainty when
        reading in measured values. A tolerance larger than half the minimum
        distance between parameter values is not allowed because that could cause
        ambiguity when inputting data points later.

        Raises:
            ValueError: If the tolerance is not safe.
        """
        # NOTE: computing all pairwise distances can be avoided if we ensure that the
        #   values are ordered (which is currently not the case)
        dists = pdist(np.asarray(self.values).reshape(-1, 1))
        max_tol = dists.min() / 2.0

        if tolerance >= max_tol:
            raise ValueError(
                f"Parameter {self.name} is initialized with tolerance "
                f"{tolerance} but due to the values {self.values} a "
                f"maximum tolerance of {max_tol} is suggested to avoid ambiguity."
            )

    @property
    def values(self) -> list:  # noqa: D102
        # See base class.
        return self._values

    @cached_property
    def comp_df(self) -> pd.DataFrame:  # noqa: D102
        # See base class.
        comp_df = pd.DataFrame({self.name: self.values}, index=self.values)
        return comp_df

    def is_in_range(self, item: float) -> bool:  # noqa: D102
        # See base class.
        differences_acceptable = [
            np.abs(val - item) <= self.tolerance for val in self.values
        ]
        return any(differences_acceptable)


@define(frozen=True, slots=False)
class NumericalContinuousParameter(Parameter):
    """Parameter class for continuous numerical parameters.

    Args:
        bounds: The bounds of the parameter.
    """

    # class variables
    is_numeric: ClassVar[bool] = True
    is_discrete: ClassVar[bool] = False

    # object variables
    bounds: Interval = field(default=None, converter=convert_bounds)

    @bounds.validator
    def _validate_bounds(self, _: Any, value: Interval) -> None:  # noqa: DOC101, DOC103
        """Validate bounds.

        Raises:
            InfiniteIntervalError: If the provided interval is infinite.
        """
        if not value.is_finite:
            raise InfiniteIntervalError(
                f"You are trying to initialize a parameter with an infinite interval "
                f"of {value.to_tuple()}. Infinite intervals for parameters are "
                f"currently not supported."
            )

    def is_in_range(self, item: float) -> bool:  # noqa: D102
        # See base class.

        return self.bounds.contains(item)


@define(frozen=True, slots=False)
class SubstanceParameter(DiscreteParameter):
    """Generic substances that are treated with cheminformatics descriptors.

    Only a decorrelated subset of descriptors should be used as otherwise this can
    result in a large number of features. For a handful of molecules, keeping only
    descriptors that have a maximum correlation of 0.7 reduces the number of
    descriptors to about 5-20. The number might be substantially higher with more
    labels given.

    Args:
        encoding: The encoding of the variable.
    """

    # TODO: Since object variables are not inherited, we need to include it here again.
    # This might change when moving towards html based documentation.
    # class variables
    is_numeric: ClassVar[bool] = False

    # object variables
    data: Dict[str, str] = field()
    decorrelate: Union[bool, float] = field(
        default=True, validator=_validate_decorrelation
    )
    encoding: Literal["MORDRED", "RDKIT", "MORGAN_FP"] = field(default="MORDRED")

    @encoding.validator
    def _validate_encoding(self, _: Any, value: str) -> None:  # noqa: DOC101, DOC103
        """Validate that the chosen encoding can be used.

        This validation is necessary since certain encodings are only useable when
        additional dependencies, in particular the ```chem``` dependency, have been
        installed.

        Raises:
            ImportError: If the ```chem```dependency was not installed but an encoding
                requiring this dependency is requested.
        """
        if value in ["MORDRED"] and not (_MORDRED_INSTALLED and _RDKIT_INSTALLED):
            raise ImportError(
                "The mordred/rdkit packages are not installed, a SubstanceParameter "
                "with MORDRED encoding cannot be used. Consider installing baybe with "
                "'chem' dependency like 'pip install baybe[chem]'"
            )
        if value in ["RDKIT", "MORGAN_FP"] and not _RDKIT_INSTALLED:
            raise ImportError(
                "The rdkit package is not installed, a SubstanceParameter with "
                "RDKIT or MORGAN_FP encoding cannot be used. Consider installing baybe "
                "with 'chem' dependency like 'pip install baybe[chem]'"
            )

    @data.validator
    def _validate_substance_data(  # noqa: DOC101, DOC103
        self, _: Any, value: Dict[str, str]
    ) -> None:
        """Validate that the substance data, provided as SMILES, is valid.

        Raises:
            ValueError: If one or more of the SMILES are invalid.
        """
        for name, smiles in value.items():
            if _RDKIT_INSTALLED and not is_valid_smiles(smiles):
                raise ValueError(
                    f"The SMILES '{smiles}' for molecule '{name}' does "
                    f"not appear to be valid."
                )

    @property
    def values(self) -> list:
        """Returns the labels of the given set of molecules."""
        # Since the order of dictionary keys is important here, this will only work
        # for Python 3.7 or higher
        return list(self.data.keys())

    @cached_property
    def comp_df(self) -> pd.DataFrame:  # noqa: D102
        # See base class.
        vals = list(self.data.values())
        pref = self.name + "_"

        # Get the raw fingerprints
        if self.encoding == "MORDRED":
            comp_df = smiles_to_mordred_features(vals, prefix=pref)
        elif self.encoding == "RDKIT":
            comp_df = smiles_to_rdkit_features(vals, prefix=pref)
        elif self.encoding == "MORGAN_FP":
            comp_df = smiles_to_fp_features(vals, prefix=pref)
        else:
            raise ValueError(
                f"Unknown parameter encoding {self.encoding} for parameter {self.name}."
            )

        # Drop NaN and constant columns
        # Due to the above if clauses pylint thinks comp_df could become None
        comp_df = comp_df.loc[
            :, ~comp_df.isna().any(axis=0)  # pylint: disable=invalid-unary-operand-type
        ]
        comp_df = df_drop_single_value_columns(comp_df)

        # If there are bool columns, convert them to int (possible for Mordred)
        comp_df.loc[:, comp_df.dtypes == bool] = comp_df.loc[
            :, comp_df.dtypes == bool
        ].astype(int)

        # Label the rows with the molecule names
        comp_df.index = pd.Index(self.values)

        # Get a decorrelated subset of the fingerprints
        if self.decorrelate:
            if isinstance(self.decorrelate, bool):
                comp_df = df_uncorrelated_features(comp_df)
            else:
                comp_df = df_uncorrelated_features(comp_df, threshold=self.decorrelate)

        return comp_df


# Available encodings for substance parameters
SUBSTANCE_ENCODINGS = get_args(get_type_hints(SubstanceParameter)["encoding"])


@define(frozen=True, slots=False)
class TaskParameter(CategoricalParameter):
    """Parameter class for task parameters.

    Args:
        encoding: The encoding of the parameter.
        active_values: An optional list of values describing for which tasks
            recommendations should be given. By default, all parameters are considered
            active.
    """

    # object variables
    # IMPROVE: The encoding effectively becomes a class variable here, but cannot be
    #   declared as such because of the inheritance relationship.
    encoding: Literal["INT"] = field(default="INT")
    active_values: list = field(converter=list)

    @active_values.default
    def _default_active_values(self) -> list:
        """Set all parameters active by default."""
        # TODO [16605]: Redesign metadata handling
        return self.values

    @active_values.validator
    def _validate_active_values(  # noqa: DOC101, DOC103
        self, _: Any, values: list
    ) -> None:
        """Validate the active parameter values.

        If no such list is provided, no validation is being performed. In particular,
        the errors listed below are only relevant if the ```values``` list is provided.

        Raises:
            ValueError: If an empty active parameters list is provided.
            ValueError: If the active parameter values are not unique.
            ValueError: If not all active values are valid parameter choices.
        """
        # TODO [16605]: Redesign metadata handling
        if len(values) == 0:
            raise ValueError(
                "If an active parameters list is provided, it must not be empty."
            )
        if len(set(values)) != len(values):
            raise ValueError("The active parameter values must be unique.")
        if not all(v in self.values for v in values):
            raise ValueError("All active values must be valid parameter choices.")


@define(frozen=True, slots=False)
class CustomDiscreteParameter(DiscreteParameter):
    """Custom parameters.

    For these parameters, the user can read in a precomputed representation for labels,
    e.g. from quantum chemistry.

    Args:
        data: The data for the custom parameter.
        decorrelate: Flag encoding whether the provided data should be decorrelated.
        encoding: The encoding of the parameter.
    """

    # class variables
    is_numeric: ClassVar[bool] = False

    # object variables
    data: pd.DataFrame = field(eq=eq_dataframe)
    decorrelate: Union[bool, float] = field(
        default=True, validator=_validate_decorrelation
    )
    encoding = field(default="CUSTOM")

    @data.validator
    def _validate_custom_data(  # noqa: DOC101, DOC103
        self, _: Any, value: pd.DataFrame
    ) -> None:
        """Validate the dataframe with the custom representation.

        Raises:
            ValueError: If the dataframe contains ```NaN```.
            ValueError: If the dataframe contains duplicated indices.
            ValueError: If the dataframe contains non-numeric values.
            ValueError: If the dataframe contains columns that only contain a single
                value.
        """
        if value.isna().any().any():
            raise ValueError(
                f"The custom dataframe for parameter {self.name} contains NaN "
                f"entries, which is not supported."
            )
        if len(value) != len(set(value.index)):
            raise ValueError(
                f"The custom dataframe for parameter {self.name} contains "
                f"duplicated indices. Please only provide dataframes with unique"
                f" indices."
            )
        if value.select_dtypes("number").shape[1] != value.shape[1]:
            raise ValueError(
                f"The custom dataframe for parameter {self.name} contains "
                f"non-numeric values."
            )
        if any(value.nunique() == 1):
            raise ValueError(
                f"The custom dataframe for parameter {self.name} has columns "
                "that contain only a single value and hence carry no information."
            )

    @property
    def values(self) -> list:
        """Returns the representing labels of the parameter."""
        return self.data.index.to_list()

    @cached_property
    def comp_df(self) -> pd.DataFrame:  # noqa: D102
        # See base class.
        # The encoding is directly provided by the user
        comp_df = self.data

        # Get a decorrelated subset of the provided features
        if self.decorrelate:
            if isinstance(self.decorrelate, bool):
                comp_df = df_uncorrelated_features(comp_df)
            else:
                comp_df = df_uncorrelated_features(comp_df, threshold=self.decorrelate)

        return comp_df


def parameter_cartesian_prod_to_df(
    parameters: Iterable[Parameter],
) -> pd.DataFrame:
    """Create the Cartesian product of all parameter values.

    Ignores continuous parameters.

    Args:
        parameters: List of parameter objects.

    Returns:
        A dataframe containing all possible discrete parameter value combinations.
    """
    lst_of_values = [
        cast(DiscreteParameter, p).values for p in parameters if p.is_discrete
    ]
    lst_of_names = [p.name for p in parameters if p.is_discrete]
    if len(lst_of_names) < 1:
        return pd.DataFrame()

    index = pd.MultiIndex.from_product(lst_of_values, names=lst_of_names)
    ret = pd.DataFrame(index=index).reset_index()

    return ret


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Temporary workaround >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# TODO: This is a temporary workaround to remove the prefixed underscore of "_values"
#   in the serialization dicts to make them user-friendly (e.g. for the BayBE config).
#   There is a better, built-in solution in attrs using the `override` approach
#   described here: https://github.com/python-attrs/cattrs/issues/23
#   However, two things need to happen before we can easily apply it:
#   * The subclassing support PR needs to merged, which avoids any further need
#     for our custom "base" hooks used below.
#     https://github.com/python-attrs/cattrs/pull/312
#   * The converter structure has been optimized so that the hooks no longer need
#     to be registered with the global converter:
#     https://***REMOVED***/_workitems/edit/12356/


def _remove_values_underscore(raw_unstructure_hook):
    """Replace the ```_values``` field with ```values```."""

    def wrapper(obj):
        dict_ = raw_unstructure_hook(obj)
        try:
            dict_["values"] = dict_.pop("_values")
        except KeyError:
            pass
        return dict_

    return wrapper


def _add_values_underscore(raw_structure_hook):
    """Replace the ```values``` field with ```_values```."""

    def wrapper(dict_, _):
        try:
            dict_["_values"] = dict_.pop("values")
        except KeyError:
            pass
        return raw_structure_hook(dict_, _)

    return wrapper


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Temporary workaround <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Register (un-)structure hooks
unstructure_hook = _remove_values_underscore(unstructure_base)
structure_hook = _add_values_underscore(get_base_unstructure_hook(Parameter))
cattrs.register_unstructure_hook(Parameter, unstructure_hook)
cattrs.register_structure_hook(Parameter, structure_hook)


def _validate_parameter_names(  # noqa: DOC101, DOC103
    parameters: List[Parameter],
) -> None:
    """Validate the parameter names.

    Raises:
        ValueError: If the given list contains parameters with the same name.
    """
    param_names = [p.name for p in parameters]
    if len(set(param_names)) != len(param_names):
        raise ValueError("All parameters must have unique names.")


def _validate_parameters(parameters: List[Parameter]) -> None:  # noqa: DOC101, DOC103
    """Validate the parameters.

    Raises:
        EmptySearchSpaceError: If the parameter list is empty.
        NotImplementedError: If more than one ```TaskParameter``` is requested.
    """
    if not parameters:
        raise EmptySearchSpaceError("At least one parameter must be provided.")

    # TODO [16932]: Remove once more task parameters are supported
    if len([p for p in parameters if isinstance(p, TaskParameter)]) > 1:
        raise NotImplementedError(
            "Currently, at most one task parameter can be considered."
        )

    # Assert: unique names
    _validate_parameter_names(parameters)
