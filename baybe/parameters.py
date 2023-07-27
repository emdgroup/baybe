# pylint: disable=missing-function-docstring
"""
Functionality for different experimental parameter types.
"""
# TODO: ForwardRefs via __future__ annotations are currently disabled due to this issue:
#  https://github.com/python-attrs/cattrs/issues/354

import logging
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

from baybe.utils import (
    df_drop_single_value_columns,
    df_uncorrelated_features,
    EmptySearchSpaceError,
    eq_dataframe,
    get_base_unstructure_hook,
    is_valid_smiles,
    smiles_to_fp_features,
    smiles_to_mordred_features,
    smiles_to_rdkit_features,
    unstructure_base,
)
from baybe.utils.interval import convert_bounds, InfiniteIntervalError, Interval
from baybe.utils.serialization import SerialMixin

log = logging.getLogger(__name__)

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


def validate_decorrelation(obj, attribute, value):
    instance_of((bool, float))(obj, attribute, value)
    if isinstance(value, float):
        gt(0.0)(obj, attribute, value)
        lt(1.0)(obj, attribute, value)


def validate_unique_values(obj, _, value) -> None:
    if len(set(value)) != len(value):
        raise ValueError(
            f"Cannot assign the following values containing duplicates to "
            f"parameter {obj.name}: {value}."
        )


@define(frozen=True, slots=False)
class Parameter(ABC, SerialMixin):
    """
    Abstract base class for all parameters. Stores information about the
    type, range, constraints, etc. and handles in-range checks, transformations etc.
    """

    # class variables
    is_numeric: ClassVar[bool]
    is_discrete: ClassVar[bool]

    # object variables
    name: str = field()

    @abstractmethod
    def is_in_range(self, item: Any) -> bool:
        """
        Tells whether an item is within the parameter range.
        """


@define(frozen=True, slots=False)
class DiscreteParameter(Parameter, ABC):
    """
    Abstract class for discrete parameters.
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
        """
        Returns the computational representation of the parameter.
        """

    def is_in_range(self, item: Any) -> bool:
        return item in self.values

    def transform_rep_exp2comp(self, data: pd.Series) -> pd.DataFrame:
        """
        Transforms data from experimental to computational representation.

        Parameters
        ----------
        data: pd.Series
            Data to be transformed.

        Returns
        -------
        pd.DataFrame
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
class Categorical(DiscreteParameter):
    """
    Parameter class for categorical parameters.
    """

    # class variables
    is_numeric = False

    # object variables
    _values: list = field(
        converter=list, validator=[min_len(2), validate_unique_values]
    )
    encoding: Literal["OHE", "INT"] = field(default="OHE")

    @property
    def values(self) -> list:
        return self._values

    @cached_property
    def comp_df(self) -> pd.DataFrame:
        """
        See base class.
        """
        if self.encoding == "OHE":
            cols = [f"{self.name}_{val}" for val in self.values]
            comp_df = pd.DataFrame(np.eye(len(self.values), dtype=int), columns=cols)
        elif self.encoding == "INT":
            comp_df = pd.DataFrame(range(len(self.values)), columns=[self.name])
        comp_df.index = pd.Index(self.values)

        return comp_df


@define(frozen=True, slots=False)
class NumericDiscrete(DiscreteParameter):
    """
    Parameter class for discrete numerical parameters (a.k.a. setpoints).
    """

    # class variables
    is_numeric = True

    # object variables
    _values: List[Union[int, float]] = field(
        converter=list,
        validator=[
            deep_iterable(instance_of((int, float)), instance_of(list)),
            min_len(2),
            validate_unique_values,
        ],
    )
    tolerance: float = field(default=0.0)

    @tolerance.validator
    def validate_tolerance(self, _, tolerance):
        """
        Validates that the tolerance (i.e. allowed experimental uncertainty when
        reading in measured values) is safe. A tolerance larger than half the minimum
        distance between parameter values is not allowed because that could cause
        ambiguity when inputting data points later.
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
    def values(self) -> list:
        return self._values

    @cached_property
    def comp_df(self) -> pd.DataFrame:
        """
        See base class.
        """
        comp_df = pd.DataFrame({self.name: self.values}, index=self.values)
        return comp_df

    def is_in_range(self, item: float) -> bool:
        """
        See base class.
        """
        differences_acceptable = [
            np.abs(val - item) <= self.tolerance for val in self.values
        ]
        return any(differences_acceptable)


@define(frozen=True, slots=False)
class NumericContinuous(Parameter):
    """
    Parameter class for continuous numerical parameters.
    """

    # class variables
    is_numeric = True
    is_discrete = False

    # object variables
    bounds: Interval = field(default=None, converter=convert_bounds)

    @bounds.validator
    def validate_bounds(self, _, value: Interval):
        if not value.is_finite:
            raise InfiniteIntervalError(
                f"You are trying to initialize a parameter with an infinite interval "
                f"of {value.to_tuple()}. Infinite intervals for parameters are "
                f"currently not supported."
            )

    def is_in_range(self, item: float) -> bool:
        """
        See base class.
        """

        return self.bounds.contains(item)


@define(frozen=True, slots=False)
class GenericSubstance(DiscreteParameter):
    """
    Parameter class for generic substances that are treated with cheminformatics
    descriptors.

    Only a decorrelated subset of descriptors should be used as otherwise this can
    result in a large number of features. For a handful of molecules, keeping only
    descriptors that have a maximum correlation of 0.7 reduces the number of
    descriptors to about 5-20. The number might be substantially higher with more
    labels given.
    """

    # class variables
    is_numeric = False

    # object variables
    data: Dict[str, str] = field()
    decorrelate: Union[bool, float] = field(
        default=True, validator=validate_decorrelation
    )
    encoding: Literal["MORDRED", "RDKIT", "MORGAN_FP"] = field(default="MORDRED")

    @data.validator
    def validate_substance_data(self, _, value):
        for name, smiles in value.items():
            if not is_valid_smiles(smiles):
                raise ValueError(
                    f"The SMILES '{smiles}' for molecule '{name}' does "
                    f"not appear to be valid."
                )

    @property
    def values(self) -> list:
        """
        Returns the labels of the given set of molecules.
        """
        # Since the order of dictionary keys is important here, this will only work
        # for Python 3.7 or higher
        return list(self.data.keys())

    @cached_property
    def comp_df(self) -> pd.DataFrame:
        """
        See base class.
        """
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
SUBSTANCE_ENCODINGS = get_args(get_type_hints(GenericSubstance)["encoding"])


@define(frozen=True, slots=False)
class Custom(DiscreteParameter):
    """
    Parameter class for custom parameters where the user can read in a precomputed
    representation for labels, e.g. from quantum chemistry.
    """

    # class variables
    is_numeric = False

    # object variables
    data: pd.DataFrame = field(eq=eq_dataframe())
    decorrelate: Union[bool, float] = field(
        default=True, validator=validate_decorrelation
    )
    encoding = field(default="CUSTOM")

    @data.validator
    def validate_custom_data(self, _, value):
        """
        Validates the dataframe with the custom representation.
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
        """
        Returns the representing labels of the parameter.
        """
        return self.data.index.to_list()

    @cached_property
    def comp_df(self) -> pd.DataFrame:
        """
        See base class.
        """
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
    """
    Creates the Cartesian product of all parameter values. Ignores continuous
    parameters.

    Parameters
    ----------
    parameters: List[Parameter]
        List of parameter objects.

    Returns
    -------
    pd.DataFrame
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


def remove_values_underscore(raw_unstructure_hook):
    def wrapper(obj):
        dict_ = raw_unstructure_hook(obj)
        try:
            dict_["values"] = dict_.pop("_values")
        except KeyError:
            pass
        return dict_

    return wrapper


def add_values_underscore(raw_structure_hook):
    def wrapper(dict_, _):
        try:
            dict_["_values"] = dict_.pop("values")
        except KeyError:
            pass
        return raw_structure_hook(dict_, _)

    return wrapper


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Temporary workaround <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Register (un-)structure hooks
unstructure_hook = remove_values_underscore(unstructure_base)
structure_hook = add_values_underscore(get_base_unstructure_hook(Parameter))
cattrs.register_unstructure_hook(Parameter, unstructure_hook)
cattrs.register_structure_hook(Parameter, structure_hook)


def _validate_parameter_names(parameters: List[Parameter]) -> None:
    """
    Asserts that a given collection of parameters has unique names.

    Parameters
    ----------
    parameters : list
        List of Parameter objects.

    Raises
    ------
    ValueError
        If the given list contains parameters with the same name.
    """
    param_names = [p.name for p in parameters]
    if len(set(param_names)) != len(param_names):
        raise ValueError("All parameters must have unique names.")


def _validate_parameters(parameters: List[Parameter]) -> None:
    """
    Asserts that a given collection of parameters is valid.

    Parameters
    ----------
    parameters : list
        List of Parameter objects.

    Raises
    ------
    ValueError
        If the given list of parameters is invalid.
    """
    # Assert that the parameter list is non-empty and contains unique names
    if not parameters:
        raise EmptySearchSpaceError("At least one parameter must be provided.")
    _validate_parameter_names(parameters)
