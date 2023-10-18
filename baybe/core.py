"""Core functionality of BayBE. Main point of interaction via Python."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import List

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field

from baybe.constraints import _validate_constraints, Constraint
from baybe.parameters import _validate_parameters, Parameter
from baybe.searchspace import SearchSpace
from baybe.strategies.strategy import Strategy
from baybe.targets import NumericalTarget, Objective
from baybe.telemetry import (
    TELEM_LABELS,
    telemetry_record_recommended_measurement_percentage,
    telemetry_record_value,
)
from baybe.utils import eq_dataframe, SerialMixin


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Temporary workaround >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# TODO[12356]: There should be a way to organize several converters, instead of
#   registering the hooks with the global converter. The global converter is currently
#   used so that all "basic" hooks (like DataFrame serializers) can be used at all
#   hierarchy levels. Ideally, however, each module would define its own converters for
#   its classes that can then be flexibly combined at the next higher level of the
#   module hierarchy. For example, there could be several "Parameter converters" that
#   implement different sorts of serialization logic. For the search space, one could
#   then implement several serialization converters as well that arbitrarily combine
#   parameter and constraint hooks/converters.


def _structure_dataframe_hook(string: str, _) -> pd.DataFrame:
    """Hook for de-serializing a DataFrame."""
    buffer = BytesIO()
    buffer.write(base64.b64decode(string.encode("utf-8")))
    return pd.read_parquet(buffer)


def _unstructure_dataframe_hook(df: pd.DataFrame) -> str:
    """Hook for serializing a DataFrame."""
    return base64.b64encode(df.to_parquet()).decode("utf-8")


cattrs.register_unstructure_hook(pd.DataFrame, _unstructure_dataframe_hook)
cattrs.register_structure_hook(pd.DataFrame, _structure_dataframe_hook)


def _searchspace_creation_hook(specs: dict, _) -> SearchSpace:
    """A structuring hook that assembles the search space."""
    # A structuring hook that assembles the search space using the alternative `create`
    # constructor, which allows to deserialize search space specifications that are
    # provided in a user-friendly format (i.e. via parameters and constraints).
    # IMPROVE: Instead of defining the partial structurings here in the hook,
    #   on *could* use a dedicated "BayBEConfig" class
    parameters = cattrs.structure(specs["parameters"], List[Parameter])
    constraints = specs.get("constraints", None)
    if constraints:
        constraints = cattrs.structure(specs["constraints"], List[Constraint])
    return SearchSpace.from_product(parameters, constraints)


def _searchspace_validation_hook(specs: dict, _) -> None:
    """A validation hook that does not create the search space."""
    # Similar to `searchspace_creation_hook` but without the actual search space
    # creation step, thus intended for validation purposes only. Additionally,
    # explicitly asserts uniqueness of parameter names, since duplicates would only be
    # noticed during search space creation.
    parameters = cattrs.structure(specs["parameters"], List[Parameter])
    _validate_parameters(parameters)

    constraints = specs.get("constraints", None)
    if constraints:
        constraints = cattrs.structure(specs["constraints"], List[Constraint])
        _validate_constraints(constraints, parameters)


_config_converter = cattrs.global_converter.copy()
_config_converter.register_structure_hook(SearchSpace, _searchspace_creation_hook)

_validation_converter = cattrs.global_converter.copy()
_validation_converter.register_structure_hook(SearchSpace, _searchspace_validation_hook)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Temporary workaround <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


@define
class BayBE(SerialMixin):
    """Main class for interaction with BayBE.

    Args:
        searchspace: The search space in which the experiments are conducted.
        objective: The optimization objective.
        strategy: The employed strategy.
        measurements_exp: The experimental representation of the conducted experiments.
        numerical_measurements_must_be_within_tolerance: Flag for forcing numerical
            measurements to be within tolerance.
        batches_done: The number of already processed batches.
        fits_done: The number of fits already done.
    """

    # DOE specifications
    searchspace: SearchSpace = field()
    objective: Objective = field()
    strategy: Strategy = field(factory=Strategy)

    # Data
    measurements_exp: pd.DataFrame = field(factory=pd.DataFrame, eq=eq_dataframe)
    numerical_measurements_must_be_within_tolerance: bool = field(default=True)

    # Metadata
    batches_done: int = field(default=0)
    fits_done: int = field(default=0)

    # Private
    _cached_recommendation: pd.DataFrame = field(factory=pd.DataFrame, eq=eq_dataframe)

    @property
    def parameters(self) -> List[Parameter]:
        """The parameters of the underlying search space."""
        return self.searchspace.parameters

    @property
    def targets(self) -> List[NumericalTarget]:
        """The targets of the underlying objective."""
        # TODO: Currently, the `Objective` class is directly coupled to
        #  `NumericalTarget`, hence the return type.
        return self.objective.targets

    @property
    def measurements_parameters_comp(self) -> pd.DataFrame:
        """The computational representation of the measured parameters."""
        if len(self.measurements_exp) < 1:
            return pd.DataFrame()
        return self.searchspace.transform(self.measurements_exp)

    @property
    def measurements_targets_comp(self) -> pd.DataFrame:
        """The computational representation of the measured targets."""
        if len(self.measurements_exp) < 1:
            return pd.DataFrame()
        return self.objective.transform(self.measurements_exp)

    @classmethod
    def from_config(cls, config_json: str) -> BayBE:
        """Create a BayBE object from a configuration JSON.

        Args:
            config_json: The string with the configuration JSON.

        Returns:
            The constructed BayBE object.
        """
        config = json.loads(config_json)
        config["searchspace"] = {
            "parameters": config.pop("parameters"),
            "constraints": config.pop("constraints", None),
        }
        return _config_converter.structure(config, BayBE)

    @classmethod
    def to_config(cls) -> str:
        """Extract the configuration of the BayBE object as JSON string.

        Note: This is not yet implemented. Use
        :func:`baybe.utils.serialization.SerialMixin.to_json` instead

        Returns:
            The configuration as JSON string.

        Raises:
            NotImplementedError: When trying to use this function.
        """
        # TODO: Ideally, this should extract a "minimal" configuration, that is,
        #   default values should not be exported, which cattrs supports via the
        #   'omit_if_default' option. Can be Implemented once the converter structure
        #   has been cleaned up.
        raise NotImplementedError()

    @classmethod
    def validate_config(cls, config_json: str) -> None:
        """Validates a given BayBE configuration JSON.

        Args:
            config_json: The JSON that should be validated.
        """
        config = json.loads(config_json)
        config["searchspace"] = {
            "parameters": config.pop("parameters"),
            "constraints": config.pop("constraints", None),
        }
        _validation_converter.structure(config, BayBE)

    def add_measurements(self, data: pd.DataFrame) -> None:
        """Add results from a dataframe to the internal database.

        Each addition of data is considered a new batch. Added results are checked for
        validity. Categorical values need to have an exact match. For numerical values,
        a BayBE flag determines if values that lie outside a specified tolerance
        are accepted.
        Note that this modifies the provided data in-place.

        Args:
            data: The data to be added (with filled values for targets). Preferably
                created via :func:`baybe.core.BayBE.recommend`.

        Raises:
            ValueError: If one of the targets has missing values or NaNs in the provided
                dataframe.
            TypeError: If the target has non-numeric entries in the provided dataframe.
        """
        # Invalidate recommendation cache first (in case of uncaught exceptions below)
        self._cached_recommendation = pd.DataFrame()

        # Check if all targets have valid values
        for target in self.targets:
            if data[target.name].isna().any():
                raise ValueError(
                    f"The target '{target.name}' has missing values or NaNs in the "
                    f"provided dataframe. Missing target values are not supported."
                )
            if data[target.name].dtype.kind not in "iufb":
                raise TypeError(
                    f"The target '{target.name}' has non-numeric entries in the "
                    f"provided dataframe. Non-numeric target values are not supported."
                )

        # Check if all targets have valid values
        for param in self.parameters:
            if data[param.name].isna().any():
                raise ValueError(
                    f"The parameter '{param.name}' has missing values or NaNs in the "
                    f"provided dataframe. Missing parameter values are not supported."
                )
            if param.is_numeric and (data[param.name].dtype.kind not in "iufb"):
                raise TypeError(
                    f"The numerical parameter '{param.name}' has non-numeric entries in"
                    f" the provided dataframe."
                )

        # Update meta data
        # TODO: refactor responsibilities
        self.searchspace.discrete.mark_as_measured(
            data, self.numerical_measurements_must_be_within_tolerance
        )

        # Read in measurements and add them to the database
        # TODO: See if np.nan can be replaced with pd.NA once (de-)serialization is
        #   in place. The current serializer set in the pydantic config does not support
        #   pd.NA. Pandas' .to_json() can handle it but reading it back gives also
        #   np.nan as result. Probably, the only way is a custom (de-)serializer.
        self.batches_done += 1
        to_insert = data.copy()
        to_insert["BatchNr"] = self.batches_done
        to_insert["FitNr"] = np.nan

        self.measurements_exp = pd.concat(
            [self.measurements_exp, to_insert], axis=0, ignore_index=True
        )

        # Telemetry
        telemetry_record_value(TELEM_LABELS["COUNT_ADD_RESULTS"], 1)
        telemetry_record_recommended_measurement_percentage(
            self._cached_recommendation,
            data,
            self.parameters,
            self.numerical_measurements_must_be_within_tolerance,
        )

    def recommend(self, batch_quantity: int = 5) -> pd.DataFrame:
        """Provide the recommendations for the next batch of experiments.

        Args:
            batch_quantity: Number of requested recommendations.

        Returns:
            Dataframe containing the recommendations in experimental representation.

        Raises:
            ValueError: If ```batch_quantity``` is smaller than 1.
        """
        if batch_quantity < 1:
            raise ValueError(
                f"You must at least request one recommendation per batch, but provided "
                f"{batch_quantity=}."
            )

        # If there are cached recommendations and the batch size of those is equal to
        # the previously requested one, we just return those
        if len(self._cached_recommendation) == batch_quantity:
            return self._cached_recommendation

        # Update recommendation meta data
        if len(self.measurements_exp) > 0:
            self.fits_done += 1
            self.measurements_exp["FitNr"].fillna(self.fits_done, inplace=True)

        # Get the recommended search space entries
        rec = self.strategy.recommend(
            self.searchspace,
            batch_quantity,
            self.measurements_parameters_comp,
            self.measurements_targets_comp,
        )

        # Cache the recommendations
        self._cached_recommendation = rec.copy()

        # Telemetry
        telemetry_record_value(TELEM_LABELS["COUNT_RECOMMEND"], 1)
        telemetry_record_value(TELEM_LABELS["BATCH_QUANTITY"], batch_quantity)

        return rec
