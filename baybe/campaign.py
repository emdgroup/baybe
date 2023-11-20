"""Functionality for managing DOE campaigns. Main point of interaction via Python."""

from __future__ import annotations

import json
from typing import Any, List

import numpy as np
import pandas as pd
from attrs import define, field

from baybe.objective import Objective
from baybe.parameters.base import Parameter
from baybe.searchspace.core import (
    SearchSpace,
    structure_searchspace_from_config,
    validate_searchspace_from_config,
)
from baybe.strategies import TwoPhaseStrategy
from baybe.strategies.base import Strategy
from baybe.targets import NumericalTarget
from baybe.telemetry import (
    TELEM_LABELS,
    telemetry_record_recommended_measurement_percentage,
    telemetry_record_value,
)
from baybe.utils import eq_dataframe
from baybe.utils.serialization import SerialMixin, converter

# Converter for config deserialization
_config_converter = converter.copy()
_config_converter.register_structure_hook(
    SearchSpace, structure_searchspace_from_config
)

# Converter for config validation
_validation_converter = converter.copy()
_validation_converter.register_structure_hook(
    SearchSpace, validate_searchspace_from_config
)


@define
class Campaign(SerialMixin):
    """Main class for interaction with BayBE.

    Campaigns define and record an experimentation process, i.e. the execution of a
    series of measurements and the iterative sequence of events involved.

    In particular, a campaign:
        * Defines the objective of an experimentation process.
        * Defines the search space over which the experimental parameter may vary.
        * Defines a strategy for traversing the search space.
        * Records the measurement data collected during the process.
        * Records metadata about the progress of the experimentation process.
    """

    # DOE specifications
    searchspace: SearchSpace = field()
    """The search space in which the experiments are conducted."""

    objective: Objective = field()
    """The optimization objective."""

    strategy: Strategy = field(factory=TwoPhaseStrategy)
    """The employed strategy"""

    # Data
    measurements_exp: pd.DataFrame = field(factory=pd.DataFrame, eq=eq_dataframe)
    """The experimental representation of the conducted experiments."""

    numerical_measurements_must_be_within_tolerance: bool = field(default=True)
    """Flag for forcing numerical measurements to be within tolerance."""

    # Metadata
    n_batches_done: int = field(default=0)
    """The number of already processed batches."""

    n_fits_done: int = field(default=0)
    """The number of fits already done."""

    # Private
    _cached_recommendation: pd.DataFrame = field(factory=pd.DataFrame, eq=eq_dataframe)
    """The cached recommendations."""

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
    def from_config(cls, config_json: str) -> Campaign:
        """Create a campaign from a configuration JSON.

        Args:
            config_json: The string with the configuration JSON.

        Returns:
            The constructed campaign.
        """
        config = json.loads(config_json)
        config["searchspace"] = {
            "parameters": config.pop("parameters"),
            "constraints": config.pop("constraints", None),
        }
        return _config_converter.structure(config, Campaign)

    @classmethod
    def to_config(cls) -> str:
        """Extract the configuration of the campaign as JSON string.

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
        """Validate a given campaign configuration JSON.

        Args:
            config_json: The JSON that should be validated.
        """
        config = json.loads(config_json)
        config["searchspace"] = {
            "parameters": config.pop("parameters"),
            "constraints": config.pop("constraints", None),
        }
        _validation_converter.structure(config, Campaign)

    def add_measurements(self, data: pd.DataFrame) -> None:
        """Add results from a dataframe to the internal database.

        Each addition of data is considered a new batch. Added results are checked for
        validity. Categorical values need to have an exact match. For numerical values,
        a campaign flag determines if values that lie outside a specified tolerance
        are accepted.
        Note that this modifies the provided data in-place.

        Args:
            data: The data to be added (with filled values for targets). Preferably
                created via :func:`baybe.campaign.Campaign.recommend`.

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
        self.n_batches_done += 1
        to_insert = data.copy()
        to_insert["BatchNr"] = self.n_batches_done
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
            ValueError: If ``batch_quantity`` is smaller than 1.
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
            self.n_fits_done += 1
            self.measurements_exp["FitNr"].fillna(self.n_fits_done, inplace=True)

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


def _unstructure_with_version(obj: Any) -> dict:
    """Add the package version to the created dictionary."""
    from baybe import __version__

    return {
        **converter.unstructure_attrs_asdict(obj),
        "version": __version__,
    }


converter.register_unstructure_hook(Campaign, _unstructure_with_version)
