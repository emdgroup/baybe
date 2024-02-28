"""Functionality for managing DOE campaigns. Main point of interaction via Python."""

from __future__ import annotations

import json
from typing import List

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field

from baybe.exceptions import DeprecationError
from baybe.objective import Objective
from baybe.parameters.base import Parameter
from baybe.recommenders.base import RecommenderProtocol
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.searchspace.core import (
    SearchSpace,
    validate_searchspace_from_config,
)
from baybe.serialization import SerialMixin, converter
from baybe.targets.base import Target
from baybe.telemetry import (
    TELEM_LABELS,
    telemetry_record_recommended_measurement_percentage,
    telemetry_record_value,
)
from baybe.utils.boolean import eq_dataframe

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
        * Defines a recommender for exploring the search space.
        * Records the measurement data collected during the process.
        * Records metadata about the progress of the experimentation process.
    """

    # DOE specifications
    searchspace: SearchSpace = field()
    """The search space in which the experiments are conducted."""

    objective: Objective = field()
    """The optimization objective."""

    recommender: RecommenderProtocol = field(factory=TwoPhaseMetaRecommender)
    """The employed recommender"""

    # Metadata
    n_batches_done: int = field(default=0, init=False)
    """The number of already processed batches."""

    n_fits_done: int = field(default=0, init=False)
    """The number of fits already done."""

    # Private
    _measurements_exp: pd.DataFrame = field(
        factory=pd.DataFrame, eq=eq_dataframe, init=False
    )
    """The experimental representation of the conducted experiments."""

    _cached_recommendation: pd.DataFrame = field(
        factory=pd.DataFrame, eq=eq_dataframe, init=False
    )
    """The cached recommendations."""

    # Deprecation
    numerical_measurements_must_be_within_tolerance: bool = field(default=None)
    """Deprecated! Raises an error when used."""

    strategy: RecommenderProtocol = field(default=None)
    """Deprecated! Raises an error when used."""

    @numerical_measurements_must_be_within_tolerance.validator
    def _validate_tolerance_flag(self, _, value) -> None:
        """Raise a DeprecationError if the tolerance flag is used."""
        if value is not None:
            raise DeprecationError(
                f"Passing 'numerical_measurements_must_be_within_tolerance' to "
                f"the constructor is deprecated. The flag has become a parameter of "
                f"{self.__class__.__name__}.{Campaign.add_measurements.__name__}."
            )

    @strategy.validator
    def _validate_strategy(self, _, value) -> None:
        """Raise a DeprecationError if the strategy attribute is used."""
        if value is not None:
            raise DeprecationError(
                "Passing 'strategy' to the constructor is deprecated. The attribute "
                "has been renamed to 'recommender'."
            )

    @property
    def measurements(self) -> pd.DataFrame:
        """The experimental data added to the Campaign."""
        return self._measurements_exp

    @property
    def parameters(self) -> List[Parameter]:
        """The parameters of the underlying search space."""
        return self.searchspace.parameters

    @property
    def targets(self) -> List[Target]:
        """The targets of the underlying objective."""
        return self.objective.targets

    @property
    def _measurements_parameters_comp(self) -> pd.DataFrame:
        """The computational representation of the measured parameters."""
        if len(self._measurements_exp) < 1:
            return pd.DataFrame()
        return self.searchspace.transform(self._measurements_exp)

    @property
    def _measurements_targets_comp(self) -> pd.DataFrame:
        """The computational representation of the measured targets."""
        if len(self._measurements_exp) < 1:
            return pd.DataFrame()
        return self.objective.transform(self._measurements_exp)

    @classmethod
    def from_config(cls, config_json: str) -> Campaign:
        """Create a campaign from a configuration JSON.

        Args:
            config_json: The string with the configuration JSON.

        Returns:
            The constructed campaign.
        """
        from baybe.deprecation import compatibilize_config

        config = json.loads(config_json)

        # Temporarily enable backward compatibility
        config = compatibilize_config(config)

        return converter.structure(config, Campaign)

    @classmethod
    def validate_config(cls, config_json: str) -> None:
        """Validate a given campaign configuration JSON.

        Args:
            config_json: The JSON that should be validated.
        """
        from baybe.deprecation import compatibilize_config

        config = json.loads(config_json)

        # Temporarily enable backward compatibility
        config = compatibilize_config(config)

        _validation_converter.structure(config, Campaign)

    def add_measurements(
        self,
        data: pd.DataFrame,
        numerical_measurements_must_be_within_tolerance: bool = True,
    ) -> None:
        """Add results from a dataframe to the internal database.

        Each addition of data is considered a new batch. Added results are checked for
        validity. Categorical values need to have an exact match. For numerical values,
        a campaign flag determines if values that lie outside a specified tolerance
        are accepted.
        Note that this modifies the provided data in-place.

        Args:
            data: The data to be added (with filled values for targets). Preferably
                created via :func:`baybe.campaign.Campaign.recommend`.
            numerical_measurements_must_be_within_tolerance: Flag indicating if
                numerical parameters need to be within their tolerances.

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
            data, numerical_measurements_must_be_within_tolerance
        )

        # Read in measurements and add them to the database
        self.n_batches_done += 1
        to_insert = data.copy()
        to_insert["BatchNr"] = self.n_batches_done
        to_insert["FitNr"] = np.nan

        self._measurements_exp = pd.concat(
            [self._measurements_exp, to_insert], axis=0, ignore_index=True
        )

        # Telemetry
        telemetry_record_value(TELEM_LABELS["COUNT_ADD_RESULTS"], 1)
        telemetry_record_recommended_measurement_percentage(
            self._cached_recommendation,
            data,
            self.parameters,
            numerical_measurements_must_be_within_tolerance,
        )

    def recommend(
        self,
        batch_size: int = 5,
        batch_quantity: int = None,  # type: ignore[assignment]
    ) -> pd.DataFrame:
        """Provide the recommendations for the next batch of experiments.

        Args:
            batch_size: Number of requested recommendations.
            batch_quantity: Deprecated! Use ``batch_size`` instead.

        Returns:
            Dataframe containing the recommendations in experimental representation.

        Raises:
            ValueError: If ``batch_size`` is smaller than 1.
        """
        if batch_quantity is not None:
            raise DeprecationError(
                f"Passing the keyword 'batch_quantity' to "
                f"'{self.__class__.__name__}.{self.recommend.__name__}' is deprecated. "
                f"Use 'batch_size' instead."
            )

        if batch_size < 1:
            raise ValueError(
                f"You must at least request one recommendation per batch, but provided "
                f"{batch_size=}."
            )

        # If there are cached recommendations and the batch size of those is equal to
        # the previously requested one, we just return those
        if len(self._cached_recommendation) == batch_size:
            return self._cached_recommendation

        # Update recommendation meta data
        if len(self._measurements_exp) > 0:
            self.n_fits_done += 1
            self._measurements_exp.fillna({"FitNr": self.n_fits_done}, inplace=True)

        # Get the recommended search space entries
        rec = self.recommender.recommend(
            self.searchspace,
            batch_size,
            self._measurements_parameters_comp,
            self._measurements_targets_comp,
        )

        # Cache the recommendations
        self._cached_recommendation = rec.copy()

        # Telemetry
        telemetry_record_value(TELEM_LABELS["COUNT_RECOMMEND"], 1)
        telemetry_record_value(TELEM_LABELS["BATCH_SIZE"], batch_size)

        return rec


def _add_version(dict_: dict) -> dict:
    """Add the package version to the created dictionary."""
    from baybe import __version__

    return {**dict_, "version": __version__}


# Register de-/serialization hooks
unstructure_hook = cattrs.gen.make_dict_unstructure_fn(
    Campaign,
    converter,
    _cattrs_include_init_false=True,
    # TODO: Remove once deprecation got expired:
    numerical_measurements_must_be_within_tolerance=cattrs.override(omit=True),
    strategy=cattrs.override(omit=True),
)
structure_hook = cattrs.gen.make_dict_structure_fn(
    Campaign, converter, _cattrs_include_init_false=True
)
converter.register_unstructure_hook(
    Campaign, lambda x: _add_version(unstructure_hook(x))
)
converter.register_structure_hook(Campaign, structure_hook)
