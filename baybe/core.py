"""
Core functionality of BayBE. Main point of interaction via Python
"""
import logging
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Extra, validator

import baybe.parameters as baybe_parameters
from baybe.parameters import Parameter, ParameterConfig
from baybe.targets import ObjectiveConfig, Target

log = logging.getLogger(__name__)


class BayBEConfig(BaseModel, extra=Extra.forbid):
    """Configuration class for BayBE."""

    project_name: str
    parameters: List[dict]
    objective: dict
    random_seed: int = 1337
    allow_repeated_recommendations: bool = True
    allow_recommending_already_measured: bool = True
    numerical_measurements_must_be_within_tolerance: bool = True

    @validator("parameters")
    def validate_parameters(cls, param_specs):
        """Turns the given list of parameters specifications into config objects."""
        return [ParameterConfig(**param) for param in param_specs]

    @validator("objective")
    def validate_objective(cls, objective_specs):
        """Turns the given objective specifications into a config object."""
        return ObjectiveConfig(**objective_specs)


class BayBE:
    """
    Main class for interaction with baybe
    """

    def __init__(self, config: BayBEConfig):
        self.batches_done = 0  # current iteration/batch number

        # Create the parameter and target objects
        self.config = config
        self.parameters = [Parameter.create(p) for p in config.parameters]
        self.targets = [Target.create(t) for t in config.objective.targets]

        # Create the experimental dataframe
        self.searchspace_exp_rep = baybe_parameters.parameter_outer_prod_to_df(
            self.parameters
        )
        self.searchspace_metadata = pd.DataFrame(
            {
                "was_recommended": False,
                "was_measured": False,
                "dont_recommend": False,
            },
            index=self.searchspace_exp_rep.index,
        )

        # Convert exp to comp dataframe
        self.searchspace_comp_rep, _ = self.transform_rep_exp2comp(
            self.searchspace_exp_rep
        )

        # Reset measurement dataframes
        self.measurements_exp_rep = None
        self.measurements_comp_rep_x = None
        self.measurements_comp_rep_y = None

    def transform_rep_exp2comp(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform a dataframe in experimental representation to the computational
        representation

        Parameters
        ----------
        data: pd.DataFrame
            Data to be transformed. Must contain parameter columns. If it also contains
            target columns these are transformed separately
        Returns
        -------
        2-tuple. First part is the X part (representing the parameters) and the second
        part is the Y part. The first part is ignored if the target columns are not in
        the data (This is the case if data corresponds to the search space).
        """
        # parameter part
        dfs = []
        for param in self.parameters:
            comp_df = param.transform_rep_exp2comp(data[param.name])
            dfs.append(comp_df)

        comp_rep_x = pd.concat(dfs, axis=1)

        # target part, unlike parameters targets can be missing from the dataframe
        comp_rep_y = None
        if all(target.name in data.columns for target in self.targets):
            dfs = []
            for target in self.targets:
                comp_df = target.transform(data[[target.name]])
                dfs.append(comp_df)
            comp_rep_y = pd.concat(dfs, axis=1)

        return comp_rep_x, comp_rep_y

    def __str__(self):
        """Print a simple summary of the BayBE object. Some info provided here might
        not be relevant for production-ready code"""

        string = "\nTarget and Parameters:\n"
        for target in self.targets:
            string += f"{target}\n"
        for param in self.parameters:
            string += f"{param}\n"

        string += "Options:\n"
        for option, value in self.config.dict().items():
            string += f"   {option}: {value}\n"

        string += "\n\nSearch Space Exp Representation:\n"
        string += f"{self.searchspace_exp_rep}"

        string += "\n\nSearch Space Comp Representation:\n"
        string += f"{self.searchspace_comp_rep}"

        string += "\n\nMeasurement Space Exp Representation:\n"
        string += f"{self.measurements_exp_rep}"

        string += "\n\nMeasurement Space Comp Representation:\n"
        string += f"{self.measurements_comp_rep_x}\n"
        string += f"{self.measurements_comp_rep_y}"

        return string

    def _match_measurement_with_searchspace_indices(
        self, data: pd.DataFrame
    ) -> pd.Index:
        """
        Matches rows in a data frame (measurements to be added to the internal data)
        to the indices in the searchspace. This is useful to have a validity check as
        well as automatically match measurements to entries int he searchspace to
        detect which ones have been measured. For categorical parameters there needs
        to be an exact match with any of the allowed values. For numerical parameters
        the user can decide via a BayBE flag whether values falling outside of the
        tolerance should be accepted. In that case no match with any searchspace entry
        will be detected.

        Parameters
        ----------
        data: pd.DataFrame
            The data that should be checked for matching entries in the searchspace

        Returns
        -------
        inds_matched: pd.Index
            The index of the matching searchspace entries
        """
        # IMPROVE neater implementation eg with fuzzy join

        inds_matched = []

        # Iterating over all input rows
        for ind, row in data.iterrows():
            # Check if row is valid input
            test = True
            for param in self.parameters:
                if "NUM" in param.TYPE:
                    if self.config.numerical_measurements_must_be_within_tolerance:
                        test &= param.is_in_range(row[param.name])
                else:
                    test &= param.is_in_range(row[param.name])

                if not test:
                    raise ValueError(
                        f"Input data on row with the index {row.name} has invalid "
                        f"values in parameter {param.name}. For categorical parameters "
                        f"values need to exactly match a valid choice defined in your "
                        f"config. For numerical parameters a match is accepted only if "
                        f"the input value is within the specified tolerance/range. Set "
                        f"the flag 'numerical_measurements_must_be_within_tolerance' "
                        f"to False to turn this behavior off."
                    )

            # Find to what indices in the searchspace this row corresponds to
            # Cat columns look for exact match. Num columns look to match the entry
            # with min deviation.
            # TODO Discuss the different scenarios that are possible
            cat_cols = [
                param.name for param in self.parameters if "NUM" not in param.TYPE
            ]
            num_cols = [param.name for param in self.parameters if "NUM" in param.TYPE]

            match = (self.searchspace_exp_rep.loc[:, cat_cols] == row[cat_cols]).all(
                axis=1, skipna=False
            )

            for numparam in list(num_cols):
                match &= (
                    self.searchspace_exp_rep.loc[:, numparam] - row[numparam]
                ).abs() == (
                    self.searchspace_exp_rep.loc[:, numparam] - row[numparam]
                ).abs().min()

            # We expect exactly one match, if thats not the case print a warning
            inds_found = self.searchspace_exp_rep.index[match].to_list()
            if len(inds_found) == 0:
                log.warning(
                    "Input row with index %s could not be matched to the "
                    "search space. This could indicate that something went "
                    "wrong.",
                    ind,
                )
            elif len(inds_found) > 1:
                log.warning(
                    "Input row with index %s corresponds to multiple matches with "
                    "the searcspace. This indicates that something could be wrong. "
                    "Matching only first occurrence.",
                    ind,
                )
                inds_matched.append(inds_found[0])
            else:
                inds_matched += inds_found

        return pd.Index(inds_matched)

    def add_results(self, data: pd.DataFrame) -> None:
        """
        Adds results from a dataframe to the internal database and update strategy.
        Each addition of data is considered a new batch. Added results are checked for
        validity. Categorical values need to have an exact match. For numerical values
        there is a flag so that values are only allowed if they are within the
        specified tolerance or not.

        Parameters
        ----------
        data : pandas DataFrame
            the dataframe with the measurements. Preferably created via the recommend
            method and with filled values for targets.

        Returns
        -------
        Nothing
        """

        # Check whether all provided data points have acceptable parameter values and
        # change the was_measured metadata
        inds_matched = self._match_measurement_with_searchspace_indices(data)
        self.searchspace_metadata.loc[inds_matched, "was_measured"] = True

        # Check if all targets have values provided
        for target in self.targets:
            if data[target.name].isna().any():
                raise NotImplementedError(
                    f"The target {target.name} has missing values or NaN in the"
                    f" provided dataframe. Missing target values are not currently"
                    f" supported."
                )

        # Read in measurements and add to database
        self.batches_done += 1
        data["BatchNr"] = self.batches_done
        if self.measurements_exp_rep is None:
            self.measurements_exp_rep = data.reset_index(drop=True)
        else:
            self.measurements_exp_rep = pd.concat(
                [self.measurements_exp_rep, data], axis=0
            ).reset_index(drop=True)

        # Transform measurement space to computational representation
        (
            self.measurements_comp_rep_x,
            self.measurements_comp_rep_y,
        ) = self.transform_rep_exp2comp(self.measurements_exp_rep)

        # TODO call strategy for retraining

    def recommend(self, batch_quantity: int = 5) -> pd.DataFrame:
        """
        Get the recommendation of the next batch
        """

        # Filter searchspace before transferring to strategy
        mask_todrop = self.searchspace_metadata["dont_recommend"].copy()
        if not self.config.allow_repeated_recommendations:
            mask_todrop |= self.searchspace_metadata["was_recommended"]
        if not self.config.allow_recommending_already_measured:
            mask_todrop |= self.searchspace_metadata["was_measured"]

        if (mask_todrop.sum() >= len(self.searchspace_exp_rep)) or (
            len(self.searchspace_exp_rep.loc[~mask_todrop]) < batch_quantity
        ):
            raise AssertionError(
                f"With the current settings there are not at at least "
                f"batch_quantity={batch_quantity} possible data points to"
                " recommend. This can be either because all data points have been"
                " measured at some point (while 'allow_repeated_recommendations' or "
                "'allow_recommending_already_measured' being False) or because all"
                " data points are marked as 'dont_recommend'"
            )

        # Get indices of recommended searchspace entries here
        # TODO call actual strategy object
        inds = self.searchspace_exp_rep.loc[~mask_todrop].sample(n=batch_quantity).index

        # Translate indices into labeled data points and update metadata
        rec = self.searchspace_exp_rep.loc[inds, :]
        self.searchspace_metadata.loc[inds, "was_recommended"] = True

        for target in self.targets:
            rec[target.name] = "<Enter value>"

        return rec

    def load(self) -> None:
        """
        Load new internal state of a DOE from a specified file
        The load and save functions could also be omitted and the user would have to
        take care of simply storing the BayBE object eg via dill. This could
        potentially create problems when code versions are different
        """
        # TODO Implement
        raise NotImplementedError("Loading a BayBE object is not implemented yet")

    def save(self) -> None:
        """
        Store the current state of the DOE on disk
        """
        # TODO Implement
        raise NotImplementedError("Saving a BayBE object is not implemented yet")


def add_fake_results(
    data: pd.DataFrame,
    obj: BayBE,
    good_reference_values: Optional[Dict[str, List]] = None,
    good_intervals: Optional[Tuple] = None,
    bad_intervals: Optional[Tuple] = None,
) -> None:
    """
    Add fake results to a dataframe which was the result of the BayBE recommendation
    action. It is possible to identify "good" values, which will be given a better
    target value. With this the algorithm can be driven towards certain optimal values
    whilst still being random. Useful for testing.

    Parameters
    ----------
    data : pandas dataframe
           Output of the recommend function of a BayBE object
    obj : BayBE class instance
          The baybe object which provides configuration, targets, etc.
    good_reference_values : dictionary
                  A dictionaries which defines parameters and respective values
                  which identify what will be considered good values.
                  Example {'parameter1': [1,4,42]}
    good_intervals : 2-tuple
                     Good entries will get a random value in the range defined by this
                     tuple
    bad_intervals : 2-tuple
                    Bad entries will get a random value in the range defined by this
                    tuple

    Returns
    -------
    Nothing since it operated directly on the data
    """
    # TODO Add support for multiple targets

    # Sanity checks for good_bad_ratio
    if good_intervals is None:
        if obj.targets[0].mode == "MAX":
            good_intervals = (66, 100)
        elif obj.targets[0].mode == "MIN":
            good_intervals = (0, 33)
        elif obj.targets[0].mode == "MATCH":
            good_intervals = tuple(*obj.targets[0].bounds)
        else:
            raise ValueError("Unrecognized target mode when trying to add fake values.")
    if bad_intervals is None:
        if obj.targets[0].mode == "MAX":
            bad_intervals = (0, 50)
        elif obj.targets[0].mode == "MIN":
            bad_intervals = (50, 100)
        elif obj.targets[0].mode == "MATCH":
            bad_intervals = (
                0.05 * obj.targets[0].bounds[0],
                0.3 * obj.targets[0].bounds[0],
            )
        else:
            raise ValueError("Unrecognized target mode when trying to add fake values.")
    if not isinstance(good_intervals, Tuple) or (len(good_intervals) != 2):
        raise TypeError("Parameter good_intervals must be a 2-tuple")
    if not isinstance(bad_intervals, Tuple) or (len(bad_intervals) != 2):
        raise TypeError("Parameter bad_intervals must be a 2-tuple")

    # Sanity check for good_values. Assure we only consider columns that are in the data
    if good_reference_values is None:
        good_reference_values = {}

    size = len(data)
    for target in obj.targets:
        # add bad values
        data[target.name] = np.random.randint(bad_intervals[0], bad_intervals[1], size)

        # add good values
        masks = []

        if len(good_reference_values) > 0:
            for param, vals in good_reference_values.items():
                if param not in data.columns:
                    raise ValueError(
                        f"When adding fake results you specified good "
                        f"values for the parameter '{param}' but this "
                        f"parameter is not in the dataframe."
                    )
                if not isinstance(vals, list):
                    raise TypeError(
                        f"Entries in parameter good_reference_values "
                        f"(which is a dictionary) must be lists, but you "
                        f"provided {vals}"
                    )
                mask = data[param].apply(partial(lambda x, v: x in v, v=vals))
                masks.append(mask)

            # Good values will be added where the parameters of the
            # corresponding datapoints match the ones defined in good_reference_values
            for k, mask in enumerate(masks):
                if k == 0:
                    final_mask = mask

                final_mask &= mask

            data.loc[final_mask, target.name] = np.random.randint(
                good_intervals[0], good_intervals[1], final_mask.sum()
            )


def add_noise(
    data: pd.DataFrame,
    obj: BayBE,
    noise_type: str = "absolute",
    noise_level: float = 1.0,
):
    """
    Adds uniform noise to parameter values of a recommendation frame. Simulates
    experimental noise and inputting numerical values that are slightly different
    than the recommendations coming from the search space.

    Parameters
    ----------
    data : pandas dataframe
           output of the recommend function of a BayBE object
    obj : BayBE class instance
          the baybe object which provides configuration, targets, etc.
    noise_type : str
        Defines whether the noise should be additive
    noise_level : float
        Level/magnitude of the noise, numerical value for type absolute and percentage
        for type relative_percent

    Returns
    -------
        Nothing
    """
    for param in obj.parameters:
        if "NUM" in param.TYPE:
            if noise_type == "relative_percent":
                data[param.name] *= np.random.uniform(
                    1.0 - noise_level / 100.0, 1.0 + noise_level / 100.0, len(data)
                )
            elif noise_type == "absolute":
                data[param.name] += np.random.uniform(
                    -noise_level, noise_level, len(data)
                )
            else:
                raise ValueError(
                    f"Parameter noise_type was {noise_type} but must be either "
                    f'"absolute" or "relative_percent"'
                )
