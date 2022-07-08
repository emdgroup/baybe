"""
Core functionality of BayBE. Main point of interaction via Python
"""
import logging
from typing import Tuple

import pandas as pd

from baybe.config import parse_config
from baybe.parameters import parameter_outer_prod_to_df

log = logging.getLogger(__name__)


class BayBE:
    """
    Main class for interaction with baybe
    """

    def __init__(self, config: dict):
        self.batches_done = 0  # current iteration/batch number

        # Parse everything from config
        parameters, targets = parse_config(config)
        self.parameters = parameters
        self.targets = targets
        self.config = config

        # Create the experimental dataframe
        self.searchspace_exp_rep = parameter_outer_prod_to_df(self.parameters)
        self.searchspace_metadata = pd.DataFrame(
            {
                "was_recommended": [False] * len(self.searchspace_exp_rep),
                "was_measured": [False] * len(self.searchspace_exp_rep),
                "dont_recommend": [False] * len(self.searchspace_exp_rep),
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
        """Print a simple summary of the BayBE object"""
        # ToDo add options info

        string = "\nTarget and Parameters:\n"
        for target in self.targets:
            string += f"{target}\n"
        for param in self.parameters:
            string += f"{param}\n"

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

    def add_results(self, data: pd.DataFrame) -> None:
        """
        Adds results from a dataframe to the internal database and retrains strategy.
        Each addition of data is considered a new batch.

        Parameters
        ----------
        data : pandas DataFrame
            the dataframe with the measurements. Preferably created via the recommend
            method and with filled values for targets.

        Returns
        -------
        Nothing
        """

        # Check whether all provided data points have acceptable parameter values
        for _, row in data.iterrows():
            unacceptable_parameters = [
                param.name
                for param in self.parameters
                if not param.is_in_range(row[param.name])
            ]
            if len(unacceptable_parameters) > 0:
                log.error(
                    "Parameter values for data point with index %s are not in the "
                    "allowed values for parameters %s",
                    row.name,
                    unacceptable_parameters,
                )
                raise ValueError(
                    "When adding measurement values it is required that all input "
                    "values for the parameters are allowed values. This means "
                    "exact match for categorical and substance parameters or within the"
                    " given tolerance/interval for numerical discrete/continuous "
                    "parameters."
                )

        # Check if all targets have values provided
        for target in self.targets:
            if data[target.name].isna().any():
                raise NotImplementedError(
                    f"The target {target.name} has missing values or NaN in the"
                    f" provided dataframe. Missing target values are not currently"
                    f" supported."
                )

        # Read in measurements and add to database
        # ToDo match indices to search space and remember indices
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

        # ToDo call strategy for training

    def recommend(self, batch_quantity: int = 5) -> pd.DataFrame:
        """
        Get the recommendation of the next batch
        """

        # Filter searchspace before transferring to strategy
        mask_todrop = self.searchspace_metadata["dont_recommend"].copy()
        if not self.config["Allow_repeated_recommendations"]:
            mask_todrop |= self.searchspace_metadata["was_recommended"]
        if not self.config["Allow_recommending_already_measured"]:
            mask_todrop |= self.searchspace_metadata["was_measured"]

        if mask_todrop.sum() >= len(self.searchspace_exp_rep):
            raise AssertionError(
                "With the current settings there are no more possible data points to"
                " recommend. This can be either because all data points have been"
                " measured at some point while not allowing repetitions or by all"
                " data points being marked as 'dont_recommend'"
            )

        # Get indices of recommended searchspace entries here
        # ToDo call actual strategy object
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

    def save(self) -> None:
        """
        Store the current state of the DOE on disk
        """
