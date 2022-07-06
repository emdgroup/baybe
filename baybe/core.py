"""
Core functionality of BayBE. Main point of interaction via Python
"""
import logging

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
                "has_been_recommended": [False] * len(self.searchspace_exp_rep),
                "do_not_recommend": [False] * len(self.searchspace_exp_rep),
            },
            index=self.searchspace_exp_rep.index,
        )

        # Convert exp to comp dataframe
        self.searchspace_comp_rep = self.convert_rep_exp2comp(
            self.searchspace_exp_rep, do_fit=True
        )

        # Reset measurement dataframes
        self.measurements_exp_rep = None
        self.measurements_comp_rep_x = None
        self.measurements_comp_rep_y = None

    def convert_rep_exp2comp(self, dataframe: pd.DataFrame, do_fit: bool = False):
        """
        Converts data in experimental representation to computational representation
        :param dataframe: dataframe in exp representation
        :return: comp_rep: dataframe in comp representation
        """
        comp_dfs = []
        for param in self.parameters:
            comp_df = param.transform_rep_exp2comp(dataframe[param.name], do_fit=do_fit)
            comp_dfs.append(comp_df)

        comp_rep = pd.concat(comp_dfs, axis=1)
        return comp_rep

    def __str__(self):
        """Print a simple summary of the BayBE object"""
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
        Adds results from a dataframe to the internal database. Each addition of data is
         considered a batch.

        Parameters
        ----------
        data : pandas DataFrame
               the dataframe with the measurements. Preferab ly created via the
               recommend method and with filled values for targets

        Returns
        -------
        Nothing
        """

        self.batches_done += 1

        # Check whether all provided data points have acceptable parameter values
        for _, row in data.iterrows():
            not_acceptable = [
                param.name
                for param in self.parameters
                if not param.is_in_range(row[param.name])
            ]
            if len(not_acceptable) > 0:
                log.error(
                    "Parameter values for data point with index %s are not in the "
                    "allowed values for parameters %s",
                    row.name,
                    not_acceptable,
                )
                raise ValueError(
                    "When adding measurement values it is required that all input "
                    "values for the parameters are allowed values. This means "
                    "hard-match for categorical and substance parameters or within the"
                    " given tolerance/interval for numerical discrete/continuous "
                    "parameters."
                )

        # Read in measurements and transform
        # ToDo match indices to search space and remember indices
        data["BatchNr"] = self.batches_done
        if self.measurements_exp_rep is None:
            self.measurements_exp_rep = data.reset_index(drop=True)
        else:
            self.measurements_exp_rep = pd.concat(
                [self.measurements_exp_rep, data], axis=0
            ).reset_index(drop=True)

        self.measurements_comp_rep_x = self.convert_rep_exp2comp(
            self.measurements_exp_rep, do_fit=False
        )

    def recommend(self, batch_quantity=5) -> pd.DataFrame:
        """
        Get the recommendation of the next batch
        """

        # Get indices of recommended searchspace entries here
        # Connect this part to the actual recommender strat
        # so far these are just randomly selected
        # strats should also have the capability to ignore some datapoints
        # (look up do_not_recommend in metadata) or do not suggest repeated ones
        # (lookup has_been_recommended in metadata)
        inds = self.searchspace_exp_rep.sample(n=batch_quantity).index

        # Translate indices into labeled datapoints and update metadata
        rec = self.searchspace_exp_rep.loc[inds, :]
        self.searchspace_metadata.loc[inds, "has_been_recommended"] = True

        for target in self.targets:
            rec[target.name] = "<Enter value>"

        return rec

    def load(self):
        """
        Load new internal state of a DOE from a specified file
        The load and save functions could also be omitted and the user would have to
        take care of simply storing the BayBE object eg via dill. This could
        potentially create problems when code versions are different
        """

    def save(self):
        """
        Store the current state of the DOE on disk
        """
