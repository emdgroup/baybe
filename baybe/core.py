# pylint: disable=E0401
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
        self.batch = 0  # current iteration/batch number

        # Parse everything from config
        parameters, targets = parse_config(config)
        self.parameters = parameters
        self.targets = targets
        self.config = config

        # Create the experimental dataframe
        self.searchspace_exp_rep = parameter_outer_prod_to_df(self.parameters)

        # Convert exp to comp dataframe
        self.searchspace_comp_rep = self.convert_rep_exp2comp(self.searchspace_exp_rep)

    def convert_rep_exp2comp(self, dataframe: pd.DataFrame):
        """
        Converts data in experimental representation to computational representation
        :param dataframe: dataframe in exp representation
        :return: comp_rep: dataframe in comp representation
        """
        comp_dfs = []
        for param in self.parameters:
            comp_df = param.transform_rep_exp2comp(dataframe[[param.name]], do_fit=True)
            comp_dfs.append(comp_df)

        comp_rep = pd.concat(comp_dfs, axis=1)
        return comp_rep

    def print_summary(self):
        """Print a simple summary of the BayBE object"""
        print("\nTarget and Parameters:")
        print(self.targets)
        for param in self.parameters:
            print(param)

        print("\nSearchspace Exp Representation:")
        print(self.searchspace_exp_rep)

        print("\nSearchspace Comp Representation:")
        print(self.searchspace_comp_rep)

    def add_results(self):
        """
        Adds results to the internal database
        """

    def initial_recommendation(self):
        """
        Performs an initial recommendation if there are no previous results
        """

    def recommend(self, batch_size=5):
        """
        Get the recommendation of the next batch
        """

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
