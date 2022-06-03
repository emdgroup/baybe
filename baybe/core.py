"""
Core functionality of BayBE. Main point of interaction via Python
"""


class BayBE:
    """
    Main class for interaction with baybe
    """

    def __init__(self):
        self.batch = 0  # current iteration/batch number
        self.config = None

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
