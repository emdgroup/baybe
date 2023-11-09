"""Custom exceptions."""


class NotEnoughPointsLeftError(Exception):
    """
    More recommendations are requested than there are viable parameter configurations
    left in the search space.
    """


class NoMCAcquisitionFunctionError(Exception):
    """
    A Monte Carlo acquisition function is required but an analytical acquisition
    function has been selected by the user.
    """


class IncompatibleSearchSpaceError(Exception):
    """
    A recommender is used with a search space that contains incompatible parts,
    e.g. a discrete recommender is used with a hybrid or continuous search space.
    """


class EmptySearchSpaceError(Exception):
    """The created search space contains no parameters."""


class NothingToSimulateError(Exception):
    """There is nothing to simulate because there are no testable configurations."""


# TODO: This exception class is needed only temporarily in `CustomONNXSurrogate` until
#   the use of `model_params` has be refactored.
class ModelParamsNotSupportedError(Exception):
    """The class does not support any model parameters."""


class NoRecommendersLeftError(Exception):
    """A recommender is requested by a strategy but there are no recommenders left."""
