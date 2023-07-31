"""Custom exceptions."""


class NotEnoughPointsLeftError(Exception):
    """An exception raised when more recommendations are requested than there are
    viable parameter configurations left in the search space."""


class NoMCAcquisitionFunctionError(Exception):
    """An exception raised when a Monte Carlo acquisition function is required
    but an analytical acquisition function has been selected by the user."""


class IncompatibleSearchSpaceError(Exception):
    """An exception raised when a recommender is used with a search space that contains
    incompatible parts, e.g. a discrete recommender is used with a hybrid or continuous
    search space."""


class EmptySearchSpaceError(Exception):
    """An exception raised when the created search space contains no parameters."""
