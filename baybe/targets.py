# pylint: disable=R0903,W0235
"""
Functionality for different type of targets
"""


class GenericTarget:
    """
    Base class for different Targets. Will handle storing info about the type, the
    range and constraints
    """

    def __init__(self, name: str = "Target"):
        self.name = name


class Numerical(GenericTarget):
    """
    Class for numerical targets
    """

    def __init__(self, limits: tuple = None):
        super().__init__()

        self.limits = None if limits is None else limits


class Categorical(GenericTarget):
    """
    Class for categorical targets
    """

    def __init__(self):
        super().__init__()
