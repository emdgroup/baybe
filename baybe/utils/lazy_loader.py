"""A utility for loading heavier modules lazily."""

import importlib

from attr import define, field


@define
class LazyLoader:
    """A class responsible for lazy loading module."""

    # Object variable
    module_name: str = field()
    """String representation of the module we wish to load."""

    # Object variable
    module = field(init=False, default=None)
    """The imported module"""

    def load(self):
        """Load the required module using importlib."""
        if self.module is None:
            self.module = importlib.import_module(self.module_name)
        return self.module
