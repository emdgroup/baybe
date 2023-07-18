"""BayBE — A Bayesian Back End for Design of Experiments."""

from importlib.metadata import PackageNotFoundError, version
from importlib.resources import path

from setuptools_scm import get_version

from baybe.core import BayBE

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass

try:
    with path(__name__, "") as package_folder:
        __version__ = get_version(
            root=str(package_folder / ".."),
            version_scheme="post-release",
            local_scheme="dirty-tag",
        )
except LookupError:
    __version__ = "unknown"

del version
del PackageNotFoundError
del path
del get_version

__all__ = [
    "__version__",
    "BayBE",
]
