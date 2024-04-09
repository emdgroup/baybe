"""BayBE — A Bayesian Back End for Design of Experiments."""

import warnings

from baybe.campaign import Campaign

# Show deprecation warnings
warnings.filterwarnings("default", category=DeprecationWarning, module="baybe")


def infer_version() -> str:  # pragma: no cover
    """Determine the package version for the different ways the code can be invoked."""
    # ----------------------------------------------------------------------------------
    # Attempt 1:
    # If the package has been installed, read the version from the metadata created
    # during the installation process. In this case, the version string has been
    # determined via setuptools_scm from the git history. (If the install had been
    # attempted without the git folder in place, setuptools_scm would have complained,
    # causing the installation process to fail.)
    from importlib.metadata import version

    try:
        return version(__name__)
    except ModuleNotFoundError:
        pass

    # ----------------------------------------------------------------------------------
    # Attempt 2:
    # If the package is not installed, try to replicate the version on the fly, using
    # the same logic by invoking setuptools_scm.
    from pathlib import Path

    from setuptools_scm import get_version

    try:
        return get_version(
            root=str(Path(__path__[0]).parent),
            version_scheme="post-release",
            local_scheme="dirty-tag",
        )
    except LookupError:
        pass

    # ----------------------------------------------------------------------------------
    # Fallback:
    # If the package has not been installed and also no git history is available,
    # there is no way to figure out the correct version.
    return "unknown"


__version__ = infer_version()
__all__ = [
    "__version__",
    "Campaign",
]

del infer_version
