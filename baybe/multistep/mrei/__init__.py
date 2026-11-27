"""Model-reference expected improvement for DAG-valued objectives.

``ModelReferenceExpectedImprovement`` (``MREI``) scores node actions with an
expected improvement whose reference comes from the fitted network itself, so
no real terminal outcome is required.
"""

from .acquisition import MREI, ModelReferenceExpectedImprovement

__all__ = ["MREI", "ModelReferenceExpectedImprovement"]
