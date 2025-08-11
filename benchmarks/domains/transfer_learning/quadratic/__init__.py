"""Quadratic transfer learning benchmarks."""

from benchmarks.domains.transfer_learning.quadratic.diff_min_few_sources import (
    quadratic_diff_min_few_sources_tl_benchmark,
)
from benchmarks.domains.transfer_learning.quadratic.diff_min_many_sources import (
    quadratic_diff_min_many_sources_tl_benchmark,
)
from benchmarks.domains.transfer_learning.quadratic.same_min_few_sources import (
    quadratic_same_min_few_sources_tl_benchmark,
)

__all__ = [
    "quadratic_same_min_few_sources_tl_benchmark",
    "quadratic_diff_min_few_sources_tl_benchmark",
    "quadratic_diff_min_many_sources_tl_benchmark",
]
