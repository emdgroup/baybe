"""Hypothesis strategies for kernels."""

import hypothesis.strategies as st

from baybe.kernels import MaternKernel

matern_kernels = st.builds(MaternKernel, st.sampled_from((0.5, 1.5, 2.5)))
