"""Utilities targeting random number generation."""

import contextlib
import os
import random

import numpy as np


def set_random_seed(seed: int):
    """Set the global random seed.

    Args:
        seed: The chosen global random seed.
    """
    import torch

    # Ensure seed limits
    seed %= 2**32

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # TODO the later is not guaranteed to be reproducible across machines anyways
    #   Also it should be set depending on the machine setup


@contextlib.contextmanager
def temporary_seed(seed: int):  # noqa: DOC402, DOC404
    """Context manager for setting a temporary random seed.

    Args:
        seed: The chosen random seed.
    """
    import torch

    # Ensure seed limits
    seed %= 2**32

    # Collect the current RNG states
    state_builtin = random.getstate()
    state_np = np.random.get_state()
    state_torch = torch.get_rng_state()

    torch_deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    torch_device = torch.get_default_device()
    torch_dtype = torch.get_default_dtype()
    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    mkl_num_threads = os.environ.get("MKL_NUM_THREADS")
    openblas_num_threads = os.environ.get("OPENBLAS_NUM_THREADS")
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    # Set the requested seed
    set_random_seed(seed)

    # Run the context-specific code
    try:
        yield

    # Restore the original RNG states
    finally:
        random.setstate(state_builtin)
        np.random.set_state(state_np)
        torch.set_rng_state(state_torch)

        torch.use_deterministic_algorithms(torch_deterministic)
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.set_default_device(torch_device)
        torch.set_default_dtype(torch_dtype)

        def reset_environment_variable(name: str, value: str | None):
            """Reset an environment variable."""
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value

        reset_environment_variable("OMP_NUM_THREADS", omp_num_threads)
        reset_environment_variable("MKL_NUM_THREADS", mkl_num_threads)
        reset_environment_variable("OPENBLAS_NUM_THREADS", openblas_num_threads)
        reset_environment_variable("CUBLAS_WORKSPACE_CONFIG", cublas_workspace_config)
