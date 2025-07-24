"""Utils for numerically stable Cholesky decomposition."""

import itertools

import torch
from torch import Tensor


def is_pd(matrix: Tensor) -> bool:
    """Check if matrix is positive definite via Cholesky decomposition."""
    try:
        torch.linalg.cholesky(matrix)
        return True
    except torch.linalg.LinAlgError:
        return False


def nearest_pd(matrix: Tensor) -> Tensor:
    """Calculate nearest positive-definite matrix (following TransferGPBO approach)."""
    # Eigendecomposition
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)

    # Account for floating-point accuracy
    spacing = torch.finfo(matrix.dtype).eps * torch.norm(matrix)

    # Clip eigenvalues at spacing
    eigenvals_clipped = torch.clamp(eigenvals, min=spacing)

    # Reconstruct matrix
    return eigenvecs @ torch.diag(eigenvals_clipped) @ eigenvecs.T


def compute_cholesky(matrix: Tensor) -> Tensor:
    """Compute Cholesky with iterative diagonal regularization (TransferGPBO style)."""
    matrix_copy = matrix.clone()

    for k in itertools.count(start=1):
        try:
            return torch.linalg.cholesky(matrix_copy)
        except torch.linalg.LinAlgError:
            # Add diagonal regularization (same as TransferGPBO)
            jitter = (10**k) * 1e-8
            matrix_copy = matrix_copy + jitter * torch.eye(
                matrix_copy.shape[-1],
                dtype=matrix_copy.dtype,
                device=matrix_copy.device,
            )
            if k > 10:  # Prevent infinite loop
                raise RuntimeError("Could not make matrix positive definite")
