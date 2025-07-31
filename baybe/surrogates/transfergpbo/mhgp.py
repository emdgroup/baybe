"""MHGP surrogate implementations."""

from typing import Any, Literal

from attrs import define, field
from botorch.models.model import Model
from typing_extensions import override

from baybe.surrogates.transfergpbo.base import TransferGPBOSurrogate
from baybe.surrogates.transfergpbo.models import MHGPModel, MHGPModelStable


@define
class MHGPGaussianProcessSurrogate(TransferGPBOSurrogate):
    """BayBE wrapper for MHGP model with optional numerical stability.

    This surrogate implements the Multi-task Hierarchical Gaussian Process (MHGP)
    approach for transfer learning in Bayesian optimization. The model sequentially
    trains a stack of Gaussian processes where each GP models the residuals from
    the previous GPs in the stack.

    The key idea is that each GP uses the posterior mean of the previous GP as its
    prior mean, creating a hierarchical structure that enables transfer learning
    from source tasks to a target task.

    This implementation provides two variants:
    - **Basic**: Standard MHGP implementation
    - **Stable**: Enhanced numerical stability

    The stability enhancements include:
    - Nearest positive-definite matrix computation for covariance matrices
    - Positive definiteness checks during prediction
    - Iterative diagonal regularization for Cholesky decomposition

    Args:
        input_dim: Dimensionality of the input space (excluding task feature).
        numerical_stability: Whether to use numerically stable implementation.
            When True, uses MHGPModelStable with enhanced stability features.
            When False, uses basic MHGPModel implementation.
            Defaults to True (recommended for production use).

    Examples:
        >>> from baybe.parameters import NumericalContinuousParameter, TaskParameter
        >>> from baybe.searchspace import SearchSpace
        >>> from baybe.objectives import SingleTargetObjective
        >>> from baybe.targets import NumericalTarget
        >>> import pandas as pd
        >>>
        >>> # Create search space with transfer learning
        >>> parameters = [
        ...     NumericalContinuousParameter("temperature", bounds=(100, 200)),
        ...     NumericalContinuousParameter("pressure", bounds=(1, 10)),
        ...     TaskParameter("catalyst", values=["cat_A", "cat_B", "cat_C"],
        ...                  active_values=["cat_C"])
        ... ]
        >>> searchspace = SearchSpace.from_product(parameters)
        >>>
        >>> # Create objective
        >>> target = NumericalTarget("yield", mode="MAX")
        >>> objective = SingleTargetObjective(target)
        >>>
        >>> # Create stable surrogate (recommended for production)
        >>> surrogate_basic = MHGPGaussianProcessSurrogate(
        ...     input_dim=2,
        ...     numerical_stability=False
        ... )
        >>>
        >>> # Or create basic surrogate for well-conditioned problems
        >>> surrogate_basic = MHGPGaussianProcessSurrogate(
        ...     input_dim=2,
        ...     numerical_stability=False
        ... )
        >>>
        >>> # Training data with multiple tasks
        >>> measurements = pd.DataFrame({
        ...     "temperature": [120, 140, 160, 180, 150, 170],
        ...     "pressure": [2, 4, 6, 8, 3, 5],
        ...     "catalyst": ["cat_A", "cat_A", "cat_B", "cat_B", "cat_C", "cat_C"],
        ...     "yield": [0.6, 0.7, 0.65, 0.75, 0.8, 0.85]
        ... })
        >>>
        >>> surrogate.fit(searchspace, objective, measurements)
        >>>
        >>> # Make predictions for target task
        >>> test_data = pd.DataFrame({
        ...     "temperature": [130, 190],
        ...     "pressure": [2.5, 7.5],
        ...     "catalyst": ["cat_C", "cat_C"]  # Target task
        ... })
        >>> posterior = surrogate.posterior(test_data)
        >>> mean_predictions = posterior.mean
        >>> uncertainty = posterior.variance.sqrt()
        >>>
        >>> # Can also predict for source tasks
        >>> source_test = pd.DataFrame({
        ...     "temperature": [125],
        ...     "pressure": [3.5],
        ...     "catalyst": ["cat_A"]  # Source task
        ... })
        >>> source_posterior = surrogate.posterior(source_test)

    References:
        Tighineanu et al. (2022): Transfer Learning with Gaussian Processes for
        Bayesian Optimization. AISTATS 2022.
    """

    numerical_stability: bool = field(default=True)
    """Whether to use numerically stable implementation.

    When True, uses enhanced numerical stability features including:
    - Nearest positive-definite matrix computation for covariance matrices
    - Positive definiteness checks during prediction
    - Iterative diagonal regularization for Cholesky decomposition

    Recommended for production use, especially with small datasets or
    ill-conditioned problems.
    """

    @override
    def _create_model(self) -> Model:
        """Create MHGP model instance with optional stability enhancements.

        Returns:
            MHGPModelStable if numerical_stability=True, otherwise MHGPModel.
        """
        if self.numerical_stability:
            return MHGPModelStable(input_dim=self.input_dim)
        return MHGPModel(input_dim=self.input_dim)

    def __str__(self) -> str:
        """Return string representation of the MHGP surrogate."""
        stability_str = "Stable" if self.numerical_stability else "Basic"
        return f"MHGP ({stability_str})"


def get_mhgp_info() -> dict[str, Any]:
    """Get comprehensive information about MHGP surrogate implementations.

    Returns:
        Dictionary containing information about the MHGP surrogate variants,
        capabilities, and usage recommendations.

    Examples:
        >>> info = get_mhgp_info()
        >>> print(info["capabilities"]["transfer_learning"])  # True
        >>> print(info["variants"]["stable"]["recommended_for"])
    """
    return {
        "surrogates": {
            "MHGPGaussianProcessSurrogate": {
                "description": "Multi-task Hierarchical Gaussian Process with"
                "configurable stability",
                "class_name": "MHGPGaussianProcessSurrogate",
                "module": "baybe.surrogates.transfergpbo.mhgp",
            }
        },
        "variants": {
            "stable": {
                "description": "Numerically stable MHGP implementation",
                "parameter": "numerical_stability=True",
                "stability": "Enhanced",
                "recommended_for": "Production use, small datasets,"
                "ill-conditioned problems",
                "underlying_model": "MHGPModelStable",
                "features": [
                    "Nearest positive-definite matrix computation",
                    "Positive definiteness checks during prediction",
                    "Iterative diagonal regularization for Cholesky decomposition",
                ],
            },
            "basic": {
                "description": "Basic MHGP implementation",
                "parameter": "numerical_stability=False",
                "stability": "Standard",
                "recommended_for": "Well-conditioned problems, faster computation",
                "underlying_model": "MHGPModel",
                "features": [
                    "Standard hierarchical GP implementation",
                    "Faster computation for well-conditioned matrices",
                ],
            },
        },
        "capabilities": {
            "transfer_learning": True,
            "multi_output": False,
            "hierarchical_modeling": True,
            "residual_modeling": True,
            "numerical_stability": True,
            "configurable_stability": True,
        },
        "usage": {
            "default_config": "numerical_stability=True",
            "recommendation": "Use stable variant (default) for production",
            "input_requirements": "Requires TaskParameter in search space",
        },
        "reference": "Tighineanu et al. (2022): Transfer Learning with Gaussian"
        "Processes for Bayesian Optimization",
        "paper_url": "https://arxiv.org/abs/2111.11223",
    }


def create_mhgp_surrogate(
    input_dim: int, variant: Literal["stable", "basic"] = "stable"
) -> MHGPGaussianProcessSurrogate:
    """Create an MHGP surrogate with specified variant.

    Args:
        input_dim: Dimensionality of the input space (excluding task feature).
        variant: Which variant to use:
            - "stable": Enhanced numerical stability (recommended for production)
            - "basic": Standard implementation (faster for well-conditioned problems)

    Returns:
        Configured MHGPGaussianProcessSurrogate instance.
    """
    numerical_stability = variant == "stable"
    return MHGPGaussianProcessSurrogate(
        input_dim=input_dim, numerical_stability=numerical_stability
    )
