"""EDBO preset :cite:p:`Shields2021`."""

from __future__ import annotations

import gc
from collections.abc import Collection
from typing import TYPE_CHECKING, ClassVar

from attrs import define
from typing_extensions import override

from baybe.kernels.basic import MaternKernel
from baybe.kernels.composite import ScaleKernel
from baybe.parameters.enum import SubstanceEncoding, _ParameterKind
from baybe.parameters.substance import SubstanceParameter
from baybe.priors.basic import GammaPrior
from baybe.searchspace.discrete import SubspaceDiscrete
from baybe.surrogates.gaussian_process.components.fit_criterion import (
    _MLLForNonTLFitCriterionFactory,
)
from baybe.surrogates.gaussian_process.components.kernel import (
    _enable_transfer_learning,
    _PureKernelFactory,
)
from baybe.surrogates.gaussian_process.components.likelihood import (
    LikelihoodFactoryProtocol,
)
from baybe.surrogates.gaussian_process.components.mean import LazyConstantMeanFactory

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from torch import Tensor

    from baybe.kernels.base import Kernel
    from baybe.searchspace.core import SearchSpace


def _contains_encoding(
    subspace: SubspaceDiscrete, encodings: Collection[SubstanceEncoding]
) -> bool:
    """Tell if any of the substance parameters uses one of the specified encodings."""
    return any(
        p.encoding in encodings
        for p in subspace.parameters
        if isinstance(p, SubstanceParameter)
    )


_EDBO_ENCODINGS = (
    SubstanceEncoding.MORDRED,
    SubstanceEncoding.RDKIT,
    SubstanceEncoding.RDKIT2DDESCRIPTORS,
)
"""Encodings relevant to EDBO logic."""


@_enable_transfer_learning
@define
class EDBOKernelFactory(_PureKernelFactory):
    """A factory providing EDBO kernels, as proposed by :cite:p:`Shields2021`.

    GitHub repository: https://github.com/b-shields/edbo
    Prior settings: https://github.com/b-shields/edbo/blob/9b41eac3f6d9e520547702fd5b0c7ef6441625a4/edbo/bro.py#L658
    """

    _uses_parameter_names: ClassVar[bool] = True
    # See base class.

    @override
    def _make(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        effective_dims = self._get_effective_dimensionality(searchspace)

        switching_condition = _contains_encoding(
            searchspace.discrete, _EDBO_ENCODINGS
        ) and (effective_dims >= 50)

        # low D priors
        if effective_dims < 5:
            lengthscale_prior = GammaPrior(1.2, 1.1)
            lengthscale_initial_value = 0.2
            outputscale_prior = GammaPrior(5.0, 0.5)
            outputscale_initial_value = 8.0

        # DFT optimized priors
        elif switching_condition and effective_dims < 100:
            lengthscale_prior = GammaPrior(2.0, 0.2)
            lengthscale_initial_value = 5.0
            outputscale_prior = GammaPrior(5.0, 0.5)
            outputscale_initial_value = 8.0

        # Mordred optimized priors
        elif switching_condition:
            lengthscale_prior = GammaPrior(2.0, 0.1)
            lengthscale_initial_value = 10.0
            outputscale_prior = GammaPrior(2.0, 0.1)
            outputscale_initial_value = 10.0

        # OHE optimized priors
        else:
            lengthscale_prior = GammaPrior(3.0, 1.0)
            lengthscale_initial_value = 2.0
            outputscale_prior = GammaPrior(5.0, 0.2)
            outputscale_initial_value = 20.0

        return ScaleKernel(
            MaternKernel(
                nu=2.5,
                lengthscale_prior=lengthscale_prior,
                lengthscale_initial_value=lengthscale_initial_value,
                parameter_names=self.get_parameter_names(searchspace),
            ),
            outputscale_prior=outputscale_prior,
            outputscale_initial_value=outputscale_initial_value,
        )


class EDBOMeanFactory(LazyConstantMeanFactory):
    """A factory providing mean functions for the EDBO preset."""


@define
class EDBOLikelihoodFactory(LikelihoodFactoryProtocol):
    """A factory providing EDBO likelihoods, as proposed by :cite:p:`Shields2021`.

    GitHub repository: https://github.com/b-shields/edbo
    Prior settings: https://github.com/b-shields/edbo/blob/9b41eac3f6d9e520547702fd5b0c7ef6441625a4/edbo/bro.py#L658
    """

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> GPyTorchLikelihood:
        import torch
        from gpytorch.likelihoods import GaussianLikelihood

        effective_dims = sum(
            len(searchspace.get_comp_rep_parameter_indices(p.name))
            for p in searchspace.parameters
            if p._kind & _ParameterKind.REGULAR
        )

        switching_condition = _contains_encoding(
            searchspace.discrete, _EDBO_ENCODINGS
        ) and (effective_dims >= 50)

        # low D priors
        if effective_dims < 5:
            prior = GammaPrior(1.05, 0.5)
            initial_value = 0.1

        # DFT optimized priors
        elif switching_condition and effective_dims < 100:
            prior = GammaPrior(1.5, 0.1)
            initial_value = 5.0

        # Mordred optimized priors
        elif switching_condition:
            prior = GammaPrior(1.5, 0.1)
            initial_value = 5.0

        # OHE optimized priors
        else:
            prior = GammaPrior(1.5, 0.1)
            initial_value = 5.0

        likelihood = GaussianLikelihood(prior.to_gpytorch())
        likelihood.noise = torch.tensor([initial_value])
        return likelihood


EDBOFitCriterionFactory = _MLLForNonTLFitCriterionFactory()
"""A factory providing fitting criteria for the EDBO preset."""

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()

# Preset defaults
KERNEL_FACTORY = EDBOKernelFactory()
MEAN_FACTORY = EDBOMeanFactory()
LIKELIHOOD_FACTORY = EDBOLikelihoodFactory()
FIT_CRITERION_FACTORY = EDBOFitCriterionFactory
