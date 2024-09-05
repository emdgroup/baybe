# TODO: This file needs to be refactored.
"""Tests various configurations for a small number of iterations."""

import pytest
from pytest import param

from baybe.acquisition import qKG, qNIPV, qTS, qUCB
from baybe.acquisition.base import AcquisitionFunction
from baybe.exceptions import UnusedObjectWarning
from baybe.kernels.base import Kernel
from baybe.kernels.basic import (
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    PiecewisePolynomialKernel,
    PolynomialKernel,
    RBFKernel,
    RFFKernel,
    RQKernel,
)
from baybe.kernels.composite import AdditiveKernel, ProductKernel, ScaleKernel
from baybe.priors import (
    GammaPrior,
    HalfCauchyPrior,
    HalfNormalPrior,
    LogNormalPrior,
    NormalPrior,
    SmoothedBoxPrior,
)
from baybe.recommenders.meta.base import MetaRecommender
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.naive import NaiveHybridSpaceRecommender
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.recommenders.pure.bayesian.botorch import (
    BotorchRecommender,
)
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpaceType
from baybe.surrogates.bandit import BetaBernoulliMultiArmedBanditSurrogate
from baybe.surrogates.base import IndependentGaussianSurrogate, Surrogate
from baybe.surrogates.custom import CustomONNXSurrogate
from baybe.surrogates.gaussian_process.presets import (
    DefaultKernelFactory,
    EDBOKernelFactory,
)
from baybe.utils.basic import get_subclasses

from .conftest import run_iterations

########################################################################################
# Settings of the individual components to be tested
########################################################################################
valid_surrogate_models = [
    cls()
    for cls in get_subclasses(Surrogate)
    if (
        not issubclass(cls, CustomONNXSurrogate)
        and not issubclass(cls, BetaBernoulliMultiArmedBanditSurrogate)
    )
]
valid_initial_recommenders = [cls() for cls in get_subclasses(NonPredictiveRecommender)]
# TODO the TwoPhaseMetaRecommender below can be removed if the SeqGreedy recommender
#  allows no training data
valid_discrete_recommenders = [
    TwoPhaseMetaRecommender(recommender=cls())
    for cls in get_subclasses(PureRecommender)
    if cls.compatibility
    in [SearchSpaceType.DISCRETE, SearchSpaceType.HYBRID, SearchSpaceType.EITHER]
]
# TODO the TwoPhaseMetaRecommender below can be removed if the SeqGreedy recommender
#  allows no training data
valid_continuous_recommenders = [
    TwoPhaseMetaRecommender(recommender=cls())
    for cls in get_subclasses(PureRecommender)
    if cls.compatibility
    in [SearchSpaceType.CONTINUOUS, SearchSpaceType.HYBRID, SearchSpaceType.EITHER]
]

valid_active_learning_acqfs = [
    qNIPV(sampling_fraction=0.2, sampling_method="Random"),
    qNIPV(sampling_fraction=0.2, sampling_method="FPS"),
    qNIPV(sampling_fraction=1.0, sampling_method="FPS"),
    qNIPV(sampling_n_points=1, sampling_method="Random"),
    qNIPV(sampling_n_points=1, sampling_method="FPS"),
]
valid_mc_acqfs = [
    a() for a in get_subclasses(AcquisitionFunction) if a.is_mc
] + valid_active_learning_acqfs
valid_nonmc_acqfs = [a() for a in get_subclasses(AcquisitionFunction) if not a.is_mc]

# List of all hybrid recommenders with default attributes. Is extended with other lists
# of hybird recommenders like naive ones or recommenders not using default arguments
# TODO the TwoPhaseMetaRecommender below can be removed if the SeqGreedy recommender
#  allows no training data
valid_hybrid_recommenders = [
    TwoPhaseMetaRecommender(recommender=cls())
    for cls in get_subclasses(PureRecommender)
    if cls.compatibility == SearchSpaceType.HYBRID
]
# List of BotorchRecommenders with different sampling strategies.
sampling_strategies = [
    # Valid combinations
    (None, 0.0),
    (None, 1.0),
    ("FPS", 0.2),
    ("FPS", 0.5),
    ("Random", 0.2),
    ("Random", 0.5),
]
# TODO the TwoPhaseMetaRecommender below can be removed if the SeqGreedy recommender
#  allows no training data
valid_hybrid_sequential_greedy_recommenders = [
    TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(hybrid_sampler=sampler, sampling_percentage=per)
    )
    for sampler, per in sampling_strategies
]

valid_discrete_non_predictive_recommenders = [
    cls()
    for cls in get_subclasses(NonPredictiveRecommender)
    if cls.compatibility
    in [SearchSpaceType.DISCRETE, SearchSpaceType.EITHER, SearchSpaceType.HYBRID]
]
valid_discrete_bayesian_recommenders = [
    cls()
    for cls in get_subclasses(BayesianRecommender)
    if cls.compatibility
    in [SearchSpaceType.DISCRETE, SearchSpaceType.EITHER, SearchSpaceType.HYBRID]
]
# TODO the TwoPhaseMetaRecommender below can be removed if the SeqGreedy recommender
#  allows no training data
valid_naive_hybrid_recommenders = [
    TwoPhaseMetaRecommender(
        recommender=NaiveHybridSpaceRecommender(
            disc_recommender=disc, cont_recommender=BotorchRecommender()
        )
    )
    for disc in [
        *valid_discrete_non_predictive_recommenders,
        *valid_discrete_bayesian_recommenders,
    ]
]

valid_hybrid_recommenders.extend(valid_naive_hybrid_recommenders)
valid_hybrid_recommenders.extend(valid_hybrid_sequential_greedy_recommenders)

valid_meta_recommenders = get_subclasses(MetaRecommender)

valid_priors = [
    GammaPrior(3, 1),
    HalfCauchyPrior(2),
    HalfNormalPrior(2),
    LogNormalPrior(1, 2),
    NormalPrior(1, 2),
    SmoothedBoxPrior(0, 3, 0.1),
]

valid_base_kernels: list[Kernel] = [
    cls(**arg_dict)
    for prior in valid_priors
    for cls, arg_dict in [
        (MaternKernel, {"lengthscale_prior": prior}),
        (LinearKernel, {"variance_prior": prior}),
        (PeriodicKernel, {"period_length_prior": prior}),
        (PeriodicKernel, {"lengthscale_prior": prior}),
        (PiecewisePolynomialKernel, {"lengthscale_prior": prior}),
        (PolynomialKernel, {"offset_prior": prior, "power": 2}),
        (RBFKernel, {"lengthscale_prior": prior}),
        (RQKernel, {"lengthscale_prior": prior}),
        (RFFKernel, {"lengthscale_prior": prior, "num_samples": 5}),
    ]
]

valid_scale_kernels = [
    ScaleKernel(base_kernel=base_kernel, outputscale_prior=HalfCauchyPrior(scale=1))
    for base_kernel in valid_base_kernels
]

valid_composite_kernels = [
    AdditiveKernel([MaternKernel(1.5), MaternKernel(2.5)]),
    AdditiveKernel([PolynomialKernel(1), PolynomialKernel(2), PolynomialKernel(3)]),
    AdditiveKernel([RBFKernel(), RQKernel(), PolynomialKernel(1)]),
    ProductKernel([MaternKernel(1.5), MaternKernel(2.5)]),
    ProductKernel([RBFKernel(), RQKernel(), PolynomialKernel(1)]),
    ProductKernel([PolynomialKernel(1), PolynomialKernel(2), PolynomialKernel(3)]),
    AdditiveKernel(
        [
            ProductKernel([MaternKernel(1.5), MaternKernel(2.5)]),
            AdditiveKernel([MaternKernel(1.5), MaternKernel(2.5)]),
        ]
    ),
]

valid_kernels = valid_base_kernels + valid_scale_kernels + valid_composite_kernels


valid_kernel_factories = [
    param(DefaultKernelFactory(), id="Default"),
    param(EDBOKernelFactory(), id="EDBO"),
]

test_targets = [
    ["Target_max"],
    ["Target_min"],
    ["Target_match_bell"],
    ["Target_match_triangular"],
    ["Target_max_bounded", "Target_min_bounded"],
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "acqf", valid_mc_acqfs, ids=[a.abbreviation for a in valid_mc_acqfs]
)
@pytest.mark.parametrize("n_iterations", [3], ids=["i3"])
def test_mc_acqfs(campaign, n_iterations, batch_size, acqf):
    if isinstance(acqf, qKG):
        pytest.skip(f"{acqf.__class__.__name__} only works with continuous spaces.")
    if isinstance(acqf, qTS) and batch_size > 1:
        pytest.skip(f"{acqf.__class__.__name__} only works with batch size 1.")

    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize(
    "acqf", valid_nonmc_acqfs, ids=[a.abbreviation for a in valid_nonmc_acqfs]
)
@pytest.mark.parametrize("n_iterations", [3], ids=["i3"])
@pytest.mark.parametrize("batch_size", [1], ids=["b1"])
def test_nonmc_acqfs(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel", valid_kernels, ids=[c.__class__ for c in valid_kernels]
)
@pytest.mark.parametrize("n_iterations", [3], ids=["i3"])
def test_kernels(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.parametrize("kernel", valid_kernel_factories)
@pytest.mark.parametrize("n_iterations", [3], ids=["i3"])
def test_kernel_factories(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize(
    "surrogate_model",
    valid_surrogate_models,
    ids=[c.__class__ for c in valid_surrogate_models],
)
def test_surrogate_models(campaign, n_iterations, batch_size, surrogate_model):
    if batch_size > 1 and isinstance(surrogate_model, IndependentGaussianSurrogate):
        pytest.skip("Batch recommendation is not supported.")
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_initial_recommenders)
def test_initial_recommenders(campaign, n_iterations, batch_size):
    with pytest.warns(UnusedObjectWarning):
        run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize("target_names", test_targets)
def test_targets(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_discrete_recommenders)
def test_recommenders_discrete(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_continuous_recommenders)
@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
def test_recommenders_continuous(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_hybrid_recommenders)
@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1", "Conti_finite2"]],
)
def test_recommenders_hybrid(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.parametrize("recommender", valid_meta_recommenders, indirect=True)
def test_meta_recommenders(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.parametrize("acqf", [qTS(), qUCB()])
@pytest.mark.parametrize("surrogate_model", [BetaBernoulliMultiArmedBanditSurrogate()])
@pytest.mark.parametrize(
    "parameter_names",
    [
        ["Categorical_1"],
        ["Switch_1"],
        ["Switch_2"],
        ["Frame_A"],
        ["Frame_B"],
    ],
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("target_names", [["Target_binary"]])
@pytest.mark.parametrize("allow_repeated_recommendations", [True])
@pytest.mark.parametrize("allow_recommending_already_measured", [True])
def test_multi_armed_bandit(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
