# TODO: This file needs to be refactored.
"""Tests various configurations for a small number of iterations."""

from contextlib import nullcontext

import pytest
from botorch.exceptions import UnsupportedError
from pytest import param

from baybe.acquisition import qKG, qNIPV, qTS, qUCB
from baybe.acquisition.base import AcquisitionFunction
from baybe.exceptions import (
    InvalidSurrogateModelError,
    OptionalImportError,
    UnusedObjectWarning,
)
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
from baybe.objectives.pareto import ParetoObjective
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
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import get_subclasses
from tests.conftest import run_iterations

########################################################################################
# Settings of the individual components to be tested
########################################################################################
valid_surrogate_models = []
for cls in get_subclasses(Surrogate):
    if issubclass(cls, CustomONNXSurrogate) or issubclass(
        cls, BetaBernoulliMultiArmedBanditSurrogate
    ):
        continue
    try:
        p = param(cls(), id=cls.__name__)
    except OptionalImportError:
        p = param(cls, marks=pytest.mark.skip(reason="missing optional dependency"))
    valid_surrogate_models.append(p)

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

acqfs_extra = [  # Additionally tested acqfs with extra configurations
    qNIPV(sampling_fraction=0.2, sampling_method="Random"),
    qNIPV(sampling_fraction=0.2, sampling_method="FPS"),
    qNIPV(sampling_fraction=1.0, sampling_method="FPS"),
    qNIPV(sampling_n_points=1, sampling_method="Random"),
    qNIPV(sampling_n_points=1, sampling_method="FPS"),
]
acqfs_batching = [
    a() for a in get_subclasses(AcquisitionFunction) if a.supports_batching
] + acqfs_extra
acqfs_non_batching = [
    a() for a in get_subclasses(AcquisitionFunction) if not a.supports_batching
]
acqfs_single_output_batching = [
    a for a in acqfs_batching if not a.supports_multi_output
]
acqfs_multi_output_batching = [a for a in acqfs_batching if a.supports_multi_output]

# List of all hybrid recommenders with default attributes. Is extended with other lists
# of hybrid recommenders like naive ones or recommenders not using default arguments
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
    HalfCauchyPrior(0.5),
    HalfNormalPrior(0.5),
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
    param(["Target_max"], id="Tmax"),
    param(["Target_min"], id="Tmin"),
    param(["Target_match_bell"], id="Tmatch_bell"),
    param(["Target_match_triangular"], id="Tmatch_triang"),
    param(["Target_max_bounded", "Target_min_bounded"], id="Tmax_bounded_Tmin_bounded"),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "acqf",
    acqfs_single_output_batching,
    ids=[a.abbreviation for a in acqfs_single_output_batching],
)
@pytest.mark.parametrize("n_iterations", [3], ids=["i3"])
def test_single_output_batching_acqfs(campaign, n_iterations, batch_size, acqf):
    context = nullcontext()
    if campaign.searchspace.type not in [
        SearchSpaceType.CONTINUOUS,
        SearchSpaceType.HYBRID,
    ] and isinstance(acqf, qKG):
        # qKG does not work with purely discrete spaces
        context = pytest.raises(UnsupportedError)

    with context:
        run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize(
    "objective",
    [ParetoObjective([NumericalTarget("t1"), NumericalTarget("t2", minimize=True)])],
)
@pytest.mark.parametrize(
    "acqf",
    acqfs_multi_output_batching,
    ids=[a.abbreviation for a in acqfs_multi_output_batching],
)
@pytest.mark.parametrize("n_iterations", [3], ids=["i3"])
def test_multi_output_batching_acqfs(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize(
    "acqf", acqfs_non_batching, ids=[a.abbreviation for a in acqfs_non_batching]
)
@pytest.mark.parametrize("n_iterations", [3], ids=["i3"])
@pytest.mark.parametrize("batch_size", [1], ids=["b1"])
def test_non_batching_acqfs(campaign, n_iterations, batch_size):
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
@pytest.mark.parametrize("surrogate_model", valid_surrogate_models)
def test_surrogate_models(campaign, n_iterations, batch_size, surrogate_model):
    context = nullcontext()
    if batch_size > 1 and isinstance(surrogate_model, IndependentGaussianSurrogate):
        context = pytest.raises(InvalidSurrogateModelError)

    with context:
        run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize(
    "recommender",
    valid_initial_recommenders,
    ids=[c.__class__ for c in valid_initial_recommenders],
)
def test_initial_recommenders(campaign, n_iterations, batch_size):
    with pytest.warns(UnusedObjectWarning):
        try:
            run_iterations(campaign, n_iterations, batch_size)
        except OptionalImportError as e:
            pytest.skip(f"Optional dependency '{e.name}' not installed.")


@pytest.mark.slow
@pytest.mark.parametrize("target_names", test_targets)
def test_targets(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize(
    "recommender",
    valid_discrete_recommenders,
    ids=[c.__class__ for c in valid_discrete_recommenders],
)
def test_recommenders_discrete(campaign, n_iterations, batch_size):
    try:
        run_iterations(campaign, n_iterations, batch_size)
    except OptionalImportError as e:
        pytest.skip(f"Optional dependency '{e.name}' not installed.")


@pytest.mark.slow
@pytest.mark.parametrize(
    "recommender",
    valid_continuous_recommenders,
    ids=[c.__class__ for c in valid_continuous_recommenders],
)
@pytest.mark.parametrize(
    "parameter_names", [["Conti_finite1", "Conti_finite2"]], ids=["conti_params"]
)
def test_recommenders_continuous(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize(
    "recommender",
    valid_hybrid_recommenders,
    ids=[c.__class__ for c in valid_hybrid_recommenders],
)
@pytest.mark.parametrize(
    "parameter_names",
    [
        [
            "Categorical_1",
            "Some_Setting",
            "Num_disc_1",
            "Conti_finite1",
            "Conti_finite2",
        ],
        [
            "Categorical_1_subset",
            "Some_Setting",
            "Num_disc_1",
            "Conti_finite1",
            "Conti_finite2",
        ],
    ],
    ids=["hybrid_params", "hybrid_params_with_active_values"],
)
def test_recommenders_hybrid(campaign, n_iterations, batch_size):
    try:
        run_iterations(campaign, n_iterations, batch_size)
    except OptionalImportError as e:
        pytest.skip(f"Optional dependency '{e.name}' not installed.")


@pytest.mark.parametrize(
    "recommender",
    valid_meta_recommenders,
    ids=[c.__class__ for c in valid_meta_recommenders],
    indirect=True,
)
def test_meta_recommenders(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.parametrize(
    "acqf", [qTS(), qUCB()], ids=[qTS.abbreviation, qUCB.abbreviation]
)
@pytest.mark.parametrize(
    "surrogate_model",
    [BetaBernoulliMultiArmedBanditSurrogate()],
    ids=["bernoulli_bandit_surrogate"],
)
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
@pytest.mark.parametrize("target_names", [["Target_binary"]], ids=["binary_target"])
@pytest.mark.parametrize("batch_size", [1], ids=["b1"])
def test_multi_armed_bandit(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
