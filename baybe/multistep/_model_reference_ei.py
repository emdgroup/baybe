"""Model-reference expected improvement for DAG-valued objectives."""

from __future__ import annotations

import gc
import math
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn

from attrs import Attribute, define, field
from attrs.validators import and_, ge, gt, instance_of, not_
from typing_extensions import override

from baybe.acquisition.base import AcquisitionFunction
from baybe.exceptions import IncompatibilityError
from baybe.objectives.base import Objective
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import SurrogateProtocol
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import classproperty

if TYPE_CHECKING:
    import pandas as pd
    from botorch.acquisition.acquisition import (
        AcquisitionFunction as BoAcquisitionFunction,
    )
    from torch import Tensor

    from .dag import MultiStepDAG
    from .data import _PendingAction
    from .model import MultiStepModel


_Reference = Literal["max_mean", "mean", "thompson_max"]
_INTEGER_VALIDATOR = and_(instance_of(int), not_(instance_of(bool)))
_MAX_DISCRETE_CONFIGURATIONS = 4096


def _draw_model_reference_seed() -> int:
    """Draw one local seed from Torch's current default random generator."""
    import torch

    return int(torch.randint(0, 2**32, (), dtype=torch.int64).item())


def _transform_final_samples(dag: MultiStepDAG, samples: Tensor) -> Tensor:
    """Apply objective ordering and its complete transformation to raw DAG samples.

    Args:
        dag: The DAG defining graph-output and objective order.
        samples: Raw network outputs with shape ``(samples, batch..., q, outputs)``.

    Returns:
        Scalar final utility samples with shape ``(samples, batch..., q)``.

    Raises:
        IncompatibilityError: If output dimensions, q, or transformed values are
            invalid.
    """
    import torch

    if (
        samples.ndim < 3
        or samples.shape[-2] != 1
        or samples.shape[-1] != len(dag.graph_output_names)
        or not torch.isfinite(samples).all()
    ):
        raise IncompatibilityError(
            "Final network samples must be finite, have q=1, and match the graph "
            "outputs."
        )

    objective_samples = samples.index_select(
        -1,
        torch.tensor(
            dag.objective_output_indices, dtype=torch.long, device=samples.device
        ),
    )
    transformation = dag.objective._full_transformation

    def move_objective(objective: Any) -> None:
        """Move wrapped BoTorch objectives, including unregistered children."""
        if callable(getattr(objective, "to", None)):
            objective.to(device=samples.device, dtype=samples.dtype)
        for child in getattr(objective, "objectives", ()):
            move_objective(child)

    move_objective(transformation)
    utilities = transformation(objective_samples, None).to(samples)
    if utilities.shape != samples.shape[:-1] or not torch.isfinite(utilities).all():
        raise IncompatibilityError(
            "The final objective must produce one finite scalar utility per network "
            "sample."
        )
    return utilities


def _compute_sample_expected_improvement(samples: Tensor, reference: float) -> Tensor:
    """Estimate EI as the sample mean of each sample's positive utility excess.

    Args:
        samples: Scalar utility draws shaped ``(samples, batch..., q=1)``.
        reference: The scalar model-based utility reference.

    Returns:
        One nonnegative expected improvement estimate per candidate batch.

    Raises:
        IncompatibilityError: If shapes or numerical values are invalid.
    """
    import torch

    if (
        samples.ndim < 3
        or samples.shape[0] < 1
        or samples.shape[-1] != 1
        or not math.isfinite(reference)
        or not torch.isfinite(samples).all()
    ):
        raise IncompatibilityError(
            "Sample EI requires finite scalar samples, finite reference, and q=1."
        )
    return (samples - reference).clamp_min(0.0).mean(dim=0).squeeze(-1)


@define(frozen=True, slots=False)
class ModelReferenceExpectedImprovement(AcquisitionFunction):
    """Score a next-node action by sample EI against a global DAG-utility reference.

    Continuous reference maxima use the configured multistart optimizer and are
    best-found values, not certificates of global optimality. Thompson references
    use finite random Fourier-feature Matheron paths and are approximate.
    """

    abbreviation: ClassVar[str] = "MREI"
    """The short name used for polymorphic type resolution."""

    supports_multi_output: ClassVar[bool] = False
    """Whether uns scalarized Pareto outputs are accepted."""

    gain_scale: ClassVar[str] = "linear"
    """The gain is nonnegative expected improvement in utility units."""

    supports_signed_gains: ClassVar[bool] = False
    """Whether finite negative gains are permitted."""

    @classproperty
    def is_analytic(cls) -> bool:
        """Declare that network Monte Carlo, not an analytic formula, is used."""
        return False

    @classproperty
    def supports_batching(cls) -> bool:
        """Declare that the campaign accepts one recommendation at a time."""
        return False

    reference: _Reference = field(default="max_mean", validator=instance_of(str))
    """The fixed model-based utility reference definition."""

    n_network_samples: int = field(default=256, validator=[_INTEGER_VALIDATOR, gt(0)])
    """The number of independent network posterior samples for mean and EI."""

    n_reference_points: int = field(default=1024, validator=[_INTEGER_VALIDATOR, gt(0)])
    """The number of uniform global configurations for the ``mean`` reference."""

    n_path_features: int = field(default=2048, validator=[_INTEGER_VALIDATOR, gt(0)])
    """The number of random Fourier features in each Thompson GP path."""

    sampling_seed: int = field(default=0, validator=[_INTEGER_VALIDATOR, ge(0)])
    """The nonnegative base seed separating the reference and EI random streams."""

    @reference.validator
    def _validate_reference(self, _attribute: Attribute, value: str) -> None:
        """Validate the supported model-reference definitions."""
        if value not in ("max_mean", "mean", "thompson_max"):
            raise ValueError(
                "'reference' must be 'max_mean', 'mean', or 'thompson_max'."
            )

    @classproperty
    def supports_pending_experiments(cls) -> bool:
        """Declare support for the shared multi-step pending-fantasy branches."""
        return True

    def validate_context(self, dag: MultiStepDAG, data: Any, batch_size: int) -> None:
        """Validate DAG/objective and discrete-size prerequisites before fitting.

        Args:
            dag: The multi-step DAG.
            data: The real node-measurement store (not used by this reference policy).
            batch_size: Requested recommendation count.

        Raises:
            IncompatibilityError: If the objective, parameter space, or batch size is
                unsupported.
        """
        if batch_size != 1:
            raise IncompatibilityError(
                f"{self.__class__.__name__} supports only batch_size=1."
            )
        objective = dag.objective
        if objective.is_multi_output or not all(
            isinstance(target, NumericalTarget) for target in objective.targets
        ):
            raise IncompatibilityError(
                f"{self.__class__.__name__} requires a scalar numerical objective."
            )
        unsupported = sorted(
            parameter.name
            for parameter in dag.parameter_by_name.values()
            if not isinstance(
                parameter, (DiscreteParameter, NumericalContinuousParameter)
            )
        )
        if unsupported:
            raise IncompatibilityError(
                "Model-reference EI supports only discrete and numerical continuous "
                f"parameters; unsupported parameters: {unsupported}."
            )
        n_discrete = math.prod(
            len(parameter.values)
            for parameter in dag.parameter_by_name.values()
            if isinstance(parameter, DiscreteParameter)
        )
        if n_discrete > _MAX_DISCRETE_CONFIGURATIONS:
            raise IncompatibilityError(
                f"The global discrete space has {n_discrete} combinations, exceeding "
                f"the model-reference EI limit of {_MAX_DISCRETE_CONFIGURATIONS}."
            )
        if any(node.searchspace.is_constrained for node in dag.nodes):
            raise IncompatibilityError(
                "Model-reference EI does not support constrained node search spaces."
            )

    def validate_model(self, model: MultiStepModel) -> None:
        """Validate persistent GP-path support for a Thompson reference.

        Other reference modes rely on native network posterior samples and do not
        require this Thompson-specific model support.

        Args:
            model: The fitted DAG network model.

        Raises:
            IncompatibilityError: If a model, likelihood, kernel, or batch shape is
                unsupported for Matheron paths.
        """
        if self.reference == "thompson_max":
            _validate_path_models(model)

    def make_botorch_branches(
        self,
        model: MultiStepModel,
        pending: tuple[_PendingAction, ...],
        *,
        call_seed: int | None = None,
    ) -> tuple[BoAcquisitionFunction, ...]:
        """Build pending-conditioned sample-EI branches using one real reference.

        Args:
            model: The fitted real DAG model.
            pending: Actions already reserved in the campaign.
            call_seed: The local seed for this numeric recommendation. If omitted,
                one seed is drawn from Torch's current random generator.

        Returns:
            One BoTorch sample-EI function per pending fantasy branch.

        Raises:
            IncompatibilityError: If the call seed or fitted path models are
                invalid.
        """
        from botorch.acquisition.acquisition import (
            AcquisitionFunction as BoAcquisitionFunction,
        )
        from botorch.utils.transforms import t_batch_mode_transform

        if call_seed is None:
            call_seed = _draw_model_reference_seed()
        if (
            not isinstance(call_seed, int)
            or isinstance(call_seed, bool)
            or call_seed < 0
        ):
            raise IncompatibilityError(
                "The model-reference recommendation seed must be a nonnegative integer."
            )
        if self.reference == "thompson_max":
            _validate_path_models(model)

        reference_value = _compute_model_reference(self, model, call_seed=call_seed)
        from .acquisition import _PENDING_FANTASIES
        from .model import _derive_seed

        pending_models = model.condition_on_pending(
            pending,
            num_fantasies=_PENDING_FANTASIES,
            sampling_seed=_derive_seed(
                self.sampling_seed + call_seed,
                "model-reference-ei",
                "pending-fantasies",
            ),
        )
        if not pending_models:
            raise IncompatibilityError(
                "Pending conditioning returned no model-reference EI branches."
            )

        branches = []
        n_network_samples = self.n_network_samples

        def make_branch(branch_model: MultiStepModel, ei_seed: int) -> Any:
            """Bind one conditioned model and one independent EI random stream."""

            @define(eq=False, repr=False, slots=False, init=False)
            class _SampleExpectedImprovement(BoAcquisitionFunction):
                """Evaluate network-sample EI against one shared real reference."""

                @override
                @t_batch_mode_transform(expected_q=1)
                def forward(self, X: Tensor) -> Tensor:
                    samples = _draw_final_utility_samples(
                        branch_model,
                        X,
                        n_samples=n_network_samples,
                        sampling_seed=ei_seed,
                    )
                    return _compute_sample_expected_improvement(
                        samples, reference_value
                    )

            return _SampleExpectedImprovement(model=branch_model)

        for branch_index, branch_model in enumerate(pending_models):
            ei_seed = _derive_seed(
                self.sampling_seed + call_seed,
                "model-reference-ei",
                f"independent-ei-samples-{branch_index}",
            )
            branches.append(make_branch(branch_model, ei_seed))
        return tuple(branches)

    @override
    def to_botorch(
        self,
        surrogate: SurrogateProtocol,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
    ) -> NoReturn:
        """Reject using DAG model-reference EI as an ordinary one-model acquisition.

        Args:
            surrogate: The surrogate supplied by the ordinary recommender.
            searchspace: The ordinary search space.
            objective: The ordinary objective.
            measurements: The ordinary measurements.
            pending_experiments: Optional ordinary pending experiments.

        Raises:
            IncompatibilityError: Always; use this policy in a multi-step campaign.
        """
        raise IncompatibilityError(
            f"'{self.__class__.__name__}' is a multi-step acquisition. Use "
            "'MultiStepCampaign.recommend()' instead."
        )


def _validate_path_models(model: MultiStepModel) -> None:
    """Validate supported unbatched scalar ExactGP models.

    The paths require Gaussian likelihoods and supported kernels.

    Args:
        model: The fitted DAG network whose node models are used for paths.

    Raises:
        IncompatibilityError: If a node output uses an unsupported GP, likelihood,
            kernel, or batch shape.
    """
    import torch
    from botorch.models.model_list_gp_regression import ModelListGP
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
    from gpytorch.likelihoods import _GaussianLikelihoodBase
    from gpytorch.models import ExactGP

    def is_supported_kernel(kernel: Any) -> bool:
        """Check the finite-feature paths supported by BoTorch."""
        if isinstance(kernel, (RBFKernel, MaternKernel)):
            return True
        return isinstance(kernel, ScaleKernel) and is_supported_kernel(
            kernel.base_kernel
        )

    for node_name in model.dag.topological_order:
        node = model.dag.node_by_name[node_name]
        node_model = model.get_node_model(node_name)
        components = (
            tuple(node_model.models)
            if isinstance(node_model, ModelListGP)
            else (node_model,)
        )
        if len(components) != len(node.output_names):
            raise IncompatibilityError(
                f"Thompson paths for node '{node_name}' require one scalar GP per "
                f"output; found {len(components)} models for {len(node.output_names)} "
                "outputs."
            )
        for output_name, component in zip(node.output_names, components):
            if not isinstance(component, ExactGP):
                raise IncompatibilityError(
                    f"Thompson paths require an unbatched scalar ExactGP for node "
                    f"'{node_name}', output '{output_name}'; found "
                    f"'{type(component).__name__}'."
                )
            if component.batch_shape != torch.Size():
                raise IncompatibilityError(
                    f"Thompson paths require an unbatched scalar ExactGP for node "
                    f"'{node_name}', output '{output_name}'; found batch shape "
                    f"{tuple(component.batch_shape)}."
                )
            likelihood = getattr(component, "likelihood", None)
            if not isinstance(likelihood, _GaussianLikelihoodBase):
                raise IncompatibilityError(
                    f"Thompson paths require a Gaussian likelihood for node "
                    f"'{node_name}', output '{output_name}'; found "
                    f"'{type(likelihood).__name__}'."
                )
            if not is_supported_kernel(component.covar_module):
                raise IncompatibilityError(
                    f"Thompson paths support only RBF, Matérn, and nested Scale "
                    f"kernels for node '{node_name}', output '{output_name}'; found "
                    f"'{type(component.covar_module).__name__}'."
                )


def _derive_acquisition_seed(
    acquisition: ModelReferenceExpectedImprovement,
    call_seed: int,
    purpose: str,
    *,
    node_name: str = "model-reference-ei",
) -> int:
    """Derive a stable, isolated random stream for one acquisition purpose."""
    from .model import _derive_seed

    return _derive_seed(acquisition.sampling_seed + call_seed, node_name, purpose)


def _draw_final_utility_samples(
    model: MultiStepModel,
    inputs: Tensor,
    *,
    n_samples: int,
    sampling_seed: int,
) -> Tensor:
    """Draw native DAG posterior paths and transform each to scalar final utility."""
    import torch

    raw_samples = model.draw_network_samples(
        inputs, n_samples, sampling_seed=sampling_seed
    )
    if (
        raw_samples.ndim != inputs.ndim + 1
        or raw_samples.shape[1:-1] != inputs.shape[:-1]
    ):
        raise IncompatibilityError(
            "The network sampler returned shape "
            f"{tuple(raw_samples.shape)} for inputs {tuple(inputs.shape)}."
        )
    utilities = _transform_final_samples(model.dag, raw_samples)
    if not torch.isfinite(utilities).all():
        raise IncompatibilityError("The transformed network utilities are not finite.")
    return utilities


def _uniform_global_configurations(
    dag: MultiStepDAG,
    n_points: int,
    reference: Tensor,
    *,
    seed: int,
) -> Tensor:
    """Sample global configurations uniformly without changing Torch's RNG state.

    Discrete choices and continuous coordinates follow their declared distributions.
    """
    import pandas as pd
    import torch

    searchspace = dag.full_searchspace
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    discrete = searchspace.discrete
    if discrete.is_empty:
        configurations = pd.DataFrame(index=range(n_points))
    else:
        choices = discrete.exp_rep
        indices = torch.randint(len(choices), (n_points,), generator=generator).tolist()
        configurations = choices.iloc[indices].reset_index(drop=True).copy()

    for parameter in searchspace.continuous.parameters:
        lower = float(parameter.bounds.lower)
        upper = float(parameter.bounds.upper)
        uniform = torch.rand(n_points, generator=generator, dtype=torch.double)
        configurations[parameter.name] = lower + (upper - lower) * uniform.numpy()

    comp_rep = searchspace.transform(configurations)
    return torch.as_tensor(comp_rep.to_numpy(copy=True)).to(reference).unsqueeze(-2)


def _make_reference_acquisition(
    model: MultiStepModel,
    score: Any,
) -> BoAcquisitionFunction:
    """Wrap a differentiable scalar reference score for BoTorch optimizers."""
    from botorch.acquisition.acquisition import (
        AcquisitionFunction as BoAcquisitionFunction,
    )
    from botorch.utils.transforms import t_batch_mode_transform

    @define(eq=False, repr=False, slots=False, init=False)
    class _ReferenceAcquisition(BoAcquisitionFunction):
        """Evaluate one model-based scalar score per q=1 candidate."""

        @override
        @t_batch_mode_transform(expected_q=1)
        def forward(self, X: Tensor) -> Tensor:
            return score(X)

    return _ReferenceAcquisition(model=model)


def _maximize_global_reference(
    model: MultiStepModel,
    score: Any,
) -> float:
    """Maximize a fixed global reference score with stock BoTorch optimizers."""
    import torch
    from botorch.optim import optimize_acqf_discrete, optimize_acqf_mixed

    from .acquisition import _N_RAW_SAMPLES, _N_RESTARTS, _OPTIMIZER_SEED

    searchspace = model.dag.full_searchspace
    reference = next(model.parameters())
    acquisition = _make_reference_acquisition(model, score)
    if searchspace.continuous.is_empty:
        if searchspace.discrete.is_empty:
            choices = reference.new_zeros((1, len(searchspace.comp_rep_columns)))
        else:
            discrete_frame = searchspace.discrete.exp_rep
            choices_frame = searchspace.discrete.transform(discrete_frame)
            choices = torch.as_tensor(choices_frame.to_numpy(copy=True)).to(reference)
        candidate, score_value = optimize_acqf_discrete(
            acq_function=acquisition, q=1, choices=choices
        )
    else:
        bounds = torch.as_tensor(searchspace.comp_rep_bounds.to_numpy(copy=True)).to(
            reference
        )
        column_index = {
            name: index for index, name in enumerate(searchspace.comp_rep_columns)
        }
        if searchspace.discrete.is_empty:
            fixed_features_list = [{}]
        else:
            discrete_frame = searchspace.discrete.exp_rep
            discrete_comp = searchspace.discrete.transform(discrete_frame)
            fixed_features_list = [
                {column_index[column]: float(value) for column, value in row.items()}
                for row in discrete_comp.to_dict(orient="records")
            ]
        with torch.random.fork_rng():
            torch.manual_seed(_OPTIMIZER_SEED)
            _candidate, score_value = optimize_acqf_mixed(
                acq_function=acquisition,
                bounds=bounds,
                q=1,
                fixed_features_list=fixed_features_list,
                num_restarts=_N_RESTARTS,
                raw_samples=_N_RAW_SAMPLES,
                options={"seed": _OPTIMIZER_SEED},
            )
    if not torch.isfinite(score_value).all():
        raise IncompatibilityError(
            "Global reference optimization returned a nonfinite value."
        )
    return float(score_value.max().detach().cpu())


def _compute_model_reference(
    acquisition: ModelReferenceExpectedImprovement,
    model: MultiStepModel,
    *,
    call_seed: int,
) -> float:
    """Compute one global reference from the real model before pending fantasies."""
    import torch

    if acquisition.reference == "mean":
        reference = next(model.parameters())
        points = _uniform_global_configurations(
            model.dag,
            acquisition.n_reference_points,
            reference,
            seed=_derive_acquisition_seed(acquisition, call_seed, "mean-points"),
        )
        utilities = _draw_final_utility_samples(
            model,
            points,
            n_samples=acquisition.n_network_samples,
            sampling_seed=_derive_acquisition_seed(
                acquisition, call_seed, "mean-reference-network-samples"
            ),
        )
        value = utilities.mean(dim=0).mean()
    elif acquisition.reference == "max_mean":
        seed = _derive_acquisition_seed(
            acquisition, call_seed, "max-mean-reference-network-samples"
        )

        def mean_score(inputs: Tensor) -> Tensor:
            """Estimate final expected utility with fixed network draws."""
            return (
                _draw_final_utility_samples(
                    model,
                    inputs,
                    n_samples=acquisition.n_network_samples,
                    sampling_seed=seed,
                )
                .mean(dim=0)
                .squeeze(-1)
            )

        value = _maximize_global_reference(model, mean_score)
        return value
    else:
        draw_path = _make_network_thompson_draw(
            acquisition,
            model,
            call_seed=call_seed,
        )
        value = _maximize_global_reference(
            model, lambda inputs: draw_path(inputs).squeeze(-1)
        )
        return value

    if not torch.isfinite(value):
        raise IncompatibilityError("The global model reference is not finite.")
    return float(value.detach().cpu())


def _make_network_thompson_draw(
    acquisition: ModelReferenceExpectedImprovement,
    model: MultiStepModel,
    *,
    call_seed: int,
) -> Any:
    """Create one persistent, coherent Matheron path through every DAG node."""
    import torch
    from botorch.models.deterministic import GenericDeterministicModel
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.sampling.pathwise.features import KernelFeatureMap, gen_kernel_features
    from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
    from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths
    from botorch.sampling.pathwise.utils import (
        ChainedTransform,
        SineCosineTransform,
        TensorTransform,
    )

    from .model import _snapshot_model

    class RandomPhaseTransform(TensorTransform):
        """Map arbitrary Fourier features to unbiased random-phase RFFs."""

        def __init__(self, phase: Tensor, scale: Tensor) -> None:
            super().__init__()
            self.register_buffer("phase", phase)
            self.register_buffer("scale", scale)

        @override
        def forward(self, values: Tensor, **_kwargs: Any) -> Tensor:
            return self.scale * torch.cos(values + self.phase)

    def make_feature_map(kernel: Any, num_inputs: int, num_outputs: int) -> Any:
        """Use paired features when even and random-phase features when odd."""
        if num_outputs % 2 == 0:
            return gen_kernel_features(kernel, num_inputs, num_outputs)

        paired = gen_kernel_features(kernel, num_inputs, 2 * num_outputs)
        phase = (
            2
            * math.pi
            * torch.rand(num_outputs, device=kernel.device, dtype=kernel.dtype)
        )
        scale = torch.as_tensor(
            math.sqrt(2.0 / num_outputs), device=kernel.device, dtype=kernel.dtype
        )

        def replace_sine_cosine(transform: Any) -> Any:
            if isinstance(transform, ChainedTransform):
                return ChainedTransform(
                    *(replace_sine_cosine(child) for child in transform.transforms)
                )
            if isinstance(transform, SineCosineTransform):
                return RandomPhaseTransform(phase, scale)
            return transform

        return KernelFeatureMap(
            kernel=kernel,
            weight=paired.weight,
            input_transform=paired.input_transform,
            output_transform=replace_sine_cosine(paired.output_transform),
        )

    path_by_node: dict[str, tuple[Any, ...]] = {}
    for node_name in model.dag.topological_order:
        node_model = model.get_node_model(node_name)
        components = (
            tuple(node_model.models)
            if isinstance(node_model, ModelListGP)
            else (node_model,)
        )
        paths = []
        for output_index, component in enumerate(components):
            seed = _derive_acquisition_seed(
                acquisition,
                call_seed,
                f"thompson-path-{output_index}",
                node_name=node_name,
            )
            device = next(component.parameters()).device
            devices = (
                [
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                ]
                if device.type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                prior_sampler = partial(
                    draw_kernel_feature_paths,
                    num_features=acquisition.n_path_features,
                    map_generator=make_feature_map,
                )
                path = (
                    draw_kernel_feature_paths(
                        component,
                        sample_shape=torch.Size([1]),
                        num_features=acquisition.n_path_features,
                        map_generator=make_feature_map,
                    )
                    if component.train_targets.numel() == 0
                    else draw_matheron_paths(
                        component,
                        sample_shape=torch.Size([1]),
                        prior_sampler=prior_sampler,
                    )
                )
            paths.append(path)
        path_by_node[node_name] = tuple(paths)

    dag = model.dag
    snapshot = _snapshot_model(model)
    for node_name, paths in path_by_node.items():
        output_names = dag.node_by_name[node_name].output_names
        if len(paths) != len(output_names):
            raise IncompatibilityError(
                f"Thompson paths for node '{node_name}' do not match its declared "
                f"outputs {output_names}."
            )

        def make_node_function(
            paths: tuple[Any, ...], output_names: tuple[str, ...], node_name: str
        ) -> Any:
            """Wrap the fixed output paths as one deterministic node model."""

            def evaluate(inputs: Tensor) -> Tensor:
                output_values = []
                for output_name, path in zip(output_names, paths, strict=True):
                    path_values = path(inputs)
                    expected_shape = inputs.shape[:-1]
                    if tuple(path_values.shape) != expected_shape:
                        raise IncompatibilityError(
                            f"Thompson path for node '{node_name}', output "
                            f"'{output_name}' returned shape "
                            f"{tuple(path_values.shape)}, "
                            f"expected {expected_shape}."
                        )
                    if not torch.isfinite(path_values).all():
                        raise IncompatibilityError(
                            f"Thompson path for node '{node_name}', output "
                            f"'{output_name}' returned nonfinite values."
                        )
                    output_values.append(path_values.unsqueeze(-1))
                return torch.cat(output_values, dim=-1)

            return evaluate

        snapshot._node_models[node_name] = GenericDeterministicModel(
            make_node_function(paths, output_names, node_name),
            num_outputs=len(paths),
        )

    def draw(inputs: Tensor) -> Tensor:
        """Evaluate the fixed paths through the snapshot's existing DAG sampler."""
        raw_samples = snapshot.draw_network_samples(inputs, 1, sampling_seed=0)
        return _transform_final_samples(dag, raw_samples)[0]

    return draw


__all__ = ["ModelReferenceExpectedImprovement", "MREI"]

MREI = ModelReferenceExpectedImprovement


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
