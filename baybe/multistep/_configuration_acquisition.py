"""Monte Carlo expected improvement for complete DAG configurations."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import pandas as pd

from baybe.acquisition.acqfs import (
    qExpectedImprovement as BayBEqExpectedImprovement,
)
from baybe.acquisition.acqfs import (
    qLogExpectedImprovement as BayBEqLogExpectedImprovement,
)
from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import IncompatibilityError, IncompatibleAcquisitionFunctionError
from baybe.multistep.model import _make_acquisition_model_adapter

if TYPE_CHECKING:
    from botorch.acquisition.acquisition import (
        AcquisitionFunction as BoAcquisitionFunction,
    )

    from baybe.multistep.model import MultiStepModel


_ALLOWED_ACQUISITIONS = (BayBEqExpectedImprovement, BayBEqLogExpectedImprovement)
_LOG_EI_OPTIONS = {
    "fat": True,
    "tau_relu": 1e-6,
    "tau_max": 1e-2,
}
_MIXED_OPTIMIZER_OPTIONS = frozenset(
    {
        "initialization_strategy",
        "maxiter_alternating",
        "maxiter_discrete",
        "maxiter_continuous",
        "tol",
        "batch_limit",
        "init_batch_limit",
        "num_spray_points",
        "max_discrete_values",
    }
)


def _require_parameters(function: Any, required: set[str], *, label: str) -> None:
    """Raise a compatibility error when an installed API lacks required keywords."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError) as ex:
        raise IncompatibleAcquisitionFunctionError(
            f"Cannot verify the installed BoTorch API for {label}."
        ) from ex
    if not required <= set(parameters):
        missing = sorted(required - set(parameters))
        raise IncompatibleAcquisitionFunctionError(
            f"The installed BoTorch API for {label} lacks required options {missing}."
        )


def _require_documented_options(
    function: Any, required: frozenset[str], *, label: str
) -> None:
    """Require explicit documentation for every option passed through a mapping."""
    documentation = inspect.getdoc(function)
    missing = sorted(
        option
        for option in required
        if documentation is None or option not in documentation
    )
    if missing:
        raise IncompatibleAcquisitionFunctionError(
            f"Cannot verify required {label} options {missing} from the installed API."
        )


def validate_configuration_optimization_api(
    acquisition_function: AcquisitionFunction | str,
    *,
    has_discrete_parameters: bool,
) -> AcquisitionFunction:
    """Resolve EI and verify the BoTorch APIs required by the configuration path.

    Args:
        acquisition_function: BayBE qEI/qLogEI instance or its registered name.
        has_discrete_parameters: Whether mixed alternating optimization is required.

    Returns:
        The resolved BayBE qEI or qLogEI acquisition.

    Raises:
        IncompatibleAcquisitionFunctionError: If the acquisition or installed
            BoTorch API is unsupported.
    """
    try:
        acquisition = convert_acqf(acquisition_function)
    except (TypeError, ValueError, KeyError) as ex:
        raise IncompatibleAcquisitionFunctionError(
            f"Cannot resolve configuration acquisition {acquisition_function!r}."
        ) from ex
    if type(acquisition) not in _ALLOWED_ACQUISITIONS:
        raise IncompatibleAcquisitionFunctionError(
            "Complete-configuration recommendations support only BayBE qEI and "
            "qLogEI acquisitions."
        )

    try:
        from botorch.acquisition.logei import qLogExpectedImprovement
        from botorch.acquisition.monte_carlo import qExpectedImprovement
        from botorch.sampling.get_sampler import get_sampler
    except ImportError as ex:
        raise IncompatibleAcquisitionFunctionError(
            "The installed BoTorch version lacks an API required for complete-"
            "configuration optimization."
        ) from ex

    _require_parameters(
        get_sampler, {"posterior", "sample_shape", "seed"}, label="get_sampler"
    )
    acquisition_class = (
        qExpectedImprovement
        if isinstance(acquisition, BayBEqExpectedImprovement)
        else qLogExpectedImprovement
    )
    required_acquisition_args = {"model", "best_f", "sampler", "objective"}
    if isinstance(acquisition, BayBEqLogExpectedImprovement):
        required_acquisition_args.update(_LOG_EI_OPTIONS)
    _require_parameters(
        acquisition_class,
        required_acquisition_args,
        label=acquisition_class.__name__,
    )
    _validate_optimizer_api(has_discrete_parameters=has_discrete_parameters)
    return acquisition


def _validate_optimizer_api(*, has_discrete_parameters: bool) -> None:
    """Verify the installed continuous or mixed optimizer signature."""
    try:
        from botorch.optim import optimize_acqf

        if has_discrete_parameters:
            from botorch.optim import optimize_acqf_mixed_alternating
    except ImportError as ex:
        raise IncompatibleAcquisitionFunctionError(
            "The installed BoTorch version lacks an API required for complete-"
            "configuration optimization."
        ) from ex
    required = {"acq_function", "bounds", "q", "num_restarts", "raw_samples", "options"}
    if has_discrete_parameters:
        _require_parameters(
            optimize_acqf_mixed_alternating,
            {
                "acq_function",
                "bounds",
                "discrete_dims",
                "options",
                "q",
                "raw_samples",
                "num_restarts",
                "sequential",
            },
            label="optimize_acqf_mixed_alternating",
        )
        _require_documented_options(
            optimize_acqf_mixed_alternating,
            _MIXED_OPTIMIZER_OPTIONS,
            label="optimize_acqf_mixed_alternating",
        )
    else:
        _require_parameters(optimize_acqf, required, label="optimize_acqf")


def make_configuration_acquisition(
    model: MultiStepModel,
    references: pd.DataFrame,
    acquisition_function: AcquisitionFunction | str,
    *,
    sampling_seed: int,
    n_mc_samples: int,
) -> BoAcquisitionFunction:
    """Construct qEI/qLogEI against successful complete-process model means.

    The reference is ``max objective(model_adapter.posterior(x).mean)`` over real
    complete successful process references. Objective transformations are applied
    once to posterior means and once by BoTorch to posterior samples. Parameters
    use their native units and EI uses the objective's utility units.

    Args:
        model: The fitted multi-step network model.
        references: Complete successful process measurements in experimental form.
        acquisition_function: BayBE qEI/qLogEI instance or registered name.
        sampling_seed: Nonnegative integer seed for the fixed MC sample set.
        n_mc_samples: Positive integer number of acquisition samples.

    Returns:
        A standard BoTorch qExpectedImprovement or qLogExpectedImprovement.

    Raises:
        IncompatibilityError: If references or the computed model reference are
            invalid.
        IncompatibleAcquisitionFunctionError: If the acquisition or BoTorch API is
            unsupported.
        TypeError: If seed or sample count is not a genuine integer.
        ValueError: If seed or sample count is outside its valid range.
    """
    import torch
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.acquisition.monte_carlo import qExpectedImprovement
    from botorch.sampling.get_sampler import get_sampler

    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
        raise TypeError("sampling_seed must be a non-negative integer.")
    if sampling_seed < 0:
        raise ValueError("sampling_seed must be a non-negative integer.")
    if isinstance(n_mc_samples, bool) or not isinstance(n_mc_samples, int):
        raise TypeError("n_mc_samples must be a positive integer.")
    if n_mc_samples < 1:
        raise ValueError("n_mc_samples must be a positive integer.")
    if references.empty:
        raise IncompatibilityError(
            "Configuration EI requires at least one complete successful process "
            "reference."
        )

    dag = model.dag
    configuration_space = dag._configuration_space
    required_columns = {
        *configuration_space.parameter_names,
        *(target.name for target in dag.objective.targets),
    }
    missing = sorted(required_columns - set(references.columns))
    if missing:
        raise IncompatibilityError(
            f"Complete configuration references are missing columns {missing}."
        )

    acquisition = validate_configuration_optimization_api(
        acquisition_function,
        has_discrete_parameters=bool(configuration_space.discrete_dims),
    )
    reference_parameter = next(model.parameters())
    comp_references = configuration_space.transform(
        references.loc[:, list(configuration_space.parameter_names)]
    )
    reference_candidates = torch.as_tensor(
        comp_references.to_numpy(copy=True),
        dtype=reference_parameter.dtype,
        device=reference_parameter.device,
    ).unsqueeze(-2)
    model_adapter = _make_acquisition_model_adapter(model)
    posterior = model_adapter.posterior(reference_candidates)
    objective = dag.objective.to_botorch()
    reference_utilities = objective(posterior.mean, reference_candidates)
    if (
        reference_utilities.numel() != len(references)
        or not torch.isfinite(reference_utilities).all()
    ):
        raise IncompatibilityError(
            "Complete-process model references must yield one finite scalar utility "
            "per real measurement."
        )
    best_f = reference_utilities.max()
    sampler = get_sampler(
        posterior,
        sample_shape=torch.Size([n_mc_samples]),
        seed=sampling_seed,
    )
    if isinstance(acquisition, BayBEqExpectedImprovement):
        return qExpectedImprovement(
            model=model_adapter,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
        )
    return qLogExpectedImprovement(
        model=model_adapter,
        best_f=best_f,
        sampler=sampler,
        objective=objective,
        **_LOG_EI_OPTIONS,
    )


def optimize_configuration(
    model: MultiStepModel,
    acquisition: BoAcquisitionFunction,
    *,
    sampling_seed: int,
    num_restarts: int,
    raw_samples: int,
) -> pd.DataFrame:
    """Optimize one EI candidate over a factorized numeric DAG parameter space.

    Args:
        model: The fitted multi-step network model.
        acquisition: The standard BoTorch qEI/qLogEI object.
        sampling_seed: The optimizer seed, independent from the MC sample seed.
        num_restarts: Positive number of continuous optimizer restarts.
        raw_samples: Number of raw initial candidates, at least ``num_restarts``.

    Returns:
        One experimental-representation row in DAG parameter declaration order.

    Raises:
        IncompatibilityError: If the optimizer returns invalid candidate values.
        IncompatibleAcquisitionFunctionError: If a required optimization API is
            unsupported.
        TypeError: If one of the seed/count arguments is not a genuine integer.
        ValueError: If seed/count relationships are invalid.
    """
    import torch

    for name, value, minimum in (
        ("sampling_seed", sampling_seed, 0),
        ("num_restarts", num_restarts, 1),
        ("raw_samples", raw_samples, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
    if raw_samples < num_restarts:
        raise ValueError("raw_samples must be at least num_restarts.")

    configuration_space = model.dag._configuration_space
    has_discrete = bool(configuration_space.discrete_dims)
    if not configuration_space.parameter_names:
        return pd.DataFrame([{}], columns=configuration_space.parameter_names)
    _validate_optimizer_api(has_discrete_parameters=has_discrete)

    reference = next(model.parameters())
    bounds = torch.as_tensor(
        configuration_space.bounds.to_numpy(copy=True),
        dtype=reference.dtype,
        device=reference.device,
    )
    devices = (
        [
            reference.device.index
            if reference.device.index is not None
            else torch.cuda.current_device()
        ]
        if reference.device.type == "cuda"
        else []
    )
    options = {
        "initialization_strategy": "random",
        "maxiter_alternating": 64,
        "maxiter_discrete": 4,
        "maxiter_continuous": 8,
        "tol": 1e-8,
        "batch_limit": 1,
        "init_batch_limit": 1,
        "num_spray_points": 0,
        "max_discrete_values": max(
            20,
            max(
                (len(values) for values in configuration_space.discrete_dims.values()),
                default=0,
            ),
        ),
    }
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(sampling_seed)
        if reference.device.type == "cuda":
            with torch.cuda.device(reference.device):
                torch.cuda.manual_seed(sampling_seed)
        if has_discrete:
            from botorch.optim import optimize_acqf_mixed_alternating

            candidate, _ = optimize_acqf_mixed_alternating(
                acq_function=acquisition,
                bounds=bounds,
                discrete_dims=configuration_space.discrete_dims,
                q=1,
                sequential=True,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
            )
        else:
            from botorch.optim import optimize_acqf

            candidate, _ = optimize_acqf(
                acq_function=acquisition,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"maxiter": 200, "batch_limit": 1},
            )

    if (
        candidate.shape
        not in (
            torch.Size([1, len(configuration_space.comp_rep_columns)]),
            torch.Size([1, 1, len(configuration_space.comp_rep_columns)]),
        )
        or not torch.isfinite(candidate).all()
    ):
        raise IncompatibilityError(
            "Configuration optimization returned an invalid candidate shape or "
            "nonfinite values."
        )
    values = candidate.reshape(-1).detach().cpu().tolist()
    if any(
        value < bounds[0, index].item() or value > bounds[1, index].item()
        for index, value in enumerate(values)
    ):
        raise IncompatibilityError(
            "Configuration optimization returned a candidate outside the bounds."
        )
    configuration = dict(zip(configuration_space.comp_rep_columns, values, strict=True))
    configuration = {
        name: configuration[name] for name in configuration_space.parameter_names
    }
    configuration_space.validate(configuration)
    return pd.DataFrame([configuration], columns=configuration_space.parameter_names)


__all__ = [
    "make_configuration_acquisition",
    "optimize_configuration",
    "validate_configuration_optimization_api",
]
