"""BoTorch network model over a DAG of independently fitted node models.

The model owns one node model per DAG node; a multi-output node model bundles
independently fitted models in the node's declared output order. It derives all
node input bookkeeping from the DAG itself. Network posteriors are formed by
Monte-Carlo propagation: every sample is one full network path, parent outputs
are drawn once and reused at forks, and the final outputs are returned jointly
over the q-axis. A lazy network posterior coordinates native BoTorch samplers
through the DAG, without resampling a precomputed ensemble.

Gaussian node draws remain differentiable with respect to the candidates.
Ensemble member selection is discrete; gradients through member predictions
depend on the local model (random forests, for example, detach their inputs).

The classes are ``attrs`` classes at the boundary to BoTorch. Two adjustments
are required because :class:`torch.nn.Module` owns the instance state and its
identity: attribute assignment must use the module's ``__setattr__`` so that
attrs does not install its own (``setters.NO_OP``), and acquisition-argument
collection must not recursively convert the model into a dictionary.
"""

from __future__ import annotations

import gc
import hashlib
import math
import warnings
from collections.abc import Mapping
from copy import copy, deepcopy
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from attrs import define, evolve, field, setters
from attrs.validators import ge, instance_of
from pandas.api.types import is_numeric_dtype
from typing_extensions import override

from baybe.exceptions import IncompatibilityError, ModelNotTrainedError
from baybe.kernels.base import Kernel
from baybe.recommenders.pure.experimental_feasibility import (
    InfeasibleMeasurementMode,
    _GaussianProcessClassifier,
)
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import Surrogate, SurrogateProtocol
from baybe.surrogates.gaussian_process.core import (
    GaussianProcessSurrogate,
    _ModelContext,
)
from baybe.surrogates.gaussian_process.presets.baybe import (
    KERNEL_FACTORY as BAYBE_DEFAULT_KERNEL_FACTORY,
)
from baybe.targets.base import Target
from baybe.targets.binary import BinaryTarget
from baybe.targets.numerical import NumericalTarget
from baybe.utils.validation import validate_target_input

from ._columns import (
    EXPERIMENT_GROUP,
    EXPERIMENT_ID,
    FEASIBILITY_FEATURE_PREFIX,
    FEASIBILITY_LABEL,
    MEASUREMENT_ID,
)
from .dag import MultiStepDAG
from .data import (
    _extract_feasibility_values,
    _get_node,
    _make_local_searchspace,
    _predicted_mean_rows,
    _validate_finite_input,
    _validate_node_inputs,
)
from .node import MultiStepNode

if TYPE_CHECKING:
    import torch
    from botorch.models.model import Model
    from botorch.models.model import Model as _ModelBase
    from botorch.posteriors.posterior import Posterior
    from botorch.posteriors.posterior import Posterior as _PosteriorBase
    from botorch.sampling.base import MCSampler
    from torch import Tensor

    from baybe.objectives.base import Objective

    from .data import _PendingAction
else:
    _ModelBase = object
    _PosteriorBase = object


@define
class _MultiStepGPClassifier(_GaussianProcessClassifier):
    """Fit the node-success classifier with a zero mean and BayBE's default kernel."""

    _is_prior: bool = field(default=False, init=False)

    def initialize_prior(self, searchspace: SearchSpace, objective: Objective) -> None:
        """Construct the unconditioned classifier without adding observations."""
        import torch
        from botorch.exceptions import InputDataWarning, UserInputWarning
        from botorch.models import SingleTaskVariationalGP
        from gpytorch.likelihoods import BernoulliLikelihood
        from gpytorch.means import ZeroMean

        n_features = len(searchspace.comp_rep_columns)
        if not n_features:
            raise IncompatibilityError(
                "The feasibility classifier requires at least one computational "
                "input feature. Check the node inputs and parent context."
            )
        measurements = pd.DataFrame(
            columns=[*searchspace.parameter_names, FEASIBILITY_LABEL]
        )
        context = _ModelContext(searchspace, objective, measurements)
        kernel = BAYBE_DEFAULT_KERNEL_FACTORY(searchspace, objective, measurements)
        if isinstance(kernel, Kernel):
            kernel = kernel.to_gpytorch(searchspace)
        # Inducing locations define the approximation only; no labels are supplied
        # and prior predictions below bypass the variational distribution entirely.
        inducing = torch.zeros((1, n_features), dtype=torch.double)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InputDataWarning)
            warnings.simplefilter("ignore", UserInputWarning)
            model = SingleTaskVariationalGP(
                inducing,
                inducing_points=inducing,
                likelihood=BernoulliLikelihood(),
                covar_module=kernel,
                mean_module=ZeroMean(),
                input_transform=self._make_input_transform(context),
            )
        self._searchspace = searchspace
        self._objective = objective
        self._measurements = measurements
        self._input_scaler = self._make_input_scaler(searchspace)
        self._output_scaler = self._make_output_scaler(objective, measurements)
        self._model = model
        self._is_prior = True

    @override
    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        try:
            super().fit(searchspace, objective, measurements)
        except Exception:
            # The base fit caches its context before fitting. A failed fit must
            # neither expose stale predictions nor make a retry look cached.
            self._model = None
            self._measurements_hash = None
            raise

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Fit variational class probabilities and kernel hyperparameters.

        Raises:
            IncompatibilityError: If the classifier input has no features.
        """
        self._model = None
        self._is_prior = False
        if train_x.shape[-1] == 0:
            raise IncompatibilityError(
                "The feasibility classifier requires at least one computational "
                "input feature. Check the node inputs and parent context."
            )

        from botorch.exceptions import InputDataWarning, UserInputWarning
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskVariationalGP
        from gpytorch.likelihoods import BernoulliLikelihood
        from gpytorch.means import ZeroMean
        from gpytorch.mlls import VariationalELBO

        assert self._searchspace is not None
        assert self._objective is not None
        assert self._measurements is not None

        context = _ModelContext(self._searchspace, self._objective, self._measurements)
        kernel = BAYBE_DEFAULT_KERNEL_FACTORY(
            context.searchspace, context.objective, context.measurements
        )
        if isinstance(kernel, Kernel):
            kernel = kernel.to_gpytorch(context.searchspace)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InputDataWarning)
            warnings.simplefilter("ignore", UserInputWarning)
            model = SingleTaskVariationalGP(
                train_x,
                train_y,
                likelihood=BernoulliLikelihood(),
                inducing_points=len(train_x),
                covar_module=kernel,
                mean_module=ZeroMean(),
                input_transform=(
                    self._make_input_transform(context)
                    if self._searchspace.comp_rep_columns
                    else None
                ),
            )
        criterion = VariationalELBO(
            model.likelihood, model.model, num_data=len(train_x)
        )
        fit_gpytorch_mll(criterion)
        self._model = model

    @override
    def posterior_probabilities(self, candidates_comp: Tensor, /) -> Tensor:
        """Return integrated class probabilities from the fitted GP.

        Raises:
            ModelNotTrainedError: If the classifier has not been fitted successfully.
        """
        if self._model is None:
            raise ModelNotTrainedError(
                f"The {self.__class__.__name__} has no successfully fitted "
                "feasibility model."
            )
        if self._is_prior:
            assert self._input_scaler is not None
            model = self._model
            model.eval()
            inputs = self._input_scaler.transform(candidates_comp)
            inputs = model.transform_inputs(inputs)
            return model.likelihood(model.model.forward(inputs)).mean.reshape(-1)
        return super().posterior_probabilities(candidates_comp)


def _make_feasibility_searchspace(
    dag: MultiStepDAG,
    node: MultiStepNode,
    inputs: pd.DataFrame,
) -> tuple[SearchSpace, dict[str, str]]:
    """Create a local feature space and names for numeric-output probabilities."""
    from baybe.multistep.data import _make_local_searchspace
    from baybe.parameters.numerical import NumericalContinuousParameter

    input_names = dag.context.input_names(node)
    base = _make_local_searchspace(dag.context, node, inputs[list(input_names)])
    conditions_by_output = {
        name
        for name in node.output_names
        if any(condition.column == name for condition in node.feasibility_conditions)
    }
    feature_names: dict[str, str] = {}
    used_names = (
        {parameter.name for parameter in dag.parameter_by_name.values()}
        | {output for dag_node in dag.nodes for output in dag_node.output_names}
        | {
            condition.column
            for dag_node in dag.nodes
            for condition in dag_node.feasibility_conditions
        }
        | {
            input_name
            for dag_node in dag.nodes
            for input_name in dag.context.input_names(dag_node)
        }
    )
    probability_parameters = []
    for output_index, output_name in enumerate(node.output_names):
        if output_name not in conditions_by_output:
            continue
        candidate = f"{FEASIBILITY_FEATURE_PREFIX}{output_index}"
        while candidate in used_names:
            candidate += "_"
        used_names.add(candidate)
        feature_names[output_name] = candidate
        probability_parameters.append(
            NumericalContinuousParameter(candidate, bounds=(0.0, 1.0))
        )
    return SearchSpace.from_product([*base.parameters, *probability_parameters]), (
        feature_names
    )


def _make_grouped_folds(frame: pd.DataFrame) -> tuple[tuple[list[int], list[int]], ...]:
    """Partition rows deterministically without separating known experiment groups."""
    if frame.empty:
        return ()
    identities = frame[MEASUREMENT_ID].tolist()
    groups = frame[EXPERIMENT_GROUP].tolist()
    members: dict[Any, list[int]] = {}
    for position, (identity, group) in enumerate(zip(identities, groups)):
        try:
            missing_group = bool(pd.isna(group))
        except (TypeError, ValueError):
            missing_group = False
        key = ("measurement", identity) if missing_group else group
        try:
            hash(key)
        except TypeError as ex:
            raise IncompatibilityError(
                f"Cross-fit group identity at row {identity!r} must be hashable."
            ) from ex
        members.setdefault(key, []).append(position)

    if len(members) == 1:
        return (([], list(range(len(frame)))),)
    n_folds = min(5, len(members))
    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    bucket_sizes = [0] * n_folds
    ordered_groups = sorted(
        members.values(), key=lambda positions: (-len(positions), positions[0])
    )
    for positions in ordered_groups:
        bucket_index = min(
            range(n_folds), key=lambda index: (bucket_sizes[index], index)
        )
        buckets[bucket_index].extend(positions)
        bucket_sizes[bucket_index] += len(positions)
    all_positions = set(range(len(frame)))
    return tuple(
        (
            sorted(all_positions - set(test_positions)),
            sorted(test_positions),
        )
        for test_positions in buckets
    )


def _interval_probability(
    node: MultiStepNode,
    output_name: str,
    posterior: Any,
) -> Tensor:
    """Compute one output's posterior mass inside its joint feasibility interval."""
    conditions = tuple(
        condition
        for condition in node.feasibility_conditions
        if condition.column == output_name
    )
    if not conditions:
        return posterior.mean.new_ones(posterior.mean.shape[:-1] + (1,))

    import torch

    output_index = node.output_names.index(output_name)
    mean = posterior.mean[..., output_index : output_index + 1]
    standard_deviation = (
        posterior.variance[..., output_index : output_index + 1].clamp_min(0.0).sqrt()
    )
    deterministic = standard_deviation == 0.0
    standard_deviation = standard_deviation.clamp_min(
        torch.finfo(standard_deviation.dtype).eps
    )
    lower_bounds = [
        cast(float, condition.threshold)
        for condition in conditions
        if condition.operator == ">=" and condition.threshold is not None
    ]
    upper_bounds = [
        cast(float, condition.threshold)
        for condition in conditions
        if condition.operator == "<=" and condition.threshold is not None
    ]
    lower = max(lower_bounds) if lower_bounds else None
    upper = min(upper_bounds) if upper_bounds else None
    if lower is not None and upper is not None and lower > upper:
        return mean.new_zeros(mean.shape)

    upper_probability = mean.new_ones(mean.shape)
    if upper is not None:
        upper_probability = torch.special.ndtr(
            (mean.new_tensor(upper) - mean) / standard_deviation
        )
    lower_probability = mean.new_zeros(mean.shape)
    if lower is not None:
        lower_probability = torch.special.ndtr(
            (mean.new_tensor(lower) - mean) / standard_deviation
        )
    probability = (upper_probability - lower_probability).clamp(0.0, 1.0)
    deterministic_passes = torch.ones_like(mean, dtype=torch.bool)
    if lower is not None:
        deterministic_passes &= mean >= lower
    if upper is not None:
        deterministic_passes &= mean <= upper
    return torch.where(deterministic, deterministic_passes.to(mean.dtype), probability)


def _output_probability_features(
    node: MultiStepNode,
    output_model: Model,
    inputs_comp: Tensor,
) -> dict[str, Tensor]:
    """Return one differentiable interval-probability feature per constrained output."""
    output_names = tuple(
        name
        for name in node.output_names
        if any(condition.column == name for condition in node.feasibility_conditions)
    )
    if not output_names:
        return {}
    posterior = cast(Any, output_model).posterior(inputs_comp)
    return {name: _interval_probability(node, name, posterior) for name in output_names}


def _make_regression_training_frames(
    dag: MultiStepDAG,
    node: MultiStepNode,
    raw_frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Apply one node's failure mode to raw rows without changing their labels."""
    input_names = dag.context.input_names(node)
    mode = node.infeasible_measurement_mode
    records = raw_frame.to_dict(orient="records")

    def is_failure(record: Mapping[str, Any]) -> bool:
        """Recognize a known failure without treating unknown labels as failures."""
        value = record[FEASIBILITY_LABEL]
        return isinstance(value, (bool, np.bool_)) and not bool(value)

    def is_success(record: Mapping[str, Any]) -> bool:
        """Recognize a known success without treating unknown labels as successes."""
        value = record[FEASIBILITY_LABEL]
        return isinstance(value, (bool, np.bool_)) and bool(value)

    selected: list[dict[str, Any]] = []
    if mode is InfeasibleMeasurementMode.OBSERVED:
        selected = [dict(record) for record in records]
    elif mode is InfeasibleMeasurementMode.EXCLUDE:
        selected = [dict(record) for record in records if not is_failure(record)]
    elif mode is InfeasibleMeasurementMode.DUMMY:
        dummy = node.infeasible_dummy_value
        if dummy is None:
            raise IncompatibilityError(
                f"Node '{node.name}' uses "
                f"'{InfeasibleMeasurementMode.DUMMY.name}' without dummy outputs."
            )
        dummy_values = (
            dict(dummy) if isinstance(dummy, Mapping) else {node.output_names[0]: dummy}
        )
        for record in records:
            row = dict(record)
            if is_failure(record):
                row.update(dummy_values)
            selected.append(row)
    else:
        assert mode is InfeasibleMeasurementMode.PREDICTED_MEAN
        selected = [dict(record) for record in records if not is_failure(record)]
        failures = [
            {name: record[name] for name in input_names}
            for record in records
            if is_failure(record)
        ]
        if failures:
            successes = [
                (
                    {name: record[name] for name in input_names},
                    {name: record[name] for name in node.output_names},
                )
                for record in records
                if is_success(record)
            ]
            selected.extend(_predicted_mean_rows(dag, node, successes, failures))

    result = {}
    for output_name in node.output_names:
        frame = pd.DataFrame(
            selected,
            columns=[*input_names, *node.output_names],
        )
        frame = frame.loc[
            frame[output_name].notna(), [*input_names, output_name]
        ].copy()
        result[output_name] = frame
    return result


def validate_network_measurements(
    dag: MultiStepDAG,
    searchspace: SearchSpace,
    objective: Objective,
    measurements: pd.DataFrame,
) -> None:
    """Validate a complete multi-step measurement frame.

    The stock ``Campaign`` validates parameter and objective-target columns,
    but it does not know about the intermediate node outputs stored as extra
    columns. This function closes that gap before any node model is fitted.

    Args:
        dag: The DAG defining the expected data contract.
        searchspace: The DAG-wide search space passed by the recommender.
        objective: The objective assembled from the final nodes.
        measurements: The measurement frame containing complete DAG runs.

    Raises:
        IncompatibilityError: If columns are missing, duplicated, or
            non-finite, or if the search space/objective context does not
            match the DAG.
    """
    if measurements.columns.duplicated().any():
        duplicates = sorted(
            set(measurements.columns[measurements.columns.duplicated()])
        )
        raise IncompatibilityError(
            f"Measurement columns must be unique, but found duplicates: {duplicates}."
        )

    expected_parameters = tuple(dag.parameter_by_name)
    actual_parameters = tuple(parameter.name for parameter in searchspace.parameters)
    if set(actual_parameters) != set(expected_parameters):
        raise IncompatibilityError(
            f"The search space parameters {sorted(actual_parameters)} do not match "
            f"the DAG parameters {sorted(expected_parameters)}."
        )

    output_names = tuple(output for node in dag.nodes for output in node.output_names)
    missing = [
        name
        for name in (*expected_parameters, *output_names)
        if name not in measurements.columns
    ]
    if missing:
        raise IncompatibilityError(
            f"The measurements are missing required columns: {missing}. Each row "
            f"must describe one completely executed DAG evaluation."
        )

    targets = tuple(
        dag.target_by_output[name]
        if name in dag.target_by_output
        else NumericalTarget(name)
        for name in output_names
    )
    try:
        validate_target_input(measurements, targets)
    except (TypeError, ValueError) as ex:
        raise IncompatibilityError(
            f"Invalid node target values in the measurement frame: {ex}"
        ) from ex

    non_numeric = [
        name for name in output_names if not is_numeric_dtype(measurements[name])
    ]
    if non_numeric:
        raise IncompatibilityError(
            f"Node output columns must be numeric, but columns {non_numeric} are not."
        )

    values = measurements[list(output_names)].to_numpy(dtype=float)
    invalid_outputs = ~np.isfinite(values).all(axis=0)
    if invalid_outputs.any():
        bad = np.asarray(output_names)[invalid_outputs].tolist()
        raise IncompatibilityError(
            f"Node output columns must be finite, but columns {bad} contain "
            f"NaN or infinite values."
        )

    expected_targets = dag.final_target_names
    actual_targets = tuple(target.name for target in objective.targets)
    if set(actual_targets) != set(expected_targets):
        raise IncompatibilityError(
            f"The objective targets {actual_targets} do not match the final node "
            f"targets {expected_targets}. The objective must be the DAG objective."
        )


def _validate_network_fit_context(
    dag: MultiStepDAG,
    searchspace: SearchSpace | None,
    objective: Objective,
) -> SearchSpace:
    """Validate and resolve the DAG-owned context of a stock ``fit`` call."""
    expected = dag.full_searchspace
    resolved = expected if searchspace is None else searchspace
    expected_parameters = {
        parameter.name: parameter for parameter in expected.parameters
    }
    supplied_parameters = {
        parameter.name: parameter for parameter in resolved.parameters
    }
    discrete_names = tuple(parameter.name for parameter in expected.discrete.parameters)
    expected_discrete = {
        tuple(row)
        for row in expected.discrete.exp_rep[list(discrete_names)].itertuples(
            index=False, name=None
        )
    }
    supplied_discrete = {
        tuple(row)
        for row in resolved.discrete.exp_rep.reindex(columns=discrete_names).itertuples(
            index=False, name=None
        )
    }
    if (
        set(supplied_parameters) != set(expected_parameters)
        or any(
            supplied_parameters[name] != parameter
            for name, parameter in expected_parameters.items()
        )
        or resolved.is_constrained
        or supplied_discrete != expected_discrete
    ):
        raise IncompatibilityError(
            "The supplied search space is incompatible with the multi-step DAG. "
            "Use the DAG's complete, unconstrained search space."
        )

    expected_targets = tuple(target.name for target in dag.objective.targets)
    supplied_targets = tuple(target.name for target in objective.targets)
    if objective != dag.objective or supplied_targets != expected_targets:
        raise IncompatibilityError(
            "The supplied objective is incompatible with the multi-step DAG. "
            "Use the DAG's objective with its target order."
        )
    return resolved


def _make_node_model(models: tuple[Model, ...]) -> Model:
    """Combine independently fitted output models for one DAG node.

    Args:
        models: The models in the node's declared output order.

    Returns:
        The sole model unchanged, or a BoTorch model-list container.

    Raises:
        IncompatibilityError: If no output models are supplied.
    """
    if not models:
        raise IncompatibilityError("A node model requires at least one output model.")
    if len(models) == 1:
        return models[0]

    from botorch.models import ModelList
    from botorch.models.gpytorch import GPyTorchModel
    from botorch.models.model_list_gp_regression import ModelListGP

    if all(isinstance(model, GPyTorchModel) for model in models):
        return ModelListGP(*(cast(GPyTorchModel, model) for model in models))
    return ModelList(*models)


def _condition_node_model(model: Model, inputs: Tensor, outputs: Tensor) -> Model:
    """Condition a local model, preserving component-native GP transforms.

    Args:
        model: The fitted local model or GP bundle.
        inputs: Raw computational inputs with shape ``(..., q, d)``.
        outputs: Joint observations with shape ``(..., q, m)``.

    Returns:
        A temporary model conditioned on the supplied joint observations.

    Raises:
        IncompatibilityError: If a GP bundle's observation width does not match its
            components.
    """
    from botorch.models.model_list_gp_regression import ModelListGP

    if not isinstance(model, ModelListGP):
        return model.condition_on_observations(X=inputs, Y=outputs)

    if (
        outputs.ndim < 2
        or outputs.shape[-2] != inputs.shape[-2]
        or outputs.shape[-1] != model.num_outputs
    ):
        raise IncompatibilityError(
            "Joint node observations must match the GP bundle's q-axis and output "
            f"width ({inputs.shape[-2]}, {model.num_outputs}); received "
            f"{tuple(outputs.shape)}."
        )

    conditioned_models = tuple(
        cast(Any, component).condition_on_observations(
            X=inputs, Y=outputs[..., output_index : output_index + 1]
        )
        for output_index, component in enumerate(model.models)
    )
    return _make_node_model(conditioned_models)


def _make_local_surrogate(
    node: MultiStepNode,
    searchspace: SearchSpace,
    target: Target,
    measurements: pd.DataFrame,
    label: str,
    template: SurrogateProtocol | None = None,
) -> SurrogateProtocol:
    """Build a local prior, a conditioned prior, or a regular fitted surrogate.

    Args:
        node: The node owning the output model.
        searchspace: The local input space for the surrogate.
        target: The output target and its pre-transformations.
        measurements: The usable local observations for this output.
        label: The output name used in contextual errors.
        template: The configured or previously retained surrogate.

    Returns:
        The independent local surrogate for the current data state.

    Raises:
        IncompatibilityError: If fewer than two observations are available for a
            surrogate without GP-prior support, or if the configured GP prior cannot
            be represented.
    """
    objective = target.to_objective()
    surrogate = deepcopy(node.surrogate_model if template is None else template)
    if len(measurements) >= 2:
        surrogate.fit(searchspace, objective, measurements)
        return surrogate

    if not isinstance(surrogate, GaussianProcessSurrogate):
        raise IncompatibilityError(
            f"Node '{node.name}' uses surrogate '{type(surrogate).__name__}', which "
            f"does not support the required GP prior for output '{label}' with "
            f"{len(measurements)} usable measurement(s)."
        )

    try:
        import botorch
        import torch
        from joblib.hashing import hash as hash_measurements

        from baybe.utils.dataframe import to_tensor

        prior_measurements = measurements.iloc[:0].copy()
        context = _ModelContext(searchspace, objective, prior_measurements)
        kernel, mean, likelihood, _ = surrogate._resolve_components(context)
        train_x, train_y = to_tensor(
            pd.DataFrame(columns=searchspace.comp_rep_columns),
            objective._pre_transform(prior_measurements, allow_extra=True),
        )
        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(-1)
        input_transform = (
            surrogate._make_input_transform(context)
            if searchspace.comp_rep_columns
            else None
        )
        surrogate._model = botorch.models.SingleTaskGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=None,
            mean_module=mean,
            covar_module=kernel,
            likelihood=likelihood,
        )
        surrogate._searchspace = searchspace
        surrogate._objective = objective
        surrogate._measurements = prior_measurements
        surrogate._measurements_hash = hash_measurements(prior_measurements)
        surrogate._input_scaler = surrogate._make_input_scaler(searchspace)
        surrogate._output_scaler = surrogate._make_output_scaler(
            objective, prior_measurements
        )
        if len(measurements) == 1:
            train_x, train_y = to_tensor(
                searchspace.transform(measurements, allow_extra=True),
                objective._pre_transform(measurements, allow_extra=True),
            )
            if train_y.ndim == 1:
                train_y = train_y.unsqueeze(-1)
            # Initialize BoTorch's prediction caches before conditioning, matching
            # the native GP fantasy path without adding a synthetic observation.
            with torch.no_grad():
                surrogate._model.posterior(train_x)
            surrogate._model = surrogate._model.condition_on_observations(
                X=train_x, Y=train_y
            )
            surrogate._measurements = measurements
            surrogate._measurements_hash = hash_measurements(measurements)
    except Exception as ex:
        raise IncompatibilityError(
            f"Could not create the configured GP prior for output '{label}' of node "
            f"'{node.name}' using surrogate '{type(surrogate).__name__}': {ex}"
        ) from ex
    return surrogate


@define(eq=False, repr=False, slots=False, on_setattr=setters.NO_OP)
class _MultiStepModelBase(_ModelBase):
    """Shared implementation of the multi-step network model.

    The model owns one fitted node model per DAG node and composes them by
    forward sampling. Outputs declared by the same node are fitted independently
    and kept in one model-list when needed. Descendants see the sampled outputs of
    their parents, so the resulting posterior reflects the full network
    propagation rather than independent per-node predictions. Node models are
    fitted from per-node training frames or from a complete measurement frame,
    and fitted state is published atomically. A later stock :meth:`fit` call always
    consumes its supplied complete measurements, regardless of how the current
    models were fitted.

    The model implements
    :class:`~baybe.surrogates.base.SurrogateProtocol`, so the stock recommenders
    consume its output posterior like any other surrogate. Feasibility remains a
    separate channel for acquisition constraints and hypothetical observations.

    The public :class:`MultiStepModel` adds the BoTorch
    :class:`~botorch.models.model.Model` base only when the model is first
    requested, so importing this module does not load Torch or BoTorch.
    """

    dag: MultiStepDAG = field(validator=instance_of(MultiStepDAG))
    """The DAG topology."""

    num_network_samples: int = field(default=512, validator=ge(1))
    """Number of paths used to estimate posterior moments.

    Acquisition samplers control their own number of complete network draws.
    """

    sampling_seed: int = field(default=0, validator=ge(0))
    """Base seed for the deterministic network draws."""

    _node_models: dict[str, Model] = field(init=False, factory=dict, eq=False)
    """Mapping from node name to a fitted model exposing that node's outputs.

    Single-output models remain unwrapped; multi-output nodes use a model-list
    container in the declared output order. Each model operates on the node's local
    computational representation.
    """

    _node_surrogates: dict[str, dict[str, SurrogateProtocol]] = field(
        init=False, factory=dict, eq=False
    )
    """Mapping from node and output names to fitted output surrogates.

    The retained surrogates cache their fit context, so refitting unchanged node
    data is a no-op. They are bound to the node configurations of the DAG the
    model was created with.
    """

    _feasibility_models: dict[str, _MultiStepGPClassifier] = field(
        init=False, factory=dict, eq=False
    )
    """Joint node-success classifiers with their local feature spaces."""

    _feasibility_feature_names: dict[str, dict[str, str]] = field(
        init=False, factory=dict, eq=False
    )
    """Output-probability feature names published with each classifier."""

    _node_names: tuple[str, ...] = field(init=False)
    """The node names in topological order."""

    def __attrs_post_init__(self) -> None:
        """Initialize the module state and the empty node-model registry."""
        import torch

        super().__init__()

        self._node_names = self.dag.topological_order
        # Register the node models with the torch module so that device and
        # dtype conversions propagate to them. The mapping itself stays a plain
        # dict because node names are arbitrary and need not be valid module
        # keys.
        self._node_modules = torch.nn.ModuleList()
        self._feasibility_modules = torch.nn.ModuleList()

    @classmethod
    def from_node_models(
        cls,
        dag: MultiStepDAG,
        node_models: Mapping[str, Model],
        *,
        num_network_samples: int = 512,
        sampling_seed: int = 0,
    ) -> MultiStepModel:
        """Create a model from already fitted node models.

        This is the advanced entry point for node models that were fitted
        outside the multi-step layer. Each model must expose the outputs of its node
        in the node's declared order. No fitting is performed, and the provided
        models are used until a subsequent fit replaces them.

        Args:
            dag: The DAG topology.
            node_models: Mapping covering every DAG node.
            num_network_samples: Number of network paths used for posterior moments.
            sampling_seed: Base seed for the deterministic network draws.

        Returns:
            The model publishing the provided node models.

        Raises:
            IncompatibilityError: If a DAG node has no model, a model belongs to an
                unknown node, or a model's output count does not match its node.
        """
        model = cls(
            dag=dag,
            num_network_samples=num_network_samples,
            sampling_seed=sampling_seed,
        )
        missing = set(dag.topological_order) - set(node_models)
        if missing:
            raise IncompatibilityError(
                f"Missing fitted models for nodes: {sorted(missing)}."
            )
        model._publish(node_models, {})
        return model

    def _publish(
        self,
        node_models: Mapping[str, Model],
        node_surrogates: Mapping[str, Mapping[str, SurrogateProtocol]],
        *,
        feasibility_models: Mapping[str, _MultiStepGPClassifier] | None = None,
        feasibility_feature_names: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        """Publish a new fitted-model context atomically.

        This is the single mutation point of any successful fit path. Partially
        fitted models are permitted; completeness is checked when a posterior is
        requested.

        Args:
            node_models: The fitted node models to publish.
            node_surrogates: The fitted surrogates the node models were derived
                from. Node models fitted outside the multi-step layer have no
                corresponding surrogate.
            feasibility_models: Joint node-success classifiers to publish.
            feasibility_feature_names: Output-probability feature layout for each
                fitted classifier.

        Raises:
            IncompatibilityError: If a model or surrogate belongs to an unknown
                node, a model has the wrong output count, or surrogate mappings do
                not cover the node's outputs exactly.
        """
        feasibility_models = {} if feasibility_models is None else feasibility_models
        feasibility_feature_names = (
            {} if feasibility_feature_names is None else feasibility_feature_names
        )
        unknown = set(feasibility_models) - {
            node.name for node in self.dag.nodes if node.feasibility_conditions
        }
        if unknown:
            raise IncompatibilityError(
                f"Feasibility models provided for unconfigured nodes: "
                f"{sorted(unknown)}."
            )
        if set(feasibility_feature_names) != set(feasibility_models):
            raise IncompatibilityError(
                "Every fitted feasibility classifier must be published with exactly "
                "one output-probability feature layout."
            )
        if not all(
            isinstance(model, _MultiStepGPClassifier)
            for model in feasibility_models.values()
        ):
            raise IncompatibilityError("Feasibility models must be native surrogates.")
        unknown = set(node_models) - set(self._node_names)
        if unknown:
            raise IncompatibilityError(
                f"Fitted models provided for unknown nodes: {sorted(unknown)}."
            )
        unknown = set(node_surrogates) - set(self._node_names)
        if unknown:
            raise IncompatibilityError(
                f"Fitted surrogates provided for unknown nodes: {sorted(unknown)}."
            )
        wrong_output_count = [
            name
            for name, model in node_models.items()
            if model.num_outputs != len(self.dag.node_by_name[name].output_names)
        ]
        if wrong_output_count:
            raise IncompatibilityError(
                "Fitted node models must produce exactly the declared outputs; "
                f"models for nodes {sorted(wrong_output_count)} have incompatible "
                "output counts."
            )
        for name, surrogates in node_surrogates.items():
            if name not in node_models:
                raise IncompatibilityError(
                    f"Fitted surrogates were provided for node '{name}' without a "
                    "fitted model."
                )
            expected_outputs = set(self.dag.node_by_name[name].output_names)
            actual_outputs = set(surrogates)
            if actual_outputs != expected_outputs:
                raise IncompatibilityError(
                    f"Fitted surrogate outputs for node '{name}' must be "
                    f"{sorted(expected_outputs)}, got {sorted(actual_outputs)}."
                )

        # Build the torch module list before touching any state so that a
        # registration failure cannot leave a partially published fit behind.
        # Re-registering keeps device and dtype conversions propagating to the
        # fitted models; only fitted nodes appear, so partial fits stay usable.
        import torch

        node_modules = torch.nn.ModuleList(
            [node_models[name] for name in self._node_names if name in node_models]
        )
        feasibility_modules = torch.nn.ModuleList(
            [
                cast(Surrogate, feasibility_models[name]).to_botorch()
                for name in self._node_names
                if name in feasibility_models
                and feasibility_models[name]._model is not None
            ]
        )

        self._node_models = dict(node_models)
        self._node_surrogates = {
            name: dict(surrogates) for name, surrogates in node_surrogates.items()
        }
        self._node_modules = node_modules
        self._feasibility_models = dict(feasibility_models)
        self._feasibility_feature_names = {
            name: dict(feature_names)
            for name, feature_names in feasibility_feature_names.items()
        }
        self._feasibility_modules = feasibility_modules

    @override
    @property
    def num_outputs(self) -> int:
        """The number of outputs produced by the final nodes."""
        return self.dag.final_output_dim

    @property
    def num_nodes(self) -> int:
        """The number of nodes in the DAG."""
        return len(self._node_names)

    @property
    def fitted_nodes(self) -> tuple[str, ...]:
        """The nodes whose every output has at least two real training values."""
        result = []
        for name in self._node_names:
            surrogates = self._node_surrogates.get(name)
            if name not in self._node_models:
                continue
            if not surrogates:
                # Models supplied through ``from_node_models`` are externally fitted.
                result.append(name)
                continue
            if all(
                (measurements := getattr(surrogate, "_measurements", None)) is not None
                and len(measurements) >= 2
                for surrogate in surrogates.values()
            ):
                result.append(name)
        return tuple(result)

    @property
    def missing_nodes(self) -> tuple[str, ...]:
        """The nodes without any model object required for network prediction."""
        return tuple(name for name in self._node_names if name not in self._node_models)

    @property
    def missing_feasibility_nodes(self) -> tuple[str, ...]:
        """Configured nodes without the models needed for feasibility prediction."""
        return tuple(
            name
            for name in self._node_names
            if self.dag.node_by_name[name].feasibility_conditions
            and not self._has_feasibility_prediction(name)
        )

    def _has_feasibility_prediction(self, node_name: str) -> bool:
        """Check whether a configured node can use its prior or fitted classifier."""
        node = self.dag.node_by_name[node_name]
        if not node.feasibility_conditions:
            return True
        classifier = self._feasibility_models.get(node_name)
        if classifier is None or classifier._model is None:
            return False
        feature_names = self._feasibility_feature_names.get(node_name, {})
        return not feature_names or node_name in self._node_models

    @cached_property
    def _full_comp_rep_index(self) -> dict[str, int]:
        """The indices of the DAG-wide comp-rep columns inside the candidate tensor.

        Returns:
            Mapping from computational-representation column name to its index.
        """
        return {
            column: index
            for index, column in enumerate(
                self.dag._configuration_space.comp_rep_columns
            )
        }

    def _validate_fitted(self) -> None:
        """Validate that every node has a model available for network prediction.

        Raises:
            ModelNotTrainedError: If at least one node has no fitted model.
        """
        missing = self.missing_nodes
        if missing:
            raise ModelNotTrainedError(
                f"The multi-step model must be fitted for all nodes before a "
                f"network posterior can be computed. Missing nodes: "
                f"{sorted(missing)}."
            )

    def fit(  # noqa: DOC502
        self,
        searchspace: SearchSpace | None,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Fit the node models on complete DAG measurement rows.

        This is the stock-BayBE entry point. The supplied complete-row data always
        replaces the previous fitted context, including one produced by
        :meth:`fit_node_data`; it is never silently ignored.

        Args:
            searchspace: The DAG-wide search space.
            objective: The objective assembled from the final node targets.
            measurements: Complete DAG evaluations in experimental
                representation.

        Raises:
            IncompatibilityError: If the data contract, search space, or objective
                context is incompatible with the DAG.
            ModelNotTrainedError: If a node has no training rows.
        """
        searchspace = _validate_network_fit_context(self.dag, searchspace, objective)
        validate_network_measurements(self.dag, searchspace, objective, measurements)

        def local_frame(node: MultiStepNode) -> pd.DataFrame:
            """Project a consistent complete row onto the configured history."""
            frame = measurements[[parameter.name for parameter in node.inputs]].copy()
            for feature in self.dag.context._features(node):
                frame[feature.name] = measurements[feature.source]
            return frame

        node_frames = {}
        for node in self.dag.nodes:
            frame = local_frame(node)
            frame[list(node.output_names)] = measurements[list(node.output_names)]
            node_frames[node.name] = frame
        feasibility_frames: dict[str, pd.DataFrame] = {}
        for node in self.dag.nodes:
            if not node.feasibility_conditions:
                continue
            auxiliary = tuple(
                dict.fromkeys(
                    condition.column
                    for condition in node.feasibility_conditions
                    if condition.column not in node.output_names
                )
            )
            raw = local_frame(node)
            raw[list(node.output_names)] = measurements[list(node.output_names)]
            for column in auxiliary:
                raw[column] = measurements[column] if column in measurements else pd.NA
            labels = _extract_feasibility_values(node, raw)
            frame = raw.copy()
            frame[FEASIBILITY_LABEL] = labels
            measurement_ids = list(range(len(frame)))
            experiment_ids = (
                measurements[EXPERIMENT_ID].tolist()
                if EXPERIMENT_ID in measurements
                else [None] * len(frame)
            )
            groups = []
            for measurement_id, experiment_id in zip(measurement_ids, experiment_ids):
                try:
                    missing_experiment = bool(pd.isna(experiment_id))
                except (TypeError, ValueError):
                    missing_experiment = False
                groups.append(
                    ("measurement", measurement_id)
                    if missing_experiment
                    else ("experiment", experiment_id)
                )
            frame[MEASUREMENT_ID] = measurement_ids
            frame[EXPERIMENT_ID] = experiment_ids
            frame[EXPERIMENT_GROUP] = groups
            feasibility_frames[node.name] = frame

        self._fit_node_frames(
            node_frames,
            allow_partial=False,
            fed_from_node_data=False,
            feasibility_frames=feasibility_frames,
        )

    def _fit_local_model(
        self,
        node: MultiStepNode,
        frame: pd.DataFrame,
        target: Target,
        template: SurrogateProtocol,
        label: str,
        *,
        fit_searchspace: SearchSpace | None = None,
    ) -> SurrogateProtocol:
        """Fit a copy of a local model on the node's training frame.

        The node's own parameters and configured history jointly form a local
        synthetic fit search space. Historical outputs are continuous dimensions
        of that local space only. Fitting always operates on a deep copy, leaving
        retained contexts and node templates untouched.

        Args:
            node: The node whose local model is fitted.
            frame: Validated training data in experimental representation.
            target: The target whose objective the model is fitted on.
            template: The unfitted model template or retained fitted model.
            label: The target column of the local model.
            fit_searchspace: A shared local search space for sibling output fits, or
                ``None`` to create the ordinary independent fit space.

        Returns:
            The independently fitted model copy.

        """
        measurements = frame[[*self.dag.context.input_names(node), label]].copy()
        if fit_searchspace is None:
            fit_searchspace = _make_local_searchspace(
                self.dag.context, node, measurements
            )
        return _make_local_surrogate(
            node,
            fit_searchspace,
            target,
            measurements,
            label,
            cast(SurrogateProtocol, template),
        )

    def _fit_cross_fitted_classifier(
        self, node: MultiStepNode, raw_frame: pd.DataFrame
    ) -> tuple[_MultiStepGPClassifier, dict[str, str]]:
        """Fit a joint-success classifier on leakage-free regression features."""
        input_names = self.dag.context.input_names(node)
        known = raw_frame[FEASIBILITY_LABEL].notna().to_numpy()
        feature_searchspace, feature_names = _make_feasibility_searchspace(
            self.dag, node, raw_frame[list(input_names)]
        )
        if not known.any():
            classifier = _MultiStepGPClassifier()
            target = BinaryTarget(
                FEASIBILITY_LABEL, success_value=True, failure_value=False
            )
            classifier.initialize_prior(feature_searchspace, target.to_objective())
            return classifier, feature_names
        constrained_outputs = tuple(feature_names)
        feature_values = {
            output_name: np.full(len(raw_frame), np.nan, dtype=float)
            for output_name in constrained_outputs
        }
        if constrained_outputs:
            import torch

            from baybe.utils.dataframe import to_tensor

            for train_positions, test_positions in _make_grouped_folds(raw_frame):
                known_test_positions = [
                    position for position in test_positions if known[position]
                ]
                if not known_test_positions:
                    continue
                training_raw = raw_frame.iloc[train_positions].reset_index(drop=True)
                training_views = _make_regression_training_frames(
                    self.dag, node, training_raw
                )
                fold_searchspace = _make_local_searchspace(
                    self.dag.context,
                    node,
                    training_raw[list(input_names)],
                )
                test_inputs = raw_frame.iloc[known_test_positions][list(input_names)]
                for output_name in constrained_outputs:
                    target = self.dag.target_by_output.get(
                        output_name, NumericalTarget(name=output_name)
                    )
                    template = self._node_surrogates.get(node.name, {}).get(
                        output_name, node.surrogate_model
                    )
                    try:
                        with torch.random.fork_rng():
                            torch.manual_seed(0)
                            surrogate = _make_local_surrogate(
                                node,
                                fold_searchspace,
                                target,
                                training_views[output_name],
                                output_name,
                                template,
                            )
                            with torch.no_grad():
                                posterior = surrogate.to_botorch().posterior(
                                    to_tensor(
                                        fold_searchspace.transform(test_inputs)
                                    ).unsqueeze(-2)
                                )
                                values = _interval_probability(
                                    node, output_name, posterior
                                ).reshape(-1)
                    except Exception as ex:
                        raise IncompatibilityError(
                            f"Could not create out-of-fold feasibility features for "
                            f"output '{output_name}' of node '{node.name}' using "
                            f"training fold {train_positions}: {ex}"
                        ) from ex
                    if (
                        values.numel() != len(known_test_positions)
                        or not torch.isfinite(values).all()
                    ):
                        raise IncompatibilityError(
                            f"The out-of-fold model for output '{output_name}' of node "
                            f"'{node.name}' returned invalid probability features."
                        )
                    feature_values[output_name][known_test_positions] = (
                        values.detach().cpu().numpy()
                    )

        known_positions = np.flatnonzero(known)
        classifier_frame = raw_frame.iloc[known_positions][list(input_names)].copy()
        for output_name, feature_name in feature_names.items():
            values = feature_values[output_name][known_positions]
            if not np.isfinite(values).all():
                raise IncompatibilityError(
                    f"Out-of-fold features for output '{output_name}' of node "
                    f"'{node.name}' are incomplete."
                )
            classifier_frame[feature_name] = values
        labels = raw_frame.iloc[known_positions][FEASIBILITY_LABEL].map(bool)
        classifier_frame[FEASIBILITY_LABEL] = labels.to_numpy(dtype=bool)

        target = BinaryTarget(
            FEASIBILITY_LABEL, success_value=True, failure_value=False
        )
        # Rebuild the stored classifier from its configuration instead of deep-copying
        # it: ``evolve`` avoids copying non-leaf GPyTorch tensors, and ``fit`` rebuilds
        # every piece of state from the new data anyway.
        template = self._feasibility_models.get(node.name, _MultiStepGPClassifier())
        classifier = evolve(template)
        classifier.fit(feature_searchspace, target.to_objective(), classifier_frame)
        return classifier, feature_names

    def _fit_node_frames(
        self,
        node_frames: Mapping[str, pd.DataFrame],
        *,
        allow_partial: bool,
        fed_from_node_data: bool,
        initialize_priors: bool = False,
        feasibility_frames: Mapping[str, pd.DataFrame] | None = None,
    ) -> None:
        """Fit the node models and publish the resulting context atomically.

        Args:
            node_frames: Mapping from node name to training data in
                experimental representation.
            allow_partial: If ``True``, nodes without enough training rows are
                left unfitted instead of raising an error.
            fed_from_node_data: Whether the frames are authoritative node data
                rather than a complete-row projection.
            initialize_priors: Whether zero/one-row GP outputs should receive an
                unfit prior or a conditioned prior.
            feasibility_frames: Explicit raw-label frames for configured nodes.

        Raises:
            IncompatibilityError: If a frame is missing columns, or raw feasibility
                labels, identities, or out-of-fold data contracts are invalid.
            ModelNotTrainedError: If a node has fewer than two rows and
                ``allow_partial`` is ``False``.
        """
        unknown = set(node_frames) - set(self._node_names)
        if unknown:
            raise IncompatibilityError(
                f"Training data provided for unknown nodes: {sorted(unknown)}."
            )

        configured = {
            node.name for node in self.dag.nodes if node.feasibility_conditions
        }
        feasibility_frames = {} if feasibility_frames is None else feasibility_frames
        if set(feasibility_frames) != configured:
            raise IncompatibilityError(
                "Provide explicit feasibility_frames for exactly the configured nodes "
                f"{sorted(configured)}, including empty frames."
            )
        normalized_feasibility_frames: dict[str, pd.DataFrame] = {}
        for node_name, frame in feasibility_frames.items():
            node = self.dag.node_by_name[node_name]
            raw_frame = frame.copy()
            for condition in node.feasibility_conditions:
                if (
                    condition.column not in node.output_names
                    and condition.column not in raw_frame
                ):
                    raw_frame[condition.column] = pd.NA
            normalized_feasibility_frames[node_name] = raw_frame
        feasibility_frames = normalized_feasibility_frames
        missing_nodes = set(self._node_names) - set(node_frames)
        if missing_nodes and not allow_partial:
            raise ModelNotTrainedError(
                "No output training data was provided for nodes "
                f"{sorted(missing_nodes)}."
            )

        new_models: dict[str, Model] = {}
        new_surrogates: dict[str, dict[str, SurrogateProtocol]] = {}
        new_feasibility: dict[str, _MultiStepGPClassifier] = {}
        new_feasibility_feature_names: dict[str, dict[str, str]] = {}
        output_frames: dict[str, dict[str, pd.DataFrame]] = {}
        fit_searchspaces: dict[str, SearchSpace] = {}

        # Validate both independent views before any expensive fitting.
        for node_name, frame in (*node_frames.items(), *feasibility_frames.items()):
            node = self.dag.node_by_name[node_name]
            if not frame.columns.is_unique:
                raise IncompatibilityError(
                    f"Training columns for node '{node_name}' must be unique."
                )
            input_names = self.dag.context.input_names(node)
            missing_inputs = set(input_names) - set(frame.columns)
            if missing_inputs:
                raise IncompatibilityError(
                    f"Training data for node '{node_name}' is missing columns: "
                    f"{sorted(missing_inputs)}."
                )
            # Node data is validated against the parameter ranges at ingestion, with
            # the batch's tolerance setting, so a refit must not enforce them again.
            _validate_node_inputs(
                node,
                frame[list(input_names)],
                numerical_measurements_must_be_within_tolerance=not fed_from_node_data,
                context=self.dag.context,
            )
        for node_name, frame in feasibility_frames.items():
            node = self.dag.node_by_name[node_name]
            required = {
                *self.dag.context.input_names(node),
                *node.output_names,
                *(
                    condition.column
                    for condition in node.feasibility_conditions
                    if condition.column not in node.output_names
                ),
                FEASIBILITY_LABEL,
                MEASUREMENT_ID,
                EXPERIMENT_ID,
                EXPERIMENT_GROUP,
            }
            missing = required - set(frame.columns)
            if missing:
                raise IncompatibilityError(
                    f"Raw feasibility data for node '{node_name}' is missing columns: "
                    f"{sorted(missing)}. Fold-safe fitting requires raw labels and "
                    "outputs, plus stable measurement/group identities."
                )
            if (
                frame[MEASUREMENT_ID].isna().any()
                or frame[MEASUREMENT_ID].duplicated().any()
            ):
                raise IncompatibilityError(
                    f"Measurement identities for node '{node_name}' must be present "
                    "and unique."
                )
            for value in frame[FEASIBILITY_LABEL]:
                if pd.isna(value):
                    continue
                if not isinstance(value, (bool, np.bool_)):
                    raise IncompatibilityError(
                        f"Feasibility labels for node '{node_name}' must be Boolean."
                    )
            for identity in frame[EXPERIMENT_GROUP]:
                try:
                    hash(identity)
                except TypeError as ex:
                    raise IncompatibilityError(
                        f"Experiment-group identities for node '{node_name}' must be "
                        "hashable."
                    ) from ex
            derived_labels = _extract_feasibility_values(node, frame)
            for index, (derived, supplied) in enumerate(
                zip(derived_labels, frame[FEASIBILITY_LABEL])
            ):
                derived_missing = pd.isna(derived)
                supplied_missing = pd.isna(supplied)
                if derived_missing != supplied_missing or (
                    not derived_missing and bool(derived) != bool(supplied)
                ):
                    raise IncompatibilityError(
                        f"The feasibility label at row {index} for node '{node_name}' "
                        "does not match its raw condition values."
                    )
            for output_name in node.output_names:
                for value in frame[output_name].dropna():
                    _validate_finite_input(output_name, value, node_name)
            experiment_groups: dict[Any, Any] = {}
            for experiment_id, group in zip(
                frame[EXPERIMENT_ID], frame[EXPERIMENT_GROUP]
            ):
                try:
                    missing_experiment = bool(pd.isna(experiment_id))
                except (TypeError, ValueError):
                    missing_experiment = False
                if missing_experiment:
                    continue
                previous = experiment_groups.setdefault(experiment_id, group)
                if previous != group:
                    raise IncompatibilityError(
                        f"Measurements sharing experiment identity {experiment_id!r} "
                        f"for node '{node_name}' must share one cross-fit group."
                    )
        for node_name, frame in node_frames.items():
            node = self.dag.node_by_name[node_name]
            missing_outputs = set(node.output_names) - set(frame.columns)
            if missing_outputs:
                raise IncompatibilityError(
                    f"Training data for node '{node_name}' is missing output columns: "
                    f"{sorted(missing_outputs)}."
                )
            per_output_frames: dict[str, pd.DataFrame] = {}
            too_few_outputs: list[tuple[str, int]] = []
            for output_name in node.output_names:
                values = frame[output_name]
                present = values.notna()
                if not fed_from_node_data and not present.all():
                    missing_index = values.index[~present][0]
                    _validate_finite_input(
                        output_name, values.loc[missing_index], node_name
                    )
                for value in values.loc[present]:
                    _validate_finite_input(output_name, value, node_name)
                n_present = int(present.sum())
                if n_present < 2:
                    too_few_outputs.append((output_name, n_present))
                per_output_frames[output_name] = frame.loc[
                    present, [*self.dag.context.input_names(node), output_name]
                ].copy()
            if too_few_outputs and not allow_partial:
                output_name, count = too_few_outputs[0]
                raise ModelNotTrainedError(
                    f"Node '{node_name}' requires at least two training rows for "
                    f"output '{output_name}', got {count}."
                )
            if too_few_outputs and not initialize_priors:
                continue
            output_frames[node_name] = per_output_frames
            fit_searchspaces[node_name] = _make_local_searchspace(
                self.dag.context,
                node,
                frame[list(self.dag.context.input_names(node))],
            )
        for node_name in self._node_names:
            node = self.dag.node_by_name[node_name]

            if node_name not in node_frames or node_name not in fit_searchspaces:
                continue
            fitted_surrogates: dict[str, SurrogateProtocol] = {}
            retained_surrogates = self._node_surrogates.get(node_name, {})
            for output_name in node.output_names:
                target = self.dag.target_by_output.get(
                    output_name, NumericalTarget(name=output_name)
                )
                # Copy retained contexts as well as templates: even a late fit
                # failure must not mutate published state. Surrogate.fit retains
                # its cache. Sibling outputs share one complete input search space.
                fitted_surrogates[output_name] = cast(
                    SurrogateProtocol,
                    self._fit_local_model(
                        node,
                        output_frames[node_name][output_name],
                        target,
                        retained_surrogates.get(output_name, node.surrogate_model),
                        output_name,
                        fit_searchspace=fit_searchspaces[node_name],
                    ),
                )

            new_surrogates[node_name] = fitted_surrogates
            new_models[node_name] = _make_node_model(
                tuple(
                    fitted_surrogates[name].to_botorch() for name in node.output_names
                )
            )

        for node_name, frame in feasibility_frames.items():
            node = self.dag.node_by_name[node_name]
            classifier, feature_names = self._fit_cross_fitted_classifier(node, frame)
            new_feasibility[node_name] = classifier
            new_feasibility_feature_names[node_name] = feature_names

        # Publish atomically: the previous fit stays intact on any error.
        self._publish(
            new_models,
            new_surrogates,
            feasibility_models=new_feasibility,
            feasibility_feature_names=new_feasibility_feature_names,
        )

    def _fit_node_data_with_priors(
        self,
        node_frames: Mapping[str, pd.DataFrame],
        feasibility_frames: Mapping[str, pd.DataFrame],
    ) -> None:
        """Fit available outputs and initialize missing output and classifier models.

        Args:
            node_frames: Authoritative local output-training views for every node.
            feasibility_frames: Raw-label frames for exactly the configured nodes.
                One shared classifier per configured node is cross-fitted from the
                raw joint labels and numeric-output posterior features.
        """
        self._fit_node_frames(
            node_frames,
            allow_partial=True,
            fed_from_node_data=True,
            initialize_priors=True,
            feasibility_frames=feasibility_frames,
        )

    def fit_node_data(  # noqa: DOC502
        self,
        node_frames: Mapping[str, pd.DataFrame],
        *,
        allow_partial: bool = False,
        feasibility_frames: Mapping[str, pd.DataFrame] | None = None,
    ) -> None:
        """Fit the node models from pre-filtered per-node training data.

        This is the direct data path used by the multi-step layer, which
        decides how failed or partial evaluations enter the training data.
        Each frame must contain the node's own parameters, configured history
        columns, and every declared output column. NaN output values represent
        missing observations and are excluded from that output's independent fit.
        The provided data is treated as authoritative, so subsequent :meth:`fit`
        calls do not discard partial node observations.

        Every configured node uses one shared joint-success classifier. Its explicit
        raw ``feasibility_frames`` must include the internal joint label, output and
        auxiliary values, stable ``MEASUREMENT_ID`` and ``EXPERIMENT_GROUP`` columns,
        plus ``EXPERIMENT_ID`` (nullable when unavailable). These raw frames must not
        contain globally imputed output values. Numeric-output probability features
        are generated with experiment-grouped out-of-fold regression fits.

        Each call publishes a complete fitted context: nodes that are skipped
        because of missing or too little data are left unfitted, and their
        previously published models are discarded.

        Args:
            node_frames: Mapping from node name to training data in
                experimental representation.
            allow_partial: If ``True``, nodes without enough training rows are
                left unfitted instead of raising an error. This permits
                recording upstream-only evaluations before a complete network
                can be trained.
            feasibility_frames: Raw condition values, outputs, nullable joint labels,
                and stable measurement/group identities for every configured node,
                including explicitly empty frames. Unconfigured callers need not
                provide this argument.

        Raises:
            IncompatibilityError: If a frame is missing raw data or stable identities,
                if a label does not match its raw conditions, or if fold-safe fitting
                is not possible.
            ModelNotTrainedError: If a node output has fewer than two usable training
                rows and ``allow_partial`` is ``False``.
        """
        self._fit_node_frames(
            node_frames,
            allow_partial=allow_partial,
            fed_from_node_data=True,
            feasibility_frames=feasibility_frames,
        )

    def predict_feasibility(self, node_name: str, inputs: pd.DataFrame) -> pd.Series:
        """Predict joint node success from local inputs and output-GP features.

        Args:
            node_name: The configured DAG node.
            inputs: Full experimental local inputs, including configured real history.

        Returns:
            One probability per row with the original index and target name.

        Raises:
            IncompatibilityError: If the node is unknown, unconfigured, or inputs
                violate its local observation contract.
            ModelNotTrainedError: If a fitted classifier requires unavailable output
                models.
        """
        node = _get_node(self.dag, node_name)
        if not node.feasibility_conditions:
            raise IncompatibilityError(
                f"Node '{node_name}' has no feasibility conditions."
            )
        _validate_node_inputs(node, inputs, context=self.dag.context)
        if not self._has_feasibility_prediction(node_name):
            raise ModelNotTrainedError(
                f"Node '{node_name}' lacks an output model needed to calculate its "
                "feasibility features."
            )
        searchspace = self._get_node_surrogate_searchspace(node_name)
        if searchspace is None:
            searchspace = _make_local_searchspace(
                self.dag.context, node, inputs[list(self.dag.context.input_names(node))]
            )
        import torch

        from baybe.utils.dataframe import to_tensor

        tensor = to_tensor(searchspace.transform(inputs, allow_extra=True)).unsqueeze(
            -2
        )
        with torch.no_grad():
            probabilities = (
                self._predict_feasibility_tensor(node_name, tensor)
                .detach()
                .cpu()
                .numpy()
            )
        return pd.Series(probabilities, index=inputs.index, name="success_probability")

    def _predict_feasibility_tensor(
        self,
        node_name: str,
        inputs_comp: Tensor,
    ) -> Tensor:
        """Predict the shared success probability while retaining gradients.

        Output-GP interval probabilities are features of one node-level classifier;
        they are not multiplied into its prediction. An unconditioned classifier is
        used until a known joint label is available. Logically empty numeric
        intervals remain impossible and return zero regardless of classifier state.

        Args:
            node_name: The configured DAG node.
            inputs_comp: Computational inputs with shape ``(batch..., 1, d)``.

        Returns:
            One probability per batch item, without the singleton q-axis.

        Raises:
            IncompatibilityError: If the node, input shape, fitted layout, or
                predicted probabilities are invalid.
        """
        if node_name not in self.dag.node_by_name:
            raise IncompatibilityError(f"Unknown DAG node '{node_name}'.")
        node = self.dag.node_by_name[node_name]
        if inputs_comp.ndim < 2 or inputs_comp.shape[-2] != 1:
            raise IncompatibilityError(
                "Feasibility scoring requires local action inputs with a singleton "
                f"q-axis, got shape {tuple(inputs_comp.shape)}."
            )
        if not node.feasibility_conditions:
            return inputs_comp.new_ones(inputs_comp.shape[:-2])
        import torch

        for output_name in node.output_names:
            lower_bounds = [
                cast(float, condition.threshold)
                for condition in node.feasibility_conditions
                if condition.column == output_name and condition.operator == ">="
            ]
            upper_bounds = [
                cast(float, condition.threshold)
                for condition in node.feasibility_conditions
                if condition.column == output_name and condition.operator == "<="
            ]
            if lower_bounds and upper_bounds and max(lower_bounds) > min(upper_bounds):
                return inputs_comp.new_zeros(inputs_comp.shape[:-2])

        classifier = self._feasibility_models.get(node_name)
        if classifier is None:
            raise ModelNotTrainedError(
                f"The feasibility model for node '{node_name}' has not been "
                "initialized."
            )
        searchspace = classifier._searchspace
        if searchspace is None:
            raise IncompatibilityError(
                f"The joint feasibility feature space for node '{node_name}' is "
                "unavailable."
            )
        feature_names = self._feasibility_feature_names[node_name]
        base_columns = tuple(
            column
            for column in searchspace.comp_rep_columns
            if column not in feature_names.values()
        )
        if len(base_columns) != inputs_comp.shape[-1]:
            raise IncompatibilityError(
                f"The fitted input layout for node '{node_name}' is incompatible with "
                f"candidate shape {tuple(inputs_comp.shape)}."
            )

        output_features: dict[str, Tensor] = {}
        if feature_names:
            if node_name not in self._node_models:
                raise ModelNotTrainedError(
                    f"Node '{node_name}' requires its output model to calculate "
                    "numeric feasibility features."
                )
            try:
                output_features = _output_probability_features(
                    node, self.get_node_model(node_name), inputs_comp
                )
            except Exception as ex:
                raise IncompatibilityError(
                    f"Could not calculate output-posterior feasibility features for "
                    f"node '{node_name}': {ex}"
                ) from ex
            ordered_features = [
                output_features[output_name] for output_name in feature_names
            ]
            classifier_inputs = torch.cat([inputs_comp, *ordered_features], dim=-1)

        source_columns = (
            self.get_node_model_columns(node_name)
            if node_name in self._node_models
            else base_columns
        )
        source_indices = {column: index for index, column in enumerate(source_columns)}
        if set(source_indices) != set(base_columns):
            raise IncompatibilityError(
                f"The fitted input columns for node '{node_name}' do not match its "
                "classifier feature layout."
            )
        feature_to_output = {
            feature_name: output_name
            for output_name, feature_name in feature_names.items()
        }
        if searchspace.comp_rep_columns:
            classifier_inputs = torch.cat(
                [
                    output_features[feature_to_output[column]]
                    if column in feature_to_output
                    else inputs_comp[
                        ..., source_indices[column] : source_indices[column] + 1
                    ]
                    for column in searchspace.comp_rep_columns
                ],
                dim=-1,
            )
        else:
            classifier_inputs = inputs_comp
        if classifier._model is not None:
            reference = next(classifier.to_botorch().parameters())
            classifier_inputs = classifier_inputs.to(reference)
        probabilities = classifier.posterior_probabilities(classifier_inputs)
        expected_shape = inputs_comp.shape[:-1]
        if probabilities.numel() != math.prod(expected_shape):
            raise IncompatibilityError(
                f"The joint classifier for node '{node_name}' returned "
                f"{probabilities.numel()} probabilities for input shape "
                f"{tuple(inputs_comp.shape)}."
            )
        probabilities = probabilities.reshape(expected_shape).squeeze(-1)
        if (
            not torch.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise IncompatibilityError(
                f"The joint classifier for node '{node_name}' returned values outside "
                "the probability interval [0, 1]."
            )
        return probabilities

    def to_botorch(self) -> MultiStepModel:
        """Return the BoTorch representation, which is the model itself.

        Returns:
            The model itself, since it already is a BoTorch
            :class:`~botorch.models.model.Model`.
        """
        return self

    @override
    def posterior(  # noqa: DOC502
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Posterior:
        """Compute the joint posterior over the final node outputs.

        Args:
            X: Candidate configurations in computational representation, with
                shape ``(batch..., q, d)``.
            output_indices: Optional graph-output indices to expose, in the
                requested order; ``None`` exposes every graph output.
            observation_noise: Not supported.
            posterior_transform: Optional transform applied to the posterior.
            **kwargs: Additional arguments are rejected.

        Returns:
            A lazy posterior drawing complete network paths using the acquisition
            sampler's node seeds.

        Raises:
            ModelNotTrainedError: If the model is not fitted for all nodes.
            NotImplementedError: For unsupported posterior arguments.
            IncompatibilityError: If the input shape does not match the DAG or
                the output indices are invalid.
        """
        self._validate_posterior_arguments(
            X,
            output_indices,
            observation_noise,
            kwargs,
            allow_output_indices=True,
        )
        indices = None if output_indices is None else tuple(output_indices)
        posterior = _make_network_posterior(self, X, output_indices=indices)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def draw_network_samples(  # noqa: DOC502
        self,
        X: Tensor,
        n_paths: int,
        *,
        sampling_seed: int | None = None,
    ) -> Tensor:
        """Forward-sample complete network paths with deterministic randomness.

        Args:
            X: Global candidate configurations in computational representation.
            n_paths: Number of joint network sample paths.
            sampling_seed: Optional seed overriding the model's default seed.

        Returns:
            Samples with shape ``(n_paths, batch..., q, num_outputs)``.

        Raises:
            ValueError: If ``n_paths`` or ``sampling_seed`` is invalid.
            ModelNotTrainedError: If the DAG is not fully fitted.
            IncompatibilityError: If the input width is incompatible.
        """
        return self._draw_network_samples(
            X,
            n_paths,
            sampling_seed=sampling_seed,
        )[0]

    def draw_network_outcomes(
        self,
        X: Tensor,
        n_paths: int,
        *,
        sampling_seed: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample outputs and their conditional network success weights.

        Each node's feasibility probability uses the parent outputs from the same
        sampled path. Shared ancestors are counted once, not once per downstream
        branch. Each configured node uses its joint-success classifier, with numeric
        output posterior probabilities supplied as classifier features.

        Args:
            X: Global candidate configurations in computational representation.
            n_paths: Number of network paths.
            sampling_seed: Optional sampling seed.

        Returns:
            Output samples and network success weights, with matching path/batch/q
            axes and without an output axis on the weights.

        Raises:
            ModelNotTrainedError: If a fitted classifier requires unavailable output
                models.
        """
        if self.missing_feasibility_nodes:
            raise ModelNotTrainedError(
                "Missing feasibility models for nodes: "
                f"{list(self.missing_feasibility_nodes)}."
            )
        return self._draw_network_samples(
            X,
            n_paths,
            sampling_seed=sampling_seed,
            include_feasibility=True,
        )

    def _draw_network_samples(
        self,
        X: Tensor,
        n_paths: int,
        *,
        sampling_seed: int | None = None,
        include_feasibility: bool = False,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward-sample outputs and optionally accumulate feasibility weights."""
        self._validate_posterior_arguments(
            X, output_indices, observation_noise, posterior_kwargs or {}
        )
        seed = self.sampling_seed if sampling_seed is None else sampling_seed
        return _draw_network_paths(
            self,
            X,
            n_paths,
            sampling_seed=seed,
            include_feasibility=include_feasibility,
        )

    def get_node_model(self, node_name: str) -> Model:
        """Retrieve the fitted model for one node.

        Args:
            node_name: The DAG node whose model is requested.

        Returns:
            The already fitted BoTorch model for the node, exposing one or more
            outputs in the node's declared order.

        Raises:
            KeyError: If the node is not part of the DAG.
            ModelNotTrainedError: If the node has no fitted model.
        """
        if node_name not in self.dag.node_by_name:
            raise KeyError(node_name)
        try:
            return self._node_models[node_name]
        except KeyError as ex:
            raise ModelNotTrainedError(
                f"Node '{node_name}' has no fitted model."
            ) from ex

    def _get_node_surrogate_searchspace(self, node_name: str) -> SearchSpace | None:
        """Validate and retrieve the common retained output-fit search space."""
        node = self.dag.node_by_name[node_name]
        surrogates = self._node_surrogates.get(node_name)
        if not surrogates:
            return None
        if set(surrogates) != set(node.output_names):
            raise IncompatibilityError(
                f"The fitted surrogate outputs for node '{node_name}' do not match "
                f"the declared outputs {node.output_names}."
            )

        searchspaces: list[SearchSpace] = []
        for output_name in node.output_names:
            surrogate = surrogates[output_name]
            if not isinstance(surrogate, Surrogate):
                raise IncompatibilityError(
                    f"The fitted surrogate for output '{output_name}' of node "
                    f"'{node_name}' cannot transform local inputs."
                )
            searchspace = surrogate._searchspace
            if searchspace is None:
                raise IncompatibilityError(
                    f"The fitted surrogate for output '{output_name}' of node "
                    f"'{node_name}' has no local fit search space."
                )
            searchspaces.append(searchspace)

        shared = searchspaces[0]
        if any(
            searchspace != shared
            or searchspace.comp_rep_columns != shared.comp_rep_columns
            for searchspace in searchspaces[1:]
        ):
            raise IncompatibilityError(
                f"The fitted output surrogates for node '{node_name}' do not share "
                "one local fit search space."
            )
        return shared

    def get_node_model_columns(self, node_name: str) -> tuple[str, ...]:  # noqa: DOC502
        """Retrieve the computational input columns of a node model.

        Args:
            node_name: The DAG node whose input order is requested.

        Returns:
            The computational input columns in model order.

        Raises:
            KeyError: If the node is not part of the DAG.
        """
        node = self.dag.node_by_name[node_name]
        searchspace = self._get_node_surrogate_searchspace(node_name)
        if searchspace is not None:
            return searchspace.comp_rep_columns

        # Models supplied through ``from_node_models`` have no retained BayBE fit
        # search space. Preserve their historical input-order contract only for
        # that advanced entry point.
        outputs = {
            feature.name: 0.0
            for feature in self.dag.context._features(node)
            if feature.parameter is None
        }
        return _make_local_searchspace(
            self.dag.context, node, pd.DataFrame([outputs])
        ).comp_rep_columns

    def make_node_fantasies(
        self,
        node: MultiStepNode,
        inputs: Tensor,
        num_fantasies: int,
        sampling_seed: int,
    ) -> MultiStepModel:
        """Create one network model with a batched fantasy node.

        BoTorch samples the selected node's predictive observation distribution as
        one joint output vector and conditions a temporary local model for each
        fantasy sample. The campaign model and real training data remain untouched,
        and no fitting is performed.

        Args:
            node: The node whose action is fantasized.
            inputs: The complete local node input in fitted computational
                representation, with shape ``(batch..., 1, d_node)``.
            num_fantasies: Number of temporary fantasy observations to draw.
            sampling_seed: Base seed for the deterministic fantasy samplers.

        Returns:
            A temporary network model whose selected node is batched over the
            fantasy and input batch dimensions.

        Raises:
            IncompatibilityError: If the node model cannot condition fantasies or the
                local inputs do not match its computational input shape.
        """
        import torch
        from botorch.models.model import Model
        from botorch.models.utils.assorted import fantasize as fantasize_flag
        from botorch.posteriors.torch import TorchPosterior
        from botorch.sampling.get_sampler import get_sampler
        from botorch.settings import propagate_grads
        from gpytorch.settings import fast_pred_var

        node_model = self.get_node_model(node.name)
        if not hasattr(node_model, "condition_on_observations"):
            raise IncompatibilityError(
                f"The fitted model of node '{node.name}' does not support fantasy "
                "conditioning."
            )

        model_columns = self.get_node_model_columns(node.name)
        if inputs.shape[-2] != 1 or inputs.shape[-1] != len(model_columns):
            raise IncompatibilityError(
                f"The fantasy input of node '{node.name}' must have shape "
                f"(batch..., 1, {len(model_columns)}), but received "
                f"{tuple(inputs.shape)}."
            )

        with fantasize_flag(), propagate_grads(True), fast_pred_var(False):
            posterior = cast(
                TorchPosterior,
                node_model.posterior(inputs, observation_noise=True),
            )
            sampler = get_sampler(
                posterior,
                torch.Size([num_fantasies]),
                seed=_derive_seed(sampling_seed, node.name, "action-fantasy"),
            )
            observations = sampler(posterior)
            try:
                conditioned = _condition_node_model(node_model, inputs, observations)
            except NotImplementedError as ex:
                raise IncompatibilityError(
                    f"The fitted model of node '{node.name}' does not support "
                    "fantasy conditioning."
                ) from ex
        node_models = dict(self._node_models)
        node_models[node.name] = cast(Model, conditioned)
        fantasy_model = self.__class__(
            dag=self.dag,
            num_network_samples=self.num_network_samples,
            sampling_seed=self.sampling_seed,
        )
        fantasy_model._publish(
            node_models,
            self._node_surrogates,
            feasibility_models=self._feasibility_models,
            feasibility_feature_names=self._feasibility_feature_names,
        )
        return fantasy_model

    def sample_node_observations(
        self,
        node_name: str,
        inputs: Tensor,
        num_samples: int,
        sampling_seed: int,
    ) -> Tensor:
        """Draw hypothetical observations of several local node executions.

        The samples come from the node model's predictive observation
        distribution, including observation noise and the correlations between
        all supplied local inputs. They are joint draw paths, not independent
        draws that would erase the GP correlation.

        Args:
            node_name: The node whose executions are sampled.
            inputs: Complete local inputs with shape ``(num_points, 1, d)`` in
                fitted computational representation.
            num_samples: Number of joint observation paths to draw.
            sampling_seed: Seed for the deterministic predictive draw.

        Returns:
            Hypothetical observations with shape ``(num_samples, num_points)`` for a
            single-output node or ``(num_samples, num_points, num_outputs)`` for a
            multi-output node.

        Raises:
            IncompatibilityError: If the node is unknown, the input shape is
                invalid, the model does not support noisy posterior samples, or
                the sampled shape is unsupported.
        """
        if node_name not in self.dag.node_by_name:
            raise IncompatibilityError(f"Node '{node_name}' is not part of the DAG.")
        if inputs.dim() != 3 or inputs.shape[-2] != 1:
            raise IncompatibilityError(
                f"Pending inputs of node '{node_name}' must have shape "
                f"(num_points, 1, d), but received {tuple(inputs.shape)}."
            )
        if num_samples < 1:
            raise ValueError(
                f"At least one pending sample is required, got {num_samples=}."
            )

        import torch
        from botorch.posteriors.torch import TorchPosterior
        from botorch.sampling.get_sampler import get_sampler

        node_model = self.get_node_model(node_name)
        with torch.no_grad():
            posterior = cast(
                TorchPosterior,
                node_model.posterior(inputs.squeeze(-2), observation_noise=True),
            )
            sampler = get_sampler(
                posterior,
                torch.Size([num_samples]),
                seed=_derive_seed(sampling_seed, node_name, "pending-observation"),
            )
            samples = sampler(posterior)
        if samples.shape != (num_samples, inputs.shape[0], node_model.num_outputs):
            raise IncompatibilityError(
                f"Sampling pending executions of node '{node_name}' produced shape "
                f"{tuple(samples.shape)}, expected a"
                f"joint q={inputs.shape[0]} axis and "
                f"{node_model.num_outputs} output(s)."
            )
        return samples.squeeze(-1) if node_model.num_outputs == 1 else samples

    def condition_on_node_observations(
        self, observations: Mapping[str, tuple[Tensor, Tensor]]
    ) -> MultiStepModel:
        """Create a model with hypothetical node observations added.

        No real training data is changed and no hyperparameter fitting is
        performed. Each node model is conditioned on the supplied observations,
        and the resulting node models are composed into a new temporary network
        model. The current model remains untouched.

        Args:
            observations: Mapping from node name to ``(inputs, outputs)``. Inputs
                must have shape ``(num_points, 1, d)`` in fitted computational
                representation; outputs must have shape ``(num_points,)`` or
                ``(num_points, 1)`` for single-output nodes, and
                ``(num_points, num_outputs)`` for multi-output nodes.

        Returns:
            A temporary network model conditioned on the observations.

        Raises:
            IncompatibilityError: If a node is unknown, a model cannot be
                conditioned, or an input or output shape is invalid.
        """
        if not observations:
            return self

        from botorch.models.model import Model

        node_models = dict(self._node_models)
        for name, (inputs, outputs) in observations.items():
            if name not in self.dag.node_by_name:
                raise IncompatibilityError(
                    f"Hypothetical observations provided for unknown node '{name}'."
                )
            node_model = self.get_node_model(name)
            if not hasattr(node_model, "condition_on_observations"):
                raise IncompatibilityError(
                    f"The fitted model of node '{name}' does not support "
                    "conditioning on hypothetical observations."
                )
            if inputs.dim() != 3 or inputs.shape[-2] != 1:
                raise IncompatibilityError(
                    f"Hypothetical inputs of node '{name}' must have shape "
                    f"(num_points, 1, d), but received {tuple(inputs.shape)}."
                )
            if node_model.num_outputs == 1:
                if outputs.shape not in {(inputs.shape[0],), (inputs.shape[0], 1)}:
                    raise IncompatibilityError(
                        f"Hypothetical outputs of node '{name}' must have shape "
                        f"({inputs.shape[0]},) or ({inputs.shape[0]}, 1), but received "
                        f"{tuple(outputs.shape)}."
                    )
                outputs_to_condition = outputs.reshape(-1, 1)
            else:
                expected_shape = (inputs.shape[0], node_model.num_outputs)
                if outputs.shape != expected_shape:
                    raise IncompatibilityError(
                        f"Hypothetical outputs of node '{name}' must have shape "
                        f"{expected_shape}, but received {tuple(outputs.shape)}."
                    )
                outputs_to_condition = outputs
            try:
                conditioned = _condition_node_model(
                    node_model, inputs.squeeze(-2), outputs_to_condition
                )
            except NotImplementedError as ex:
                raise IncompatibilityError(
                    f"The fitted model of node '{name}' does not support "
                    "conditioning on hypothetical observations."
                ) from ex
            node_models[name] = cast(Model, conditioned)

        conditioned_model = self.__class__(
            dag=self.dag,
            num_network_samples=self.num_network_samples,
            sampling_seed=self.sampling_seed,
        )
        conditioned_model._publish(
            node_models,
            self._node_surrogates,
            feasibility_models=self._feasibility_models,
            feasibility_feature_names=self._feasibility_feature_names,
        )
        return conditioned_model

    def condition_on_pending(
        self,
        pending: tuple[_PendingAction, ...],
        *,
        num_fantasies: int,
        sampling_seed: int,
    ) -> tuple[MultiStepModel, ...]:
        """Create one conditioned network model per joint pending fantasy.

        Pending executions of a node are sampled jointly so that their GP
        correlations and observation noise are retained. Nodes are independent
        given their local data, so joint network fantasies combine node samples
        by index instead of enumerating their Cartesian product.

        Args:
            pending: The reserved local executions.
            num_fantasies: Number of joint pending outcome paths.
            sampling_seed: Base seed for the deterministic predictive draws.

        Returns:
            One conditioned model per pending fantasy. Without pending actions,
            the original model is returned as the only element.

        Raises:
            IncompatibilityError: If a pending node is unknown or a local input
                cannot be transformed.
        """
        if not pending:
            return (self,)

        grouped: dict[str, list[_PendingAction]] = {}
        for action in pending:
            grouped.setdefault(action.node, []).append(action)

        node_inputs: dict[str, Tensor] = {}
        node_samples: dict[str, Tensor] = {}
        for index, (node_name, actions) in enumerate(grouped.items()):
            node = self.dag.node_by_name.get(node_name)
            if node is None:
                raise IncompatibilityError(
                    f"Pending action references unknown node '{node_name}'."
                )
            frame = pd.DataFrame(
                [dict(action.inputs) for action in actions],
                columns=list(self.dag.context.input_names(node)),
            )
            inputs = self.transform_node_inputs(node_name, frame).unsqueeze(-2)
            node_inputs[node_name] = inputs
            node_samples[node_name] = self.sample_node_observations(
                node_name,
                inputs,
                num_fantasies,
                sampling_seed + 2 * index,
            )

        return tuple(
            self.condition_on_node_observations(
                {
                    node_name: (
                        node_inputs[node_name],
                        node_samples[node_name][fantasy_index],
                    )
                    for node_name in grouped
                }
            )
            for fantasy_index in range(num_fantasies)
        )

    def _transform_local_inputs(
        self,
        inputs: pd.DataFrame,
        searchspace: SearchSpace | None,
        model: Model,
        *,
        node_name: str,
    ) -> Tensor:
        """Transform experimental local inputs into fitted model inputs.

        Args:
            inputs: The node's own parameters and configured historical features in
                experimental representation.
            searchspace: The common search space used to fit the local outputs.
            model: The fitted BoTorch model supplying device and dtype.
            node_name: The node name used in error messages.

        Returns:
            A computational input tensor in the fitted column order and with the
            model's device and dtype.

        Raises:
            IncompatibilityError: If the fitted search space is unavailable.
        """
        if searchspace is None:
            raise IncompatibilityError(
                f"The fitted surrogate for node '{node_name}' has no local fit "
                f"search space."
            )
        comp_rep = searchspace.transform(inputs)
        reference = next(model.parameters())
        from baybe.utils.dataframe import to_tensor

        return to_tensor(comp_rep).to(reference)

    def transform_node_inputs(self, node_name: str, inputs: pd.DataFrame) -> Tensor:
        """Transform experimental local inputs into fitted model inputs.

        Args:
            node_name: The DAG node whose inputs are transformed.
            inputs: The node's own parameters and configured historical features in
                experimental representation.

        Returns:
            A computational input tensor in the fitted column order and with the
            node model's device and dtype.

        Raises:
            IncompatibilityError: If the fitted surrogate is unavailable or the
                transformed inputs are inconsistent with the node model.
        """
        node_model = self.get_node_model(node_name)
        searchspace = self._get_node_surrogate_searchspace(node_name)
        if searchspace is None:
            raise IncompatibilityError(
                f"The fitted surrogate for node '{node_name}' cannot transform "
                f"local inputs."
            )

        return self._transform_local_inputs(
            inputs,
            searchspace,
            node_model,
            node_name=node_name,
        )

    def _validate_posterior_arguments(
        self,
        X: Tensor,
        output_indices: list[int] | None,
        observation_noise: bool | Tensor,
        kwargs: Mapping[str, Any],
        *,
        allow_output_indices: bool = False,
    ) -> None:
        """Validate the shared posterior arguments.

        Args:
            X: Candidate configurations with shape ``(batch..., q, d)``.
            output_indices: Optional graph-output indices.
            observation_noise: Optional observation noise; only ``False`` is
                supported.
            kwargs: Additional keyword arguments; none are supported.
            allow_output_indices: Whether restricted graph-output selection is
                supported by the calling posterior path.

        Raises:
            NotImplementedError: If an unsupported argument is provided.
            IncompatibilityError: If the candidate tensor has too few
                dimensions, the wrong computational-representation width, or
                invalid output indices.
        """
        if output_indices is not None:
            if not allow_output_indices:
                raise NotImplementedError(
                    "Output indices are not supported by this sampling path."
                )
            if not output_indices:
                raise IncompatibilityError(
                    "At least one output index must be requested."
                )
            if any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= self.num_outputs
                for index in output_indices
            ):
                raise IncompatibilityError(
                    f"Output indices must be integers in "
                    f"[0, {self.num_outputs}), got {output_indices}."
                )
            if len(set(output_indices)) != len(output_indices):
                raise IncompatibilityError(
                    f"Output indices must be unique, got {output_indices}."
                )
        import torch

        if isinstance(observation_noise, torch.Tensor) or observation_noise:
            raise NotImplementedError(
                "Observation noise is not supported by the multi-step model."
            )
        if kwargs:
            raise NotImplementedError(
                f"Unsupported posterior arguments: {sorted(kwargs)}."
            )
        if X.dim() < 2:
            raise IncompatibilityError(
                "Expected candidate configurations with at least two dimensions, "
                f"but received a tensor with {X.dim()} dimension(s)."
            )

        n_input = len(self.dag._configuration_space.comp_rep_columns)
        if X.shape[-1] != n_input:
            raise IncompatibilityError(
                f"Expected candidate configurations with {n_input} comp-rep "
                f"dimensions, but received {X.shape[-1]}."
            )


def _snapshot_model(model: MultiStepModel) -> MultiStepModel:
    """Retain the published fit context without copying or evaluating node models."""
    snapshot = copy(model)
    snapshot._modules = model._modules.copy()
    snapshot._node_models = model._node_models.copy()
    snapshot._node_surrogates = {
        name: surrogates.copy() for name, surrogates in model._node_surrogates.items()
    }
    snapshot._feasibility_models = model._feasibility_models.copy()
    snapshot._feasibility_feature_names = {
        name: feature_names.copy()
        for name, feature_names in model._feasibility_feature_names.items()
    }
    return snapshot


@define(frozen=True, slots=False, eq=False)
class _NetworkPosteriorBase(_PosteriorBase):
    """Propagate samples from the local BoTorch posteriors through the DAG.

    Each network base sample contains one seed per node. The corresponding
    local BoTorch sampler owns the actual posterior-specific base samples.

    The BoTorch :class:`~botorch.posteriors.posterior.Posterior` base is added by
    :func:`_get_network_posterior_class` so that importing this module does not
    load BoTorch.
    """

    model: MultiStepModel = field(converter=_snapshot_model)
    """The fitted network supplying local models and input bookkeeping."""

    X: Tensor = field()
    """Candidate configurations in computational representation."""

    output_indices: tuple[int, ...] | None = field(default=None)
    """Optional graph-output indices to expose, in the requested order."""

    def __attrs_post_init__(self) -> None:
        """Validate the network context without evaluating any node posterior.

        Raises:
            IncompatibilityError: If candidate values are non-finite.
            ModelNotTrainedError: If any node model is missing.
        """
        import torch

        self.model._validate_fitted()
        if not torch.isfinite(self.X).all():
            raise IncompatibilityError("Network candidates must be finite.")

    @property
    @override
    def base_sample_shape(self) -> torch.Size:
        """The number of independent node-sampler seeds per network path."""
        import torch

        return torch.Size([self.model.num_nodes])

    @property
    @override
    def batch_range(self) -> tuple[int, int]:
        """No candidate batch axes are stored in the network seed tensor."""
        return (0, 0)

    @property
    def batch_shape(self) -> torch.Size:
        """The candidate batch shape, including any supplied fantasy axes."""
        return self.X.shape[:-2]

    @property
    @override
    def device(self) -> torch.device:
        """The device of the candidate configurations."""
        return self.X.device

    @property
    @override
    def dtype(self) -> torch.dtype:
        """The dtype of the candidate configurations."""
        return self.X.dtype

    @property
    def mean(self) -> Tensor:
        """The deterministic Monte-Carlo estimate of the final-output means."""
        return self._moment_samples.mean(dim=0)

    @cached_property
    def _moment_samples(self) -> Tensor:
        """The shared network draws used to estimate posterior moments."""
        import torch

        return self.draw_outcomes(
            torch.Size([self.model.num_network_samples]),
            sampling_seed=self.model.sampling_seed,
        )[0]

    @property
    def variance(self) -> Tensor:
        """The sample variance of the final outputs, or zero for a single draw."""
        return self._moment_samples.var(
            dim=0, unbiased=self._moment_samples.shape[0] > 1
        )

    def _draw_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
        *,
        include_feasibility: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Propagate local samples and optionally compute success weights.

        Args:
            sample_shape: The independent network-sample axes.
            base_samples: Integer node-sampler seeds for each network path.
            include_feasibility: Whether to accumulate local success probabilities.

        Returns:
            Complete network outputs and the corresponding success weights.

        Raises:
            IncompatibilityError: If the base samples have incompatible shapes or
                non-finite values, or local node sampling is incompatible.
        """
        import torch

        expected_shape = sample_shape + self.base_sample_shape
        if base_samples.shape != expected_shape or base_samples.dtype != torch.long:
            raise IncompatibilityError(
                "Network base samples must be integer node-sampler seeds with "
                f"shape {tuple(expected_shape)}, got {tuple(base_samples.shape)} "
                f"and dtype {base_samples.dtype}. Use the network's BoTorch sampler."
            )
        if (base_samples < 0).any():
            raise IncompatibilityError(
                "Network node-sampler seeds must be nonnegative."
            )
        n_paths = sample_shape.numel()
        if n_paths < 1:
            raise IncompatibilityError("The network requires at least one sample path.")
        samples, weights = _propagate_network_samples(
            self.model,
            self.X,
            base_samples.reshape(n_paths, self.model.num_nodes),
            include_feasibility=include_feasibility,
        )
        full_shape = (
            sample_shape + self.X.shape[:-1] + torch.Size([self.model.num_outputs])
        )
        samples = samples.reshape(full_shape)
        if self.output_indices is not None:
            samples = samples[..., list(self.output_indices)]
        return (
            samples,
            weights.reshape(sample_shape + self.X.shape[:-1]),
        )

    @override
    def _extended_shape(self, sample_shape: torch.Size | None = None) -> torch.Size:
        import torch

        shape = torch.Size() if sample_shape is None else sample_shape
        n_outputs = (
            self.model.num_outputs
            if self.output_indices is None
            else len(self.output_indices)
        )
        return shape + self.X.shape[:-1] + torch.Size([n_outputs])

    def draw_outcomes(
        self,
        sample_shape: torch.Size,
        *,
        sampling_seed: int | None = None,
        include_feasibility: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Draw complete network paths and optional conditional success weights.

        Args:
            sample_shape: Shape of the independent network draws.
            sampling_seed: Optional seed for the network sampler.
            include_feasibility: Whether to accumulate local success probabilities.

        Returns:
            Final-output draws and success weights with matching sample/batch/q axes.
        """
        sampler = _make_network_sampler(self, sample_shape, seed=sampling_seed)
        return self._draw_from_base_samples(
            sample_shape,
            _make_network_base_samples(self, sample_shape, sampler.seed),
            include_feasibility=include_feasibility,
        )

    @override
    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        import torch

        shape = torch.Size([1]) if sample_shape is None else sample_shape
        return self.draw_outcomes(shape)[0]

    @override
    def rsample_from_base_samples(
        self, sample_shape: torch.Size, base_samples: Tensor
    ) -> Tensor:
        return self._draw_from_base_samples(sample_shape, base_samples)[0]


def _make_network_sampler(
    posterior: _NetworkPosteriorBase,
    sample_shape: torch.Size,
    *,
    seed: int | None = None,
) -> MCSampler:
    """Create a sampler that delegates each local draw to BoTorch.

    Args:
        posterior: The network posterior for sampler dispatch.
        sample_shape: The independent network-sample axes.
        seed: Optional seed for the node-sampler seeds.

    Returns:
        A sampler over complete network paths.
    """
    return _get_network_sampler_class()(sample_shape=sample_shape, seed=seed)


def _make_network_base_samples(
    posterior: Posterior, sample_shape: torch.Size, seed: int
) -> Tensor:
    """Draw independent, reproducible seeds for local node samplers."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(
        0, 2**31, sample_shape + posterior.base_sample_shape, generator=generator
    )


@lru_cache(maxsize=1)
def _get_network_sampler_class() -> type[MCSampler]:
    """Create the BoTorch sampler that coordinates local node samplers."""
    from botorch.sampling.base import MCSampler

    class _NetworkSampler(MCSampler):
        """Retain independent node seeds for reproducible network paths."""

        base_samples: Tensor | None

        def _construct_base_samples(self, posterior: Posterior) -> None:
            """Create a fixed seed for every path and node."""
            shape = self.sample_shape + posterior.base_sample_shape
            if self.base_samples is None or self.base_samples.shape != shape:
                self.register_buffer(
                    "base_samples",
                    _make_network_base_samples(posterior, self.sample_shape, self.seed),
                )

        @override
        def forward(self, posterior: Posterior) -> Tensor:
            self._construct_base_samples(posterior)
            assert self.base_samples is not None
            return posterior.rsample_from_base_samples(
                sample_shape=self.sample_shape, base_samples=self.base_samples
            )

        @override
        def _update_base_samples(
            self, posterior: Posterior, base_sampler: MCSampler
        ) -> None:
            self._instance_check(base_sampler)
            self._construct_base_samples(posterior)
            assert self.base_samples is not None
            previous = cast("Tensor | None", base_sampler.base_samples)
            if previous is not None:
                if self.base_samples.shape != previous.shape:
                    raise IncompatibilityError(
                        "Cannot reuse network base samples with different shapes."
                    )
                self.base_samples = previous.detach().clone()

    return _NetworkSampler


@lru_cache(maxsize=1)
def _get_network_posterior_class() -> type[_NetworkPosteriorBase]:
    """Create the BoTorch posterior subclass on first use.

    Returns:
        A posterior class combining the network implementation with BoTorch's
        :class:`~botorch.posteriors.posterior.Posterior` interface.
    """
    from botorch.posteriors.posterior import Posterior
    from botorch.sampling.get_sampler import GetSampler

    class _NetworkPosterior(_NetworkPosteriorBase, Posterior):
        """A BoTorch-compatible lazy network posterior."""

    _NetworkPosterior.__qualname__ = "_NetworkPosterior"
    GetSampler.register(_NetworkPosterior)(_make_network_sampler)
    return _NetworkPosterior


def _make_network_posterior(
    model: MultiStepModel,
    X: Tensor,
    *,
    output_indices: tuple[int, ...] | None = None,
) -> _NetworkPosteriorBase:
    """Instantiate the lazy network posterior.

    Args:
        model: The fitted network supplying local models and input bookkeeping.
        X: Candidate configurations in computational representation.
        output_indices: Optional graph-output indices to expose, in order.

    Returns:
        A posterior over complete network paths, optionally restricted to the
        requested graph outputs.
    """
    return _get_network_posterior_class()(model, X, output_indices=output_indices)


def _derive_seed(base_seed: int, node_name: str, purpose: str) -> int:
    """Separate local observations from complete network-path randomness.

    Args:
        base_seed: The model- or policy-level seed.
        node_name: The node whose observations are sampled.
        purpose: The observation sampling domain.

    Returns:
        A stable unsigned 32-bit seed.
    """
    payload = f"{base_seed}:{node_name}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _sample_node_posterior(model: Model, inputs: Tensor, sampling_seed: int) -> Tensor:
    """Draw a joint local output using the posterior's native BoTorch sampler.

    Args:
        model: The fitted node model, which may bundle multiple outputs.
        inputs: Node inputs with shape ``(*batch_shape, q, d)``.
        sampling_seed: Seed for the native local sampler.

    Returns:
        Node samples with shape ``(*batch_shape, q, num_outputs)``.

    Raises:
        IncompatibilityError: If no native sampler is available or the resulting
            sample shape is unsupported.
    """
    import torch
    from botorch.posteriors.torch import TorchPosterior
    from botorch.sampling.get_sampler import get_sampler

    posterior = model.posterior(inputs, observation_noise=False)
    try:
        sampler = get_sampler(
            cast(TorchPosterior, posterior), torch.Size(), seed=sampling_seed
        )
        samples = sampler(posterior)
    except NotImplementedError as ex:
        raise IncompatibilityError(
            f"The posterior of node model '{type(model).__name__}' does not "
            "support the native BoTorch sampling required by the network."
        ) from ex

    expected_shape = (*inputs.shape[:-1], model.num_outputs)
    if tuple(samples.shape) != expected_shape:
        raise IncompatibilityError(
            f"Sampling the fitted model of type '{type(model).__name__}' produced "
            f"shape {tuple(samples.shape)}, expected {expected_shape}."
        )
    return samples


def _draw_network_paths(
    model: MultiStepModel,
    X: Tensor,
    n_paths: int,
    *,
    sampling_seed: int,
    include_feasibility: bool,
) -> tuple[Tensor, Tensor]:
    """Forward-sample the DAG with one native sampler over complete network paths.

    Args:
        model: The fitted network model.
        X: Validated candidates with shape ``(batch..., q, d)``.
        n_paths: The number of network paths.
        sampling_seed: The base seed for all node draws.
        include_feasibility: Whether to accumulate network success weights.

    Returns:
        Output samples with shape ``(n_paths, *batch_shape, q, num_outputs)`` and
        success weights with shape ``(n_paths, *batch_shape, q)``.

    Raises:
        ValueError: If ``n_paths`` or ``sampling_seed`` is invalid.
        ModelNotTrainedError: If a node model is missing.
        IncompatibilityError: If node features or fitted model batch shapes are
            incompatible.
    """
    import torch

    if not isinstance(n_paths, int) or isinstance(n_paths, bool) or n_paths < 1:
        raise ValueError(f"'n_paths' must be a positive integer, got {n_paths!r}.")
    if (
        not isinstance(sampling_seed, int)
        or isinstance(sampling_seed, bool)
        or sampling_seed < 0
    ):
        raise ValueError(
            f"'sampling_seed' must be a nonnegative integer, got {sampling_seed!r}."
        )
    posterior = _make_network_posterior(model, X)
    return posterior.draw_outcomes(
        torch.Size([n_paths]),
        sampling_seed=sampling_seed,
        include_feasibility=include_feasibility,
    )


def _propagate_network_samples(
    model: MultiStepModel,
    X: Tensor,
    base_samples: Tensor,
    *,
    include_feasibility: bool,
) -> tuple[Tensor, Tensor]:
    """Forward-propagate one complete DAG path per supplied base sample.

    Args:
        model: The fitted network context.
        X: Global candidate configurations in computational representation.
        base_samples: Local sampler seeds with shape ``(n_paths, n_nodes)``.
        include_feasibility: Whether to accumulate conditional success probabilities.

    Returns:
        Final-output draws and success weights with aligned path/batch/q axes.

    Raises:
        IncompatibilityError: If local features or node posteriors are
            incompatible with the requested network paths.
    """
    import torch

    full_index = model._full_comp_rep_index
    layouts: dict[
        str,
        tuple[dict[str, str], dict[str, tuple[str, int]], tuple[str, ...]],
    ] = {}
    for node_name in model._node_names:
        node = model.dag.node_by_name[node_name]
        parameter_columns = {column: column for column in node._own_comp_rep_columns}
        output_columns: dict[str, tuple[str, int]] = {}
        for feature in model.dag.context._features(node):
            if feature.parameter is None:
                owner_name = feature.path[-1]
                owner = model.dag.node_by_name[owner_name]
                try:
                    output_index = owner.output_names.index(feature.source)
                except ValueError as ex:
                    raise IncompatibilityError(
                        f"History feature '{feature.name}' refers to undeclared output "
                        f"'{feature.source}' of node '{owner_name}'."
                    ) from ex
                output_columns[feature.name] = (owner_name, output_index)
            else:
                local_parameter = feature.make_parameter()
                parameter_columns.update(
                    zip(
                        local_parameter.comp_rep_columns,
                        feature.parameter.comp_rep_columns,
                    )
                )
        layouts[node_name] = (
            parameter_columns,
            output_columns,
            model.get_node_model_columns(node_name),
        )
    paths: list[Tensor] = []
    weights: list[Tensor] = []
    for path_seeds in base_samples:
        samples: dict[str, Tensor] = {}
        success_weights = X.new_ones(X.shape[:-1])
        for node_index, node_name in enumerate(model._node_names):
            node = model.dag.node_by_name[node_name]
            features: dict[str, Tensor] = {}
            parameter_columns, output_columns, columns = layouts[node_name]
            for column, source in parameter_columns.items():
                try:
                    features[column] = X[..., full_index[source]].unsqueeze(-1)
                except KeyError as ex:
                    raise IncompatibilityError(
                        f"The computational-representation column '{source}' of node "
                        f"'{node_name}' is missing from the DAG-wide search space."
                    ) from ex
            for column, (owner_name, output_index) in output_columns.items():
                features[column] = samples[owner_name][
                    ..., output_index : output_index + 1
                ].to(X)
            missing = [column for column in columns if column not in features]
            if missing:
                raise IncompatibilityError(
                    f"The node-model features of node '{node_name}' are incomplete: "
                    f"{missing}."
                )
            node_x = torch.cat([features[column] for column in columns], dim=-1)

            if include_feasibility and node.feasibility_conditions:
                probabilities = model._predict_feasibility_tensor(
                    node_name,
                    node_x.reshape(-1, 1, node_x.shape[-1]),
                )
                success_weights = success_weights * probabilities.reshape(
                    success_weights.shape
                )

            samples[node_name] = _sample_node_posterior(
                model._node_models[node_name], node_x, int(path_seeds[node_index])
            )
        paths.append(
            torch.cat([samples[name] for name in model.dag.final_nodes], dim=-1)
        )
        weights.append(success_weights)

    return torch.stack(paths), torch.stack(weights)


class _AcquisitionModelAdapterBase(_ModelBase):
    """Expose a network model without attrs recursion in stock acquisition setup.

    The adapter exposes graph outputs in objective-target order, so stock
    acquisitions consume the same columns their objective describes.

    The BoTorch :class:`~botorch.models.model.Model` base is added by
    :func:`_get_acquisition_model_adapter_class` on first use.
    """

    def __init__(self, model: MultiStepModel, output_indices: tuple[int, ...]) -> None:
        super().__init__()
        self._model = model
        self._output_indices = output_indices

    @property
    @override
    def batch_shape(self) -> torch.Size:
        """Return the unbatched parameter shape of network models."""
        import torch

        return torch.Size()

    @property
    @override
    def num_outputs(self) -> int:
        """Return the number of objective-relevant network outputs."""
        return len(self._output_indices)

    @override
    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any = None,
    ) -> Posterior:
        """Delegate posterior computation with objective-ordered outputs."""
        indices = self._output_indices
        if output_indices is not None:
            if any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= len(indices)
                for index in output_indices
            ):
                raise IncompatibilityError(
                    f"Output indices must be integers in [0, {len(indices)}), "
                    f"got {output_indices}."
                )
            indices = tuple(indices[index] for index in output_indices)
        return self._model.posterior(
            X,
            output_indices=list(indices),
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )


@lru_cache(maxsize=1)
def _get_acquisition_model_adapter_class() -> type[_AcquisitionModelAdapterBase]:
    """Create the BoTorch acquisition adapter subclass on first use.

    Returns:
        An adapter class combining the network adapter with BoTorch's model base.
    """
    from botorch.models.model import Model

    class _AcquisitionModelAdapter(_AcquisitionModelAdapterBase, Model):
        """A BoTorch model adapter around a fitted network model."""

    _AcquisitionModelAdapter.__qualname__ = "_AcquisitionModelAdapter"
    return _AcquisitionModelAdapter


def _make_acquisition_model_adapter(model: MultiStepModel) -> Model:
    """Wrap a fitted network model for stock acquisition setup.

    Args:
        model: The fitted network model.

    Returns:
        A non-attrs BoTorch model exposing the posterior outputs in objective
        target order.
    """
    return _get_acquisition_model_adapter_class()(
        model, model.dag.objective_output_indices
    )


class _AcquisitionSurrogateAdapter:
    """Keep attrs-based acquisition collection from recursing into the model."""

    def __init__(self, model: MultiStepModel) -> None:
        self._model = model

    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Reject fitting through the acquisition-only adapter.

        Args:
            searchspace: The proposed search space.
            objective: The proposed objective.
            measurements: The proposed measurements.

        Raises:
            IncompatibilityError: Always; fit the network on its node data instead.
        """
        raise IncompatibilityError(
            "The acquisition adapter cannot fit the network. Fit the multi-step "
            "model on node data before constructing the acquisition."
        )

    def to_botorch(self) -> Model:
        """Return a non-attrs BoTorch adapter around the fitted network model."""
        return _make_acquisition_model_adapter(self._model)


@lru_cache(maxsize=1)
def _get_model_class() -> type[_MultiStepModelBase]:
    """Create the BoTorch network model subclass on first use.

    Returns:
        A model class combining the network implementation with BoTorch's model
        interface.
    """
    from botorch.models.model import Model

    class MultiStepModel(_MultiStepModelBase, Model):
        """A BoTorch model exposing the final outputs of a DAG as a posterior.

        The model owns one node model per DAG node and composes them by forward
        sampling. Outputs from one node are independently fitted and bundled when
        needed. Descendants see the sampled outputs of their
        parents, so the resulting posterior reflects the full network
        propagation rather than independent per-node predictions. Node models
        are fitted from per-node training frames or from a complete measurement
        frame, and fitted state is published atomically. Direct :meth:`fit` calls
        always consume their supplied measurements, including after node-data fits.

        The model implements
        :class:`~baybe.surrogates.base.SurrogateProtocol`, so the stock
        recommenders consume its output posterior like any other surrogate.
        Feasibility remains a separate channel for acquisition constraints and
        hypothetical observations.
        """

    MultiStepModel.__qualname__ = "MultiStepModel"
    return MultiStepModel


def __getattr__(name: str) -> Any:
    """Load the BoTorch model class only when it is actually requested."""
    if name == "MultiStepModel":
        value = _get_model_class()
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    MultiStepModel = _MultiStepModelBase


__all__ = ["MultiStepModel", "validate_network_measurements"]


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
