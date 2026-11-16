"""Single BoTorch model over a DAG of independently fitted node models.

The model owns one independently fitted model per DAG node and derives all node
input bookkeeping from the DAG itself. Network posteriors are formed by
Monte-Carlo propagation: every sample is one full network path, parent outputs
are drawn once and reused at forks, and the final outputs are returned jointly
over the q-axis. The result is exposed as a BoTorch
:class:`~botorch.posteriors.ensemble.EnsemblePosterior`, so the stock samplers
and acquisition functions consume it without a custom posterior class. Already
observed node outputs can replace the sampled path.

Node posteriors backed by base samples remain differentiable with respect to the
candidates; ensemble posteriors select precomputed member predictions and are
therefore not differentiable.

The classes are ``attrs`` classes at the boundary to BoTorch. Two adjustments
are required because :class:`torch.nn.Module` owns the instance state and its
identity: attribute assignment must use the module's ``__setattr__`` so that
attrs does not install its own (``setters.NO_OP``), and acquisition-argument
collection must not recursively convert the model into a dictionary.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Mapping
from copy import deepcopy
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from attrs import define, field, setters
from attrs.validators import ge, instance_of
from botorch.models.model import Model
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.posteriors.posterior import Posterior
from pandas.api.types import is_numeric_dtype
from torch import Tensor
from typing_extensions import override

from baybe.exceptions import IncompatibilityError, ModelNotTrainedError
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import SurrogateProtocol
from baybe.targets.numerical import NumericalTarget

from .dag import MultiStepDAG
from .node import MultiStepNode

if TYPE_CHECKING:
    from baybe.objectives.base import Objective
    from baybe.parameters.base import Parameter


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

    output_names = tuple(node.output for node in dag.nodes)
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

    non_numeric = [
        name for name in output_names if not is_numeric_dtype(measurements[name])
    ]
    if non_numeric:
        raise IncompatibilityError(
            f"Node output columns must be numeric, but columns {non_numeric} are not."
        )

    values = measurements[list(output_names)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [
            name
            for name, column in zip(
                output_names, measurements[list(output_names)].to_numpy(dtype=float).T
            )
            if not np.isfinite(column).all()
        ]
        raise IncompatibilityError(
            f"Node output columns must be finite, but columns {bad} contain "
            f"NaN or infinite values."
        )

    expected_targets = dag.final_target_names
    actual_targets = tuple(target.name for target in objective.targets)
    if actual_targets != expected_targets:
        raise IncompatibilityError(
            f"The objective targets {actual_targets} do not match the final node "
            f"targets {expected_targets}. The objective must be the DAG objective "
            f"and use the same target order."
        )


def _make_node_fit_searchspace(
    node: MultiStepNode, measurements: pd.DataFrame
) -> SearchSpace:
    """Build the fit-only search space of a node model.

    The measured outputs of a node's parents enter its model as fixed
    covariates: they are model inputs, but they are neither searchable nor
    bounded. Since a surrogate derives its kernel dimensions and input
    transformation from the search space it is fitted on, the covariates are
    represented as continuous parameters whose bounds span their observed
    values. These parameters exist only in the fitting context; the node itself
    remains searchable through its own parameters alone.

    Args:
        node: The node to fit.
        measurements: The node training frame containing the node's own
            parameters and the parent output columns.

    Returns:
        The search space describing the node model input.
    """
    from baybe.parameters.numerical import NumericalContinuousParameter

    covariates: list[Parameter] = []
    for name in node.parent_output_names:
        values = measurements[name].to_numpy(dtype=float)
        lower = float(np.min(values))
        upper = float(np.max(values))
        # A degenerate range would make the input transformation divide by zero
        if upper == lower:
            upper = lower + 1.0
        covariates.append(NumericalContinuousParameter(name, bounds=(lower, upper)))

    return SearchSpace.from_product([*node.searchspace.parameters, *covariates])


@define(eq=False, repr=False, slots=False, on_setattr=setters.NO_OP)
class MultiStepModel(Model, SurrogateProtocol):
    """A BoTorch model exposing the final outputs of a DAG as a posterior.

    The model owns one independently fitted model per node and composes them by
    forward sampling. Descendants see the sampled outputs of their parents, so
    the resulting posterior reflects the full network propagation rather than
    independent per-node predictions. Node models are fitted from per-node
    training frames or from a complete measurement frame, and fitted state is
    published atomically. Data passed through :meth:`fit_node_data` is treated
    as authoritative, so a later :meth:`fit` call on such a model is a no-op.

    The model implements
    :class:`~baybe.surrogates.base.SurrogateProtocol`, so the stock recommenders
    consume it like any other surrogate.

    Args:
        dag: The DAG topology.
        num_network_samples: Number of network paths stored in the ensemble
            posterior. The same paths provide the posterior moments and the
            samples used by the acquisition function.
        sampling_seed: Base seed for the deterministic network draws.
    """

    dag: MultiStepDAG = field(validator=instance_of(MultiStepDAG))
    """The DAG topology."""

    num_network_samples: int = field(default=512, validator=ge(1))
    """Number of network paths stored in the ensemble posterior."""

    sampling_seed: int = field(default=0, validator=ge(0))
    """Base seed for the deterministic network draws."""

    _node_models: dict[str, Model] = field(init=False, factory=dict, eq=False)
    """Mapping from node name to a fitted BoTorch model. Each model is
    single-output and operates on the node's computational representation."""

    _node_surrogates: dict[str, SurrogateProtocol] = field(
        init=False, factory=dict, eq=False
    )
    """Mapping from node name to the fitted surrogate it was derived from.

    The retained surrogates cache their fit context, so refitting unchanged node
    data is a no-op. They are bound to the node configurations of the DAG the
    model was created with.
    """

    _fed_from_node_data: bool = field(init=False, default=False, eq=False)
    """Whether the node models were fitted from authoritative node data.

    Models in this state are owned by the multi-step layer. When the stock
    recommender calls :meth:`fit` with a complete-row projection, the call is
    intentionally a no-op because the projection would discard valid
    upstream-only evaluations.
    """

    _node_names: tuple[str, ...] = field(init=False)
    """The node names in topological order."""

    def __attrs_post_init__(self) -> None:
        """Initialize the module state and the empty node-model registry."""
        super().__init__()

        self._node_names = self.dag.topological_order
        # Register the node models with the torch module so that device and
        # dtype conversions propagate to them. The mapping itself stays a plain
        # dict because node names are arbitrary and need not be valid module
        # keys.
        self._node_modules = torch.nn.ModuleList()

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
        outside the multi-step layer. No fitting is performed, and the provided
        models are treated as authoritative, so a subsequent :meth:`fit` call is
        a no-op.

        Args:
            dag: The DAG topology.
            node_models: Mapping covering every DAG node.
            num_network_samples: Number of network paths stored in the ensemble
                posterior.
            sampling_seed: Base seed for the deterministic network draws.

        Returns:
            The model publishing the provided node models.

        Raises:
            IncompatibilityError: If a DAG node has no model or a model belongs
                to an unknown node.
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
        model._publish(node_models, {}, fed_from_node_data=True)
        return model

    def _publish(
        self,
        node_models: Mapping[str, Model],
        node_surrogates: Mapping[str, SurrogateProtocol],
        *,
        fed_from_node_data: bool,
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
            fed_from_node_data: Whether the models are authoritative node models
                rather than derived from a complete-row projection.

        Raises:
            IncompatibilityError: If a model or surrogate belongs to an unknown
                node, or a model does not produce exactly one output.
        """
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
        multi_output = [
            name for name, model in node_models.items() if model.num_outputs != 1
        ]
        if multi_output:
            raise IncompatibilityError(
                f"Node models must produce exactly one output, but models for "
                f"nodes {sorted(multi_output)} produce multiple outputs."
            )

        # Build the torch module list before touching any state so that a
        # registration failure cannot leave a partially published fit behind.
        # Re-registering keeps device and dtype conversions propagating to the
        # fitted models; only fitted nodes appear, so partial fits stay usable.
        node_modules = torch.nn.ModuleList(
            [node_models[name] for name in self._node_names if name in node_models]
        )

        self._node_models = dict(node_models)
        self._node_surrogates = dict(node_surrogates)
        self._fed_from_node_data = fed_from_node_data
        self._node_modules = node_modules

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
        """The nodes with a fitted node model, in topological order."""
        return tuple(name for name in self._node_names if name in self._node_models)

    @property
    def missing_nodes(self) -> tuple[str, ...]:
        """The nodes without a fitted node model, in topological order."""
        return tuple(name for name in self._node_names if name not in self._node_models)

    def _node_input_columns(self, node_name: str) -> tuple[str, ...]:
        """Return the experimental training columns of a node.

        Args:
            node_name: The DAG node.

        Returns:
            The node's own parameters and parent output columns, in fitted
            order.
        """
        return self.dag.node_by_name[node_name].model_input_names

    @cached_property
    def _full_comp_rep_index(self) -> dict[str, int]:
        """The indices of the DAG-wide comp-rep columns inside the candidate tensor.

        Returns:
            Mapping from computational-representation column name to its index.
        """
        return {
            column: index
            for index, column in enumerate(self.dag.full_searchspace.comp_rep_columns)
        }

    @cached_property
    def _node_model_columns(self) -> dict[str, tuple[str, ...]]:
        """The model input columns of each node, in fitted order.

        The node's own tunable inputs form the computational representation. The
        parent outputs are fixed features appended after them; they are not
        search-space dimensions.

        Returns:
            Mapping from node name to the computational-representation columns
            consumed by that node's model.
        """
        return {
            node.name: (*node.searchspace.comp_rep_columns, *node.parent_output_names)
            for node in self.dag.nodes
        }

    def _validate_fitted(self) -> None:
        """Validate that every node has a fitted model.

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

    def unfixed_nodes(
        self, fixed_outputs: Mapping[str, Tensor] | None = None
    ) -> tuple[str, ...]:
        """Return the nodes whose outputs are sampled, in topological order.

        Args:
            fixed_outputs: Optional observed node outputs. These nodes consume
                no base samples.

        Returns:
            The node names that are not covered by ``fixed_outputs``.

        Raises:
            IncompatibilityError: If a fixed output refers to an unknown node.
        """
        fixed = {} if fixed_outputs is None else fixed_outputs
        unknown = set(fixed) - set(self._node_names)
        if unknown:
            raise IncompatibilityError(
                f"Fixed outputs provided for unknown nodes: {sorted(unknown)}."
            )
        return tuple(name for name in self._node_names if name not in fixed)

    # The documented exceptions are raised by validate_network_measurements and
    # _fit_node_frames.
    @override
    def fit(  # noqa: DOC502
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Fit the node models on complete DAG measurement rows.

        This is the stock-BayBE entry point. If the model was already fitted
        from authoritative node data, the call is a documented no-op because a
        complete-row projection would discard valid upstream-only evaluations.
        A previous complete-row fit does not block refitting: subsequent calls
        replace the fitted context with the new measurements.

        Args:
            searchspace: The DAG-wide search space.
            objective: The objective assembled from the final node targets.
            measurements: Complete DAG evaluations in experimental
                representation.

        Raises:
            IncompatibilityError: If the data contract is violated.
            ModelNotTrainedError: If a node has no training rows.
        """
        if self._fed_from_node_data:
            # The multi-step layer already fitted the node models from the
            # authoritative per-node views. Re-fitting from the complete-row
            # projection would discard valid upstream-only evaluations.
            return

        validate_network_measurements(self.dag, searchspace, objective, measurements)

        node_frames: dict[str, pd.DataFrame] = {}
        for node in self.dag.nodes:
            input_columns = self._node_input_columns(node.name)
            df_node = measurements[list(input_columns)].copy()
            df_node[node.output] = measurements[node.output].to_numpy()
            node_frames[node.name] = df_node

        self._fit_node_frames(
            node_frames, allow_partial=False, fed_from_node_data=False
        )

    def _fit_node_frames(
        self,
        node_frames: Mapping[str, pd.DataFrame],
        *,
        allow_partial: bool,
        fed_from_node_data: bool,
    ) -> None:
        """Fit the node models and publish the resulting context atomically.

        Args:
            node_frames: Mapping from node name to training data in
                experimental representation.
            allow_partial: If ``True``, nodes without enough training rows are
                left unfitted instead of raising an error.
            fed_from_node_data: Whether the frames are authoritative node data
                rather than a complete-row projection.

        Raises:
            IncompatibilityError: If a frame is missing columns.
            ModelNotTrainedError: If a node has fewer than two rows and
                ``allow_partial`` is ``False``.
        """
        unknown = set(node_frames) - set(self._node_names)
        if unknown:
            raise IncompatibilityError(
                f"Training data provided for unknown nodes: {sorted(unknown)}."
            )

        new_models: dict[str, Model] = {}
        new_surrogates: dict[str, SurrogateProtocol] = {}

        for node_name in self._node_names:
            node = self.dag.node_by_name[node_name]
            input_columns = self._node_input_columns(node_name)

            if node_name not in node_frames:
                if allow_partial:
                    continue
                raise ModelNotTrainedError(
                    f"No training data was provided for node '{node_name}'."
                )
            df_node = node_frames[node_name]
            missing = [
                column
                for column in (*input_columns, node.output)
                if column not in df_node.columns
            ]
            if missing:
                raise IncompatibilityError(
                    f"Training data for node '{node_name}' is missing columns: "
                    f"{missing}."
                )
            if len(df_node) < 2:
                if allow_partial:
                    continue
                raise ModelNotTrainedError(
                    f"Node '{node_name}' requires at least two training rows, but "
                    f"only {len(df_node)} row(s) were provided."
                )

            df_node = df_node[list(input_columns) + [node.output]].copy()
            fit_searchspace = _make_node_fit_searchspace(node, df_node)
            target = self.dag.target_by_output.get(
                node.output, NumericalTarget(name=node.output)
            )

            # Reuse the retained surrogate so that its fit cache can no-op when
            # the node data is unchanged. The template is copied only on the
            # first fit of a node, so the DAG configurations are never fitted in
            # place.
            surrogate = self._node_surrogates.get(node_name)
            if surrogate is None:
                surrogate = deepcopy(node.surrogate_model)
            try:
                surrogate.fit(fit_searchspace, target.to_objective(), df_node)
            except Exception:
                # Discard a possibly half-fitted instance so that the next fit
                # starts from the template instead of trusting a stale cache.
                self._node_surrogates.pop(node_name, None)
                raise

            new_surrogates[node_name] = surrogate
            new_models[node_name] = surrogate.to_botorch()

        # Publish atomically: the previous fit stays intact on any error.
        self._publish(new_models, new_surrogates, fed_from_node_data=fed_from_node_data)

    def fit_node_data(  # noqa: DOC502
        self,
        node_frames: Mapping[str, pd.DataFrame],
        *,
        allow_partial: bool = False,
    ) -> None:
        """Fit the node models from pre-filtered per-node training data.

        This is the direct data path used by the multi-step layer, which
        decides how failed or partial evaluations enter the training data.
        Each frame must contain the node's own parameters, parent output
        columns, and the node output column. The
        provided data is treated as authoritative, so a subsequent :meth:`fit`
        call is a no-op.

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

        Raises:
            IncompatibilityError: If a frame is missing columns.
            ModelNotTrainedError: If a node has fewer than two rows and
                ``allow_partial`` is ``False``.
        """
        self._fit_node_frames(
            node_frames, allow_partial=allow_partial, fed_from_node_data=True
        )

    @override
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
            output_indices: Not supported.
            observation_noise: Not supported.
            posterior_transform: Optional transform applied to the posterior.
            **kwargs: Additional arguments are rejected.

        Returns:
            The network posterior over the final node outputs, represented as an
            :class:`~botorch.posteriors.ensemble.EnsemblePosterior` over the
            network paths.

        Raises:
            ModelNotTrainedError: If the model is not fitted for all nodes.
            NotImplementedError: For unsupported posterior arguments.
            IncompatibilityError: If the input shape does not match the DAG.
        """
        self._validate_fitted()
        self._validate_posterior_arguments(X, output_indices, observation_noise, kwargs)

        posterior: Posterior = EnsemblePosterior(
            self.draw_network_samples(X, self.num_network_samples).movedim(0, -3)
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def posterior_conditioned(
        self,
        X: Tensor,
        fixed_outputs: Mapping[str, Tensor],
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Posterior:
        """Compute the posterior with some node outputs treated as observed.

        The outputs in ``fixed_outputs`` are not sampled but taken as given,
        and all other nodes are sampled as usual. This fixes already observed
        node outputs and resamples the rest; it is neither Bayesian
        conditioning nor a refit of the node models.

        Args:
            X: Candidate configurations with shape ``(batch..., q, d)``.
            fixed_outputs: Observed node outputs, keyed by node name. Values
                must broadcast against ``(n_paths, batch..., q, 1)``.
            output_indices: Not supported.
            observation_noise: Not supported.
            posterior_transform: Optional transform applied to the posterior.
            **kwargs: Additional arguments are rejected.

        Returns:
            The conditioned network posterior, represented as an
            :class:`~botorch.posteriors.ensemble.EnsemblePosterior` over the
            network paths.

        Raises:
            ModelNotTrainedError: If the model is not fitted for all nodes.
            NotImplementedError: For unsupported posterior arguments.
            IncompatibilityError: If a fixed output refers to an unknown node
                or the input shape does not match the DAG.
        """
        self._validate_fitted()
        self._validate_posterior_arguments(X, output_indices, observation_noise, kwargs)
        unknown = set(fixed_outputs) - set(self._node_names)
        if unknown:
            raise IncompatibilityError(
                f"Fixed outputs provided for unknown nodes: {sorted(unknown)}."
            )

        posterior: Posterior = EnsemblePosterior(
            self.draw_network_samples(
                X, self.num_network_samples, fixed_outputs=fixed_outputs
            ).movedim(0, -3)
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def draw_network_samples(
        self,
        X: Tensor,
        n_paths: int,
        fixed_outputs: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Forward-sample the DAG with deterministic base samples.

        One base sample is drawn per unfixed node, in topological order.

        Args:
            X: Candidate configurations with shape ``(batch..., q, d)``.
            n_paths: Number of joint network sample paths.
            fixed_outputs: Optional observed node outputs replacing sampled
                values. The leading dimension must match ``n_paths`` or be
                broadcastable.

        Returns:
            Samples with shape ``(n_paths, batch..., q, num_outputs)``.

        Raises:
            ValueError: If ``n_paths`` is not positive.
            ModelNotTrainedError: If the DAG is not fully fitted.
            IncompatibilityError: If the input width or a fixed output is
                incompatible.
        """
        if n_paths < 1:
            raise ValueError(f"'n_paths' must be positive, got {n_paths}.")
        self._validate_fitted()
        self._validate_posterior_arguments(X, None, False, {})

        unfixed = self.unfixed_nodes(fixed_outputs)
        generator = torch.Generator(device=X.device)
        generator.manual_seed(self.sampling_seed)
        base_samples = torch.randn(
            (n_paths, *X.shape[:-2], X.shape[-2], len(unfixed)),
            generator=generator,
            device=X.device,
            dtype=X.dtype,
        )
        return self._draw_network_samples(X, base_samples, fixed_outputs)

    def _draw_network_samples(
        self,
        X: Tensor,
        base_samples: Tensor,
        fixed_outputs: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Forward-sample the DAG using the provided per-node base samples.

        Args:
            X: Candidate configurations with shape ``(batch..., q, d)``.
            base_samples: Standard-normal base samples with shape
                ``(n_paths, batch..., q, n_unfixed_nodes)``. The last dimension
                is indexed in topological order of the unfixed nodes.
            fixed_outputs: Optional observed node outputs replacing sampled
                values.

        Returns:
            Samples with shape ``(n_paths, batch..., q, num_outputs)``.

        Raises:
            IncompatibilityError: If the base-sample shape, input width, or a
                fixed output is incompatible.
        """
        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        n_paths = base_samples.shape[0]
        fixed_outputs = {} if fixed_outputs is None else fixed_outputs
        unfixed = self.unfixed_nodes(fixed_outputs)
        expected_shape = torch.Size([n_paths, *batch_shape, q, len(unfixed)])
        if base_samples.shape != expected_shape:
            raise IncompatibilityError(
                f"Expected network base samples of shape "
                f"{tuple(expected_shape)}, but received "
                f"{tuple(base_samples.shape)}."
            )
        positions = {name: i for i, name in enumerate(unfixed)}
        batch_size = torch.Size(batch_shape).numel()
        samples: dict[str, Tensor] = {}
        full_index = self._full_comp_rep_index

        for name in self._node_names:
            node = self.dag.node_by_name[name]

            if name in fixed_outputs:
                target_shape = torch.Size([n_paths, *batch_shape, q, 1])
                value = fixed_outputs[name].to(X)
                # Accept values with any prefix of leading singleton axes, e.g.
                # ``(1, 1, q, 1)`` for an action fantasy that is broadcast over
                # all network paths.
                while value.dim() < len(target_shape):
                    value = value.unsqueeze(0)
                try:
                    samples[name] = value.expand(target_shape)
                except RuntimeError as ex:
                    raise IncompatibilityError(
                        f"Fixed output for node '{name}' with shape "
                        f"{tuple(fixed_outputs[name].shape)} cannot be broadcast to "
                        f"{tuple(target_shape)}."
                    ) from ex
                continue

            # Assemble the node-model features in the order used during fitting:
            # own inputs from the candidates, parent outputs from the sampled
            # network paths.
            features: dict[str, Tensor] = {}
            for column in node.searchspace.comp_rep_columns:
                index = full_index.get(column)
                if index is None:
                    raise IncompatibilityError(
                        f"The computational-representation column '{column}' of node "
                        f"'{name}' is missing from the DAG-wide search space."
                    )
                value = X[..., index]
                features[column] = value.expand(n_paths, *value.shape).unsqueeze(-1)
            for parent in node.parents:
                features[parent.output] = samples[parent.name].to(X)

            columns = self._node_model_columns[name]
            missing = [column for column in columns if column not in features]
            if missing:
                raise IncompatibilityError(
                    f"The node-model features of node '{name}' are incomplete: "
                    f"{missing}."
                )
            node_x = torch.cat([features[column] for column in columns], dim=-1)

            # Flatten the network-path and t-batch dimensions so that every
            # node model sees a plain ``(t, q, d)`` input. This keeps node
            # models usable that do not support multiple t-batch dimensions,
            # such as ensemble models.
            node_x_flat = node_x.reshape(n_paths * batch_size, q, node_x.shape[-1])
            node_base = base_samples[..., positions[name]].reshape(
                n_paths * batch_size, q
            )

            model = self._node_models[name]
            node_posterior = model.posterior(node_x_flat)
            node_samples = self._sample_node_posterior(
                name, node_posterior, node_base, X
            )
            samples[name] = node_samples.reshape(n_paths, *batch_shape, q, 1)

        return torch.cat([samples[name] for name in self.dag.final_nodes], dim=-1)

    def _validate_posterior_arguments(
        self,
        X: Tensor,
        output_indices: list[int] | None,
        observation_noise: bool | Tensor,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Validate the shared posterior arguments.

        Args:
            X: Candidate configurations with shape ``(batch..., q, d)``.
            output_indices: Optional output indices; only ``None`` is supported.
            observation_noise: Optional observation noise; only ``False`` is
                supported.
            kwargs: Additional keyword arguments; none are supported.

        Raises:
            NotImplementedError: If an unsupported argument is provided.
            IncompatibilityError: If the candidate tensor has too few
                dimensions or the wrong computational-representation width.
        """
        if output_indices is not None:
            raise NotImplementedError(
                "Output indices are not supported by the multi-step model."
            )
        if isinstance(observation_noise, Tensor) or observation_noise:
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

        n_input = len(self.dag.full_searchspace.comp_rep_columns)
        if X.shape[-1] != n_input:
            raise IncompatibilityError(
                f"Expected candidate configurations with {n_input} comp-rep "
                f"dimensions, but received {X.shape[-1]}."
            )

    def _sample_node_posterior(
        self,
        node_name: str,
        posterior: Posterior,
        base_samples: Tensor,
        reference: Tensor,
    ) -> Tensor:
        """Draw one deterministic sample path from a node posterior.

        Posteriors that expose base samples consume the standard-normal base
        samples directly. Ensemble posteriors, which do not expose latent base
        samples, instead map the base samples to deterministic member indices,
        preserving reproducibility while remaining compatible with
        non-Gaussian node models.

        Args:
            node_name: The node whose posterior is sampled.
            posterior: The node posterior.
            base_samples: Standard-normal base samples with shape ``(n_rows, q)``.
            reference: A tensor providing device and dtype.

        Returns:
            Samples with shape ``(n_rows, q, 1)``.

        Raises:
            IncompatibilityError: If the posterior supports neither base
                samples nor ensemble index sampling, or the base-sample shape
                is incompatible.
        """
        if isinstance(posterior, EnsemblePosterior):
            return self._sample_ensemble_posterior(
                node_name, posterior, base_samples[:, 0], reference
            )

        try:
            expected = posterior.base_sample_shape
        except NotImplementedError as ex:
            raise IncompatibilityError(
                f"The posterior of node '{node_name}' of type "
                f"'{type(posterior).__name__}' does not support deterministic "
                f"sampling. Expected either base-sample support or an "
                f"'{EnsemblePosterior.__name__}'."
            ) from ex

        if expected.numel() == 0:
            node_base = torch.empty(0, device=reference.device, dtype=reference.dtype)
        elif expected != base_samples.shape:
            if expected.numel() != base_samples.numel():
                raise IncompatibilityError(
                    f"The posterior of node '{node_name}' expects base samples of "
                    f"shape {tuple(expected)}, but {tuple(base_samples.shape)} were "
                    f"provided."
                )
            node_base = base_samples.reshape(expected)
        else:
            node_base = base_samples

        try:
            samples = posterior.rsample_from_base_samples(torch.Size(), node_base)
        except NotImplementedError as ex:
            raise IncompatibilityError(
                f"The posterior of node '{node_name}' of type "
                f"'{type(posterior).__name__}' advertises base-sample support but "
                f"does not implement 'rsample_from_base_samples'."
            ) from ex

        expected = torch.Size([base_samples.shape[0], base_samples.shape[-1], 1])
        if samples.shape != expected:
            if samples.numel() != expected.numel():
                raise IncompatibilityError(
                    f"The posterior of node '{node_name}' produced samples of "
                    f"shape {tuple(samples.shape)}, but {tuple(expected)} was "
                    f"expected."
                )
            samples = samples.reshape(expected)
        return samples

    def _sample_ensemble_posterior(
        self,
        node_name: str,
        posterior: EnsemblePosterior,
        base_samples: Tensor,
        reference: Tensor,
    ) -> Tensor:
        """Select deterministic ensemble members for a node posterior.

        Each network path selects exactly one ensemble member from its base
        sample. That member applies to all ``q`` candidates of the path, so the
        joint q-batch keeps a single member per path.

        Args:
            node_name: The node whose posterior is sampled.
            posterior: The ensemble posterior.
            base_samples: Standard-normal base samples with shape ``(n_rows,)``.
            reference: A tensor providing device and dtype.

        Returns:
            Samples with shape ``(n_rows, q, 1)``.

        Raises:
            IncompatibilityError: If the posterior batch shape or ensemble
                weights do not match the expected t-batch layout.
        """
        n_rows = base_samples.shape[0]
        expected_batch_shape = torch.Size([n_rows])
        values = posterior.values
        if values.shape[:-3] != expected_batch_shape:
            raise IncompatibilityError(
                f"The ensemble posterior of node '{node_name}' has batch shape "
                f"{tuple(values.shape[:-3])}, but "
                f"{tuple(expected_batch_shape)} was expected for {n_rows} "
                f"network path(s)."
            )

        weights = posterior.weights
        if weights.ndim != 1 or weights.numel() != posterior.ensemble_size:
            raise IncompatibilityError(
                f"The ensemble posterior of node '{node_name}' has weights of "
                f"shape {tuple(weights.shape)}, but a one-dimensional weight "
                f"vector with {posterior.ensemble_size} entries was expected."
            )

        cumulative = weights.to(reference).cumsum(dim=-1)
        uniform = 0.5 * (1.0 + torch.erf(base_samples.to(reference) / math.sqrt(2.0)))
        indices = torch.searchsorted(cumulative, uniform).clamp(
            max=posterior.ensemble_size - 1
        )
        rows = torch.arange(n_rows, device=reference.device)
        return values[rows, indices]


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
