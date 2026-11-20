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
from collections.abc import Mapping
from copy import deepcopy
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

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
from baybe.recommenders.pure.experimental_feasibility import (
    _FeasibilityModelProtocol,
    _make_feasibility_model,
)
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import Surrogate, SurrogateProtocol
from baybe.targets.base import Target
from baybe.targets.binary import BinaryTarget
from baybe.targets.numerical import NumericalTarget

from ._columns import FEASIBILITY_LABEL
from .dag import MultiStepDAG
from .data import (
    MultiStepData,
    _get_node,
    _validate_finite_input,
    _validate_node_inputs,
)
from .node import MultiStepNode

if TYPE_CHECKING:
    from baybe.objectives.base import Objective


class _FittableModel(Protocol):
    """Protocol for local models that can be fitted in a modelling context."""

    __slots__ = ()

    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Fit the model to training data in a given modelling context.

        Args:
            searchspace: The search space on which the model operates.
            objective: The objective for which the model is trained.
            measurements: The measurements used for training.
        """


_ModelT = TypeVar("_ModelT", bound=_FittableModel)
"""Type of a local model that is copied before fitting."""


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
    invalid_outputs = ~np.isfinite(values).all(axis=0)
    if invalid_outputs.any():
        bad = np.asarray(output_names)[invalid_outputs].tolist()
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


@define(eq=False, repr=False, slots=False, on_setattr=setters.NO_OP)
class MultiStepModel(Model, SurrogateProtocol):
    """A BoTorch model exposing the final outputs of a DAG as a posterior.

    The model owns one independently fitted model per node and composes them by
    forward sampling. Descendants see the sampled outputs of their parents, so
    the resulting posterior reflects the full network propagation rather than
    independent per-node predictions. Node models are fitted from per-node
    training frames or from a complete measurement frame, and fitted state is
    published atomically. Data passed through :meth:`fit_node_data` is treated
    as authoritative, so a later :meth:`fit` call is a no-op.

    The model implements
    :class:`~baybe.surrogates.base.SurrogateProtocol`, so the stock recommenders
    consume its output posterior like any other surrogate. Feasibility remains a
    separate channel for acquisition constraints and hypothetical observations.
    """

    dag: MultiStepDAG = field(validator=instance_of(MultiStepDAG))
    """The DAG topology."""

    num_network_samples: int = field(default=512, validator=ge(1))
    """Number of paths used for posterior moments and acquisition samples."""

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

    _feasibility_models: dict[str, _FeasibilityModelProtocol] = field(
        init=False, factory=dict, eq=False
    )
    """Independent feasibility models retaining their own local fit search spaces."""

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
        feasibility_models: Mapping[str, _FeasibilityModelProtocol] | None = None,
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
            feasibility_models: Independent feasibility models to publish.

        Raises:
            IncompatibilityError: If a model or surrogate belongs to an unknown
                node, or a model does not produce exactly one output.
        """
        feasibility_models = {} if feasibility_models is None else feasibility_models
        unknown = set(feasibility_models) - {
            node.name for node in self.dag.nodes if node.feasibility_conditions
        }
        if unknown:
            raise IncompatibilityError(
                f"Feasibility models provided for unconfigured nodes: "
                f"{sorted(unknown)}."
            )
        if not all(
            isinstance(model, Surrogate) for model in feasibility_models.values()
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
        feasibility_modules = torch.nn.ModuleList(
            [
                cast(Surrogate, feasibility_models[name]).to_botorch()
                for name in self._node_names
                if name in feasibility_models
            ]
        )

        self._node_models = dict(node_models)
        self._node_surrogates = dict(node_surrogates)
        self._fed_from_node_data = fed_from_node_data
        self._node_modules = node_modules
        self._feasibility_models = dict(feasibility_models)
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
        """The nodes with a fitted node model, in topological order."""
        return tuple(name for name in self._node_names if name in self._node_models)

    @property
    def missing_nodes(self) -> tuple[str, ...]:
        """The nodes without a fitted node model, in topological order."""
        return tuple(name for name in self._node_names if name not in self._node_models)

    @property
    def missing_feasibility_nodes(self) -> tuple[str, ...]:
        """Configured nodes without a fitted feasibility model."""
        return tuple(
            name
            for name in self._node_names
            if self.dag.node_by_name[name].feasibility_conditions
            and name not in self._feasibility_models
        )

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

        The local fit search space sorts its discrete parameters and its
        continuous parameters alphabetically and concatenates both
        representations. Parent outputs enter that space as continuous local
        parameters, so they are sorted together with any continuous own
        parameters. Reproducing that order here keeps the network features
        aligned with the fitted model input.

        Returns:
            Mapping from node name to the computational-representation columns
            consumed by that node's model.
        """
        result: dict[str, tuple[str, ...]] = {}
        for node in self.dag.nodes:
            if node.inputs:
                discrete = node.searchspace.discrete.comp_rep_columns
                continuous = node.searchspace.continuous.comp_rep_columns
            else:
                discrete = ()
                continuous = ()
            local_continuous = tuple(sorted((*continuous, *node.parent_output_names)))
            result[node.name] = (*discrete, *local_continuous)
        return result

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
        searchspace: SearchSpace | None,
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
            IncompatibilityError: If the data contract is violated or no search
                space is given.
            ModelNotTrainedError: If a node has no training rows.
        """
        if searchspace is None:
            raise IncompatibilityError(
                f"'{self.__class__.__name__}' requires the DAG-wide search space."
            )
        if self._fed_from_node_data:
            # The multi-step layer already fitted the node models from the
            # authoritative per-node views. Re-fitting from the complete-row
            # projection would discard valid upstream-only evaluations.
            return

        validate_network_measurements(self.dag, searchspace, objective, measurements)
        if any(node.feasibility_conditions for node in self.dag.nodes):
            complete_data = MultiStepData(complete_measurements=measurements)
            complete_data.validate(self.dag)
        else:
            complete_data = MultiStepData(complete_measurements=measurements)

        self._fit_node_frames(
            {
                node.name: measurements[[*node.model_input_names, node.output]]
                for node in self.dag.nodes
            },
            allow_partial=False,
            fed_from_node_data=False,
            feasibility_frames={
                node.name: complete_data.node_feasibility_training_data(
                    self.dag, node.name
                )
                for node in self.dag.nodes
                if node.feasibility_conditions
            },
        )

    def _fit_local_model(
        self,
        node: MultiStepNode,
        frame: pd.DataFrame,
        target: Target,
        template: _ModelT,
        label: str,
    ) -> _ModelT:
        """Fit a copy of a local model on the node's training frame.

        The node's own parameters and its measured parent outputs jointly form a
        local synthetic fit search space. The parent outputs are continuous
        dimensions of that local space only; they never become part of the
        DAG-wide search space or a recommendation candidate. Fitting always
        operates on a deep copy, leaving retained contexts and node templates
        untouched.

        Args:
            node: The node whose local model is fitted.
            frame: Validated training data in experimental representation.
            target: The target whose objective the model is fitted on.
            template: The unfitted model template or retained fitted model.
            label: The target column of the local model.

        Returns:
            The independently fitted model copy.

        Raises:
            IncompatibilityError: If a parent output contains non-finite values.
        """
        from baybe.parameters.numerical import NumericalContinuousParameter

        measurements = frame[[*node.model_input_names, label]].copy()
        parent_parameters: list[NumericalContinuousParameter] = []
        for name in node.parent_output_names:
            values = measurements[name].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise IncompatibilityError(
                    f"Parent output '{name}' contains NaN or infinite values."
                )
            lower = float(np.min(values))
            upper = float(np.max(values))
            if lower == upper:
                # A degenerate range would make the input transformation divide by zero
                delta = max(abs(lower) * 0.5, 1.0)
                lower -= delta
                upper += delta
            parent_parameters.append(
                NumericalContinuousParameter(name, bounds=(lower, upper))
            )

        fit_searchspace = SearchSpace.from_product([*node.inputs, *parent_parameters])
        model = deepcopy(template)
        model.fit(fit_searchspace, target.to_objective(), measurements)
        return model

    def _fit_node_frames(
        self,
        node_frames: Mapping[str, pd.DataFrame],
        *,
        allow_partial: bool,
        fed_from_node_data: bool,
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
            feasibility_frames: Explicit raw-label frames for configured nodes.

        Raises:
            IncompatibilityError: If a frame is missing columns.
            ModelNotTrainedError: If a node has fewer than two rows and
                ``allow_partial`` is ``False``.
            IncompleteMeasurementsError: If raw feasibility labels are missing.
            InvalidTargetValueError: If feasibility labels violate the target contract.
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
        missing_nodes = set(self._node_names) - set(node_frames)
        if missing_nodes and not allow_partial:
            raise ModelNotTrainedError(
                "No output training data was provided for nodes "
                f"{sorted(missing_nodes)}."
            )

        new_models: dict[str, Model] = {}
        new_surrogates: dict[str, SurrogateProtocol] = {}
        new_feasibility: dict[str, _FeasibilityModelProtocol] = {}

        # Validate both independent views before any expensive fitting.
        for node_name, frame in (*node_frames.items(), *feasibility_frames.items()):
            node = self.dag.node_by_name[node_name]
            if len(frame) < 2 and not allow_partial:
                raise ModelNotTrainedError(
                    f"Node '{node_name}' requires at least two training rows "
                    "for each output and feasibility model."
                )
            if not frame.columns.is_unique:
                raise IncompatibilityError(
                    f"Training columns for node '{node_name}' must be unique."
                )
            missing_inputs = set(node.model_input_names) - set(frame.columns)
            if missing_inputs:
                raise IncompatibilityError(
                    f"Training data for node '{node_name}' is missing columns: "
                    f"{sorted(missing_inputs)}."
                )
            # Node data is validated against the parameter ranges at ingestion, with
            # the batch's tolerance setting, so a refit must not enforce them again.
            _validate_node_inputs(
                self.dag,
                node,
                frame[list(node.model_input_names)],
                numerical_measurements_must_be_within_tolerance=not fed_from_node_data,
            )
        for node_name, frame in node_frames.items():
            node = self.dag.node_by_name[node_name]
            if node.output not in frame:
                raise IncompatibilityError(
                    f"Training data for node '{node_name}' is missing output column "
                    f"'{node.output}'."
                )
            for value in frame[node.output]:
                _validate_finite_input(node.output, value, node_name)
        for node_name, frame in feasibility_frames.items():
            if FEASIBILITY_LABEL not in frame:
                raise IncompatibilityError(
                    f"Feasibility training data for node '{node_name}' is missing "
                    f"the derived label '{FEASIBILITY_LABEL}'."
                )
            for value in frame[FEASIBILITY_LABEL]:
                if not isinstance(value, (bool, np.bool_)):
                    raise IncompatibilityError(
                        f"Feasibility labels for node '{node_name}' must be Boolean."
                    )

        for node_name in self._node_names:
            node = self.dag.node_by_name[node_name]

            if node_name not in node_frames:
                continue
            df_node = node_frames[node_name]
            if len(df_node) < 2:
                continue

            target = self.dag.target_by_output.get(
                node.output, NumericalTarget(name=node.output)
            )
            # Copy retained contexts as well as templates: even a late fit failure
            # must not mutate published state. Surrogate.fit retains its cache.
            surrogate = self._fit_local_model(
                node,
                df_node,
                target,
                self._node_surrogates.get(node_name, node.surrogate_model),
                node.output,
            )

            new_surrogates[node_name] = surrogate
            new_models[node_name] = surrogate.to_botorch()

        for node_name, frame in feasibility_frames.items():
            node = self.dag.node_by_name[node_name]
            if len(frame) < 2:
                continue
            target = BinaryTarget(
                FEASIBILITY_LABEL, success_value=True, failure_value=False
            )
            template = self._feasibility_models.get(
                node_name, _make_feasibility_model(target)
            )
            new_feasibility[node_name] = self._fit_local_model(
                node, frame, target, template, FEASIBILITY_LABEL
            )

        # Publish atomically: the previous fit stays intact on any error.
        self._publish(
            new_models,
            new_surrogates,
            fed_from_node_data=fed_from_node_data,
            feasibility_models=new_feasibility,
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
        Each frame must contain the node's own parameters, parent output
        columns, and the node output column. The
        provided data is treated as authoritative, so subsequent :meth:`fit` calls
        do not discard partial node observations.

        Feasibility uses independent raw-label frames and requires at least two
        observations, even if all failed and no output model is available. It is
        independent of the successful-output model.

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
            feasibility_frames: Raw target labels and full local inputs for every
                configured node, including explicitly empty frames. Unconfigured
                callers need not provide this argument. Partial mode permits either
                output or feasibility models to remain unavailable until two rows exist.

        Raises:
            IncompatibilityError: If a frame is missing columns.
            ModelNotTrainedError: If a node has fewer than two rows and
                ``allow_partial`` is ``False``.
            IncompleteMeasurementsError: If raw feasibility labels are missing.
            InvalidTargetValueError: If feasibility labels violate the target contract.
        """
        self._fit_node_frames(
            node_frames,
            allow_partial=allow_partial,
            fed_from_node_data=True,
            feasibility_frames=feasibility_frames,
        )

    def predict_feasibility(self, node_name: str, inputs: pd.DataFrame) -> pd.Series:
        """Predict local feasibility without requiring an output or network model.

        Args:
            node_name: The configured DAG node.
            inputs: Full experimental local inputs, including real parent outputs.

        Returns:
            One probability per row with the original index and target name.

        Raises:
            IncompatibilityError: If the node is unknown, unconfigured, or inputs
                violate its local observation contract.
            ModelNotTrainedError: If no feasibility model has been fitted.
        """
        node = _get_node(self.dag, node_name)
        if not node.feasibility_conditions:
            raise IncompatibilityError(
                f"Node '{node_name}' has no feasibility conditions."
            )
        _validate_node_inputs(self.dag, node, inputs)
        if node_name not in self._feasibility_models:
            raise ModelNotTrainedError(
                f"Node '{node_name}' has no fitted feasibility model."
            )
        model = self._feasibility_models[node_name]
        surrogate = cast(Surrogate, model)
        tensor = self._transform_local_inputs(
            inputs,
            surrogate,
            self.get_node_model_columns(node_name),
            surrogate.to_botorch(),
            node_name=node_name,
        )
        with torch.no_grad():
            probabilities = model.posterior_probabilities(tensor).detach().cpu().numpy()
        return pd.Series(probabilities, index=inputs.index, name="success_probability")

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
        return self.posterior_conditioned(
            X,
            fixed_outputs={},
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

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

        Unfitted models, unsupported posterior arguments, and inputs that do
        not match the DAG are rejected by the validation performed before
        sampling.

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
        """
        self._validate_fitted()
        self._validate_posterior_arguments(X, output_indices, observation_noise, kwargs)

        posterior: Posterior = EnsemblePosterior(
            self.draw_network_samples(
                X, self.num_network_samples, fixed_outputs=fixed_outputs
            ).movedim(0, -3)
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def draw_network_samples(  # noqa: DOC502
        self,
        X: Tensor,
        n_paths: int,
        fixed_outputs: Mapping[str, Tensor] | None = None,
        *,
        sampling_seed: int | None = None,
    ) -> Tensor:
        """Forward-sample the DAG with deterministic random draws.

        One sample is drawn per unfixed node, in topological order.

        Args:
            X: Candidate configurations with shape ``(batch..., q, d)``.
            n_paths: Number of joint network sample paths.
            fixed_outputs: Optional observed node outputs replacing sampled
                values. The leading dimension must match ``n_paths`` or be
                broadcastable.
            sampling_seed: Optional seed overriding the model's default seed.

        Returns:
            Samples with shape ``(n_paths, batch..., q, num_outputs)``.

        Raises:
            ValueError: If ``n_paths`` is not positive.
            ValueError: If ``sampling_seed`` is not a nonnegative integer.
            ModelNotTrainedError: If the DAG is not fully fitted.
            IncompatibilityError: If the input width or a fixed output is
                incompatible.
        """
        return self._draw_network_samples(
            X,
            n_paths,
            fixed_outputs,
            sampling_seed=sampling_seed,
        )[0]

    def draw_network_outcomes(
        self,
        X: Tensor,
        n_paths: int,
        *,
        sampling_seed: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample successful outputs and their conditional network success weights.

        Each node's feasibility probability uses the parent outputs from the same
        sampled path. Shared ancestors are counted once, not once per downstream
        branch. Only real measurements enter the feasibility models; hypothetical
        outcomes are not conditioned on.

        Args:
            X: Global candidate configurations in computational representation.
            n_paths: Number of network paths.
            sampling_seed: Optional sampling seed.

        Returns:
            Output samples and network success weights, with matching path/batch/q
            axes and without an output axis on the weights.

        Raises:
            ModelNotTrainedError: If a configured feasibility model is unavailable.
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
        fixed_outputs: Mapping[str, Tensor] | None = None,
        *,
        sampling_seed: int | None = None,
        include_feasibility: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Forward-sample outputs and optionally accumulate feasibility weights."""
        if n_paths < 1:
            raise ValueError(f"'n_paths' must be positive, got {n_paths}.")
        seed = self.sampling_seed if sampling_seed is None else sampling_seed
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(
                f"'sampling_seed' must be a nonnegative integer, got {seed!r}."
            )
        self._validate_fitted()
        self._validate_posterior_arguments(X, None, False, {})

        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        fixed = self._prepare_fixed_outputs(X, fixed_outputs, n_paths)

        samples: dict[str, Tensor] = {}
        success_weights = X.new_ones((n_paths, *X.shape[:-1]))
        full_index = self._full_comp_rep_index
        nodes_by_name = self.dag.node_by_name

        for node_index, node_name in enumerate(self._node_names):
            if node_name in fixed:
                samples[node_name] = fixed[node_name]
                continue

            node = nodes_by_name[node_name]

            # Assemble the node-model features in the order used during fitting:
            # own inputs from the candidates, parent outputs from the sampled
            # network paths.
            features: dict[str, Tensor] = {}
            for column in node._own_comp_rep_columns:
                index = full_index.get(column)
                if index is None:
                    raise IncompatibilityError(
                        f"The computational-representation column '{column}' of node "
                        f"'{node_name}' is missing from the DAG-wide search space."
                    )
                value = X[..., index]
                features[column] = value.expand(n_paths, *value.shape).unsqueeze(-1)
            for parent in node.parents:
                features[parent.output] = samples[parent.name].to(X)

            columns = self._node_model_columns[node_name]
            missing = [column for column in columns if column not in features]
            if missing:
                raise IncompatibilityError(
                    f"The node-model features of node '{node_name}' are incomplete: "
                    f"{missing}."
                )
            node_x = torch.cat([features[column] for column in columns], dim=-1)

            if include_feasibility and node.feasibility_conditions:
                probabilities = self._feasibility_models[
                    node_name
                ].posterior_probabilities(node_x)
                success_weights = success_weights * probabilities.reshape(
                    node_x.shape[:-1]
                )

            node_model = self._node_models[node_name]
            # A fantasy-conditioned node model carries the fantasy batch axes.
            # Its batch shape must broadcast into the trailing candidate batch
            # axes so that BoTorch broadcasts it to the full path/batch shape.
            # Conditioned models may additionally carry a trailing singleton
            # output axis which broadcasts against any candidate axis.
            model_batch = tuple(node_model.batch_shape)
            if model_batch:
                n_model = len(model_batch)
                trailing = tuple(batch_shape[-n_model:])
                try:
                    broadcast = torch.broadcast_shapes(trailing, model_batch)
                except RuntimeError as ex:
                    raise IncompatibilityError(
                        f"The batch shape {model_batch} of the fitted model for node "
                        f"'{node_name}' does not broadcast into the trailing "
                        f"candidate batch axes {tuple(batch_shape)}."
                    ) from ex
                if broadcast != trailing:
                    raise IncompatibilityError(
                        f"The batch shape {model_batch} of the fitted model for node "
                        f"'{node_name}' does not broadcast into the trailing "
                        f"candidate batch axes {tuple(batch_shape)}."
                    )
                node_inputs = node_x
            else:
                # Flatten the path and candidate axes for ensemble models that
                # require a single input batch.
                node_inputs = node_x.reshape(
                    n_paths * batch_shape.numel(), q, node_x.shape[-1]
                )
            posterior = node_model.posterior(node_inputs, observation_noise=False)
            with torch.random.fork_rng():
                torch.manual_seed(seed + 2 * node_index)
                if isinstance(posterior, EnsemblePosterior):
                    # Select one member per path, shared across all t-batches.
                    draws = torch.multinomial(
                        posterior.weights, n_paths, replacement=True
                    )
                    base = (
                        draws.reshape(n_paths, *([1] * len(batch_shape)))
                        .expand(n_paths, *batch_shape)
                        .reshape(posterior.batch_shape)
                    )
                else:
                    from botorch.posteriors.gpytorch import GPyTorchPosterior

                    if not isinstance(posterior, GPyTorchPosterior):
                        raise IncompatibilityError(
                            f"Node '{node_name}' requires a GP or ensemble posterior "
                            "with deterministic base samples."
                        )
                    # W has only path and q axes. Candidate, reference and fantasy
                    # batch axes reuse it; changing their sizes must not change SAA.
                    draws = torch.randn(n_paths, q, device=X.device, dtype=X.dtype)
                    base = (
                        draws.reshape(n_paths, *([1] * len(batch_shape)), q)
                        .expand(n_paths, *batch_shape, q)
                        .reshape(posterior.base_sample_shape)
                    )
                node_samples = posterior.rsample_from_base_samples(
                    torch.Size([1]), base.unsqueeze(0)
                ).squeeze(0)
            expected_shape = (n_paths, *batch_shape, q, node_model.num_outputs)
            try:
                node_samples = node_samples.reshape(expected_shape)
            except RuntimeError as ex:
                raise IncompatibilityError(
                    f"Sampling node '{node_name}' produced shape "
                    f"{tuple(node_samples.shape)}, expected {expected_shape}."
                ) from ex
            samples[node_name] = node_samples

        return (
            torch.cat([samples[name] for name in self.dag.final_nodes], dim=-1),
            success_weights,
        )

    def _prepare_fixed_outputs(
        self,
        X: Tensor,
        fixed_outputs: Mapping[str, Tensor] | None,
        n_paths: int,
    ) -> dict[str, Tensor]:
        """Validate and broadcast observed node outputs to network paths.

        Args:
            X: Candidate configurations with shape ``(batch..., q, d)``.
            fixed_outputs: Optional observed node outputs keyed by node name.
                Values with a prefix of leading singleton axes are accepted.
            n_paths: Number of joint network sample paths.

        Returns:
            The fixed outputs expanded to ``(n_paths, batch..., q, 1)``.

        Raises:
            IncompatibilityError: If a fixed output refers to an unknown node
                or cannot be broadcast to the network-path shape.
        """
        fixed = {} if fixed_outputs is None else fixed_outputs
        unknown = set(fixed) - set(self._node_names)
        if unknown:
            raise IncompatibilityError(
                f"Fixed outputs provided for unknown nodes: {sorted(unknown)}."
            )

        target_shape = torch.Size([n_paths, *X.shape[:-2], X.shape[-2], 1])
        prepared: dict[str, Tensor] = {}
        for name, value in fixed.items():
            try:
                prepared[name] = value.to(X).expand(target_shape)
            except RuntimeError as ex:
                raise IncompatibilityError(
                    f"Fixed output for node '{name}' with shape "
                    f"{tuple(value.shape)} cannot be broadcast to "
                    f"{tuple(target_shape)}."
                ) from ex
        return prepared

    def get_node_model(self, node_name: str) -> Model:
        """Retrieve the fitted model for one node.

        Args:
            node_name: The DAG node whose model is requested.

        Returns:
            The already fitted BoTorch node model.

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

    def get_node_model_columns(self, node_name: str) -> tuple[str, ...]:  # noqa: DOC502
        """Retrieve the computational input columns of a node model.

        Args:
            node_name: The DAG node whose input order is requested.

        Returns:
            The computational input columns in model order.

        Raises:
            KeyError: If the node is not part of the DAG.
        """
        return self._node_model_columns[node_name]

    def make_node_fantasies(
        self,
        node: MultiStepNode,
        inputs: Tensor,
        num_fantasies: int,
        sampling_seed: int,
    ) -> MultiStepModel:
        """Create one network model with a batched fantasy node.

        BoTorch's ``fantasize`` API samples the selected node's predictive
        observation distribution and conditions a temporary local GP for each
        fantasy sample. The campaign model and all real training data remain
        untouched, and no fitting is performed. Each sample uses a deterministic
        BoTorch sampler seed shared across candidate actions.

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
            IncompatibilityError: If the node model cannot fantasize or the local
                inputs do not match its computational input shape.
        """
        from botorch.models.model import FantasizeMixin
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.settings import fast_pred_var

        node_model = self.get_node_model(node.name)
        if not isinstance(node_model, FantasizeMixin):
            raise IncompatibilityError(
                f"The fitted model of node '{node.name}' does not support fantasizing."
            )

        model_columns = self._node_model_columns[node.name]
        if inputs.shape[-2] != 1 or inputs.shape[-1] != len(model_columns):
            raise IncompatibilityError(
                f"The fantasy input of node '{node.name}' must have shape "
                f"(batch..., 1, {len(model_columns)}), but received "
                f"{tuple(inputs.shape)}."
            )

        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([num_fantasies]), seed=sampling_seed
        )
        with fast_pred_var(False):
            conditioned = node_model.fantasize(
                X=inputs, sampler=sampler, observation_noise=None, propagate_grads=True
            )
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
            fed_from_node_data=True,
            feasibility_models=self._feasibility_models,
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
            Hypothetical observations with shape ``(num_samples, num_points)``.

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

        node_model = self.get_node_model(node_name)
        with torch.no_grad():
            posterior = node_model.posterior(inputs.squeeze(-2), observation_noise=True)
            with torch.random.fork_rng():
                torch.manual_seed(sampling_seed)
                samples = posterior.rsample(torch.Size([num_samples]))
        if samples.shape != (num_samples, inputs.shape[0], node_model.num_outputs):
            raise IncompatibilityError(
                f"Sampling pending executions of node '{node_name}' produced shape "
                f"{tuple(samples.shape)}, expected a"
                f"joint q={inputs.shape[0]} axis and "
                f"{node_model.num_outputs} output(s)."
            )
        if node_model.num_outputs != 1:
            raise IncompatibilityError(
                f"Pending node observations require a single output, but node "
                f"'{node_name}' produces {node_model.num_outputs}."
            )
        return samples.squeeze(-1)

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
                ``(num_points, 1)``.

        Returns:
            A temporary network model conditioned on the observations.

        Raises:
            IncompatibilityError: If a node is unknown, a model cannot be
                conditioned, or an input or output shape is invalid.
        """
        if not observations:
            return self

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
            if outputs.shape not in {(inputs.shape[0],), (inputs.shape[0], 1)}:
                raise IncompatibilityError(
                    f"Hypothetical outputs of node '{name}' must have shape "
                    f"({inputs.shape[0]},) or ({inputs.shape[0]}, 1), but received "
                    f"{tuple(outputs.shape)}."
                )
            conditioned = node_model.condition_on_observations(
                inputs.squeeze(-2), outputs.reshape(-1, 1)
            )
            node_models[name] = cast(Model, conditioned)

        conditioned_model = self.__class__(
            dag=self.dag,
            num_network_samples=self.num_network_samples,
            sampling_seed=self.sampling_seed,
        )
        conditioned_model._publish(
            node_models,
            self._node_surrogates,
            fed_from_node_data=True,
            feasibility_models=self._feasibility_models,
        )
        return conditioned_model

    def _transform_local_inputs(
        self,
        inputs: pd.DataFrame,
        surrogate: Surrogate,
        model_columns: tuple[str, ...],
        model: Model,
        *,
        node_name: str,
    ) -> Tensor:
        """Transform experimental local inputs into fitted model inputs.

        Args:
            inputs: The node's own parameters and observed parent outputs in
                experimental representation.
            surrogate: The fitted surrogate owning the local fit search space.
            model_columns: The computational input columns in fitted model order.
            model: The fitted BoTorch model supplying device and dtype.
            node_name: The node name used in error messages.

        Returns:
            A computational input tensor in the fitted column order and with the
            model's device and dtype.

        Raises:
            IncompatibilityError: If the fitted search space is unavailable or the
                transformed inputs are inconsistent with the fitted model.
        """
        searchspace = surrogate._searchspace
        if searchspace is None:
            raise IncompatibilityError(
                f"The fitted surrogate for node '{node_name}' has no local fit "
                f"search space."
            )
        comp_rep = searchspace.transform(inputs)
        if tuple(comp_rep.columns) != model_columns:
            raise IncompatibilityError(
                f"The fitted search space of node '{node_name}' produces input "
                f"columns {list(comp_rep.columns)}, but the model uses "
                f"{list(model_columns)}."
            )

        reference = next(model.parameters())
        return torch.as_tensor(
            comp_rep.to_numpy(dtype=float).copy(),
            device=reference.device,
            dtype=reference.dtype,
        )

    def transform_node_inputs(self, node_name: str, inputs: pd.DataFrame) -> Tensor:
        """Transform experimental local inputs into fitted model inputs.

        Args:
            node_name: The DAG node whose inputs are transformed.
            inputs: The node's own parameters and observed parent outputs in
                experimental representation.

        Returns:
            A computational input tensor in the fitted column order and with the
            node model's device and dtype.

        Raises:
            IncompatibilityError: If the fitted surrogate is unavailable or the
                transformed inputs are inconsistent with the node model.
        """
        node_model = self.get_node_model(node_name)
        surrogate = self._node_surrogates.get(node_name)
        if not isinstance(surrogate, Surrogate):
            raise IncompatibilityError(
                f"The fitted surrogate for node '{node_name}' cannot transform "
                f"local inputs."
            )

        return self._transform_local_inputs(
            inputs,
            surrogate,
            self.get_node_model_columns(node_name),
            node_model,
            node_name=node_name,
        )

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


class _AcquisitionModelAdapter(Model):
    """Expose a network model without attrs recursion in stock acquisition setup."""

    def __init__(self, model: MultiStepModel) -> None:
        super().__init__()
        self._model = model

    @property
    @override
    def batch_shape(self) -> torch.Size:
        """Return the unbatched parameter shape of network models."""
        return torch.Size()

    @property
    @override
    def num_outputs(self) -> int:
        """Return the number of network outputs."""
        return self._model.num_outputs

    @override
    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any = None,
    ) -> Posterior:
        """Delegate posterior computation to the network model."""
        return self._model.posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
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
        """Delegate fitting; authoritative network fits are already a no-op."""
        self._model.fit(searchspace, objective, measurements)

    def to_botorch(self) -> Model:
        """Return a non-attrs BoTorch adapter around the fitted network model."""
        return _AcquisitionModelAdapter(self._model)


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
