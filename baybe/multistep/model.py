"""Single BoTorch model over a DAG of independently fitted node models.

The model owns one independently fitted model per DAG node and derives all node
input bookkeeping from the DAG itself. Network posteriors are formed by
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
from collections.abc import Mapping
from copy import copy, deepcopy
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from attrs import define, field, setters
from attrs.validators import ge, instance_of
from pandas.api.types import is_numeric_dtype
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
from baybe.utils.validation import validate_target_input

from ._columns import FEASIBILITY_LABEL
from .dag import MultiStepDAG
from .data import (
    _extract_feasibility_values,
    _get_node,
    _make_local_searchspace,
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
    auxiliary_names = tuple(
        dict.fromkeys(
            condition.column
            for node in dag.nodes
            for condition in node.feasibility_conditions
            if condition.column != node.output
        )
    )
    missing = [
        name
        for name in (*expected_parameters, *output_names, *auxiliary_names)
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


@define(eq=False, repr=False, slots=False, on_setattr=setters.NO_OP)
class _MultiStepModelBase(_ModelBase):
    """Shared implementation of the multi-step network model.

    The model owns one independently fitted model per node and composes them by
    forward sampling. Descendants see the sampled outputs of their parents, so
    the resulting posterior reflects the full network propagation rather than
    independent per-node predictions. Node models are fitted from per-node
    training frames or from a complete measurement frame, and fitted state is
    published atomically. A later stock :meth:`fit` call always consumes its
    supplied complete measurements, regardless of how the current models were
    fitted.

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
        outside the multi-step layer. No fitting is performed, and the provided
        models are used until a subsequent fit replaces them.

        Args:
            dag: The DAG topology.
            node_models: Mapping covering every DAG node.
            num_network_samples: Number of network paths used for posterior moments.
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
        model._publish(node_models, {})
        return model

    def _publish(
        self,
        node_models: Mapping[str, Model],
        node_surrogates: Mapping[str, SurrogateProtocol],
        *,
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
        import torch

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

        node_frames = {
            node.name: measurements[[*node.model_input_names, node.output]].copy()
            for node in self.dag.nodes
        }
        feasibility_frames: dict[str, pd.DataFrame] = {}
        for node in self.dag.nodes:
            if not node.feasibility_conditions:
                continue
            auxiliary = tuple(
                dict.fromkeys(
                    condition.column
                    for condition in node.feasibility_conditions
                    if condition.column != node.output
                )
            )
            raw = measurements[
                [*node.model_input_names, node.output, *auxiliary]
            ].copy()
            labels = _extract_feasibility_values(node, raw)
            frame = raw[list(node.model_input_names)].copy()
            frame[FEASIBILITY_LABEL] = labels
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
        template: SurrogateProtocol | _FeasibilityModelProtocol,
        label: str,
    ) -> SurrogateProtocol | _FeasibilityModelProtocol:
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

        """
        measurements = frame[[*node.model_input_names, label]].copy()
        fit_searchspace = _make_local_searchspace(node, measurements)
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
            surrogate = cast(
                SurrogateProtocol,
                self._fit_local_model(
                    node,
                    df_node,
                    target,
                    self._node_surrogates.get(node_name, node.surrogate_model),
                    node.output,
                ),
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
            new_feasibility[node_name] = cast(
                _FeasibilityModelProtocol,
                self._fit_local_model(node, frame, target, template, FEASIBILITY_LABEL),
            )

        # Publish atomically: the previous fit stays intact on any error.
        self._publish(
            new_models,
            new_surrogates,
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
        _validate_node_inputs(node, inputs)
        if node_name not in self._feasibility_models:
            raise ModelNotTrainedError(
                f"Node '{node_name}' has no fitted feasibility model."
            )
        model = self._feasibility_models[node_name]
        surrogate = cast(Surrogate, model)
        tensor = self._transform_local_inputs(
            inputs,
            surrogate,
            surrogate.to_botorch(),
            node_name=node_name,
        )
        import torch

        with torch.no_grad():
            probabilities = model.posterior_probabilities(tensor).detach().cpu().numpy()
        return pd.Series(probabilities, index=inputs.index, name="success_probability")

    def _predict_feasibility_tensor(
        self, node_name: str, inputs_comp: Tensor
    ) -> Tensor:
        """Predict local action success while retaining candidate gradients.

        ``inputs_comp`` uses the output model's computational representation.
        The candidate columns are reordered by name into the independently
        retained feasibility fit space, whose input transform and bounds remain
        authoritative for this prediction.

        Args:
            node_name: The configured DAG node.
            inputs_comp: Computational inputs with shape ``(batch..., 1, d)``.

        Returns:
            One success probability per batch item, without the singleton q-axis.

        Raises:
            IncompatibilityError: If the node, input shape, fitted layout, or
                predicted probabilities are invalid.
            ModelNotTrainedError: If the configured feasibility model is absent.
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
        try:
            feasibility_model = self._feasibility_models[node_name]
        except KeyError as ex:
            raise ModelNotTrainedError(
                f"Node '{node_name}' has no fitted feasibility model."
            ) from ex

        surrogate = cast(Surrogate, feasibility_model)
        searchspace = surrogate._searchspace
        if searchspace is None:
            raise IncompatibilityError(
                f"The feasibility fit space for node '{node_name}' is unavailable."
            )
        source_columns = self.get_node_model_columns(node_name)
        target_columns = searchspace.comp_rep_columns
        source_indices = {column: index for index, column in enumerate(source_columns)}
        missing = sorted(set(target_columns) - set(source_indices))
        if missing or len(source_columns) != inputs_comp.shape[-1]:
            raise IncompatibilityError(
                f"The fitted input layouts for node '{node_name}' are incompatible; "
                f"missing feasibility columns: {missing}."
            )

        import torch

        indices = torch.tensor(
            [source_indices[column] for column in target_columns],
            dtype=torch.long,
            device=inputs_comp.device,
        )
        # Each action needs a marginal probability; retain the singleton q-axis
        # so candidate batches do not become a joint covariance calculation.
        candidates = inputs_comp.index_select(-1, indices)
        reference = next(surrogate.to_botorch().parameters())
        candidates = candidates.to(reference)
        probabilities = feasibility_model.posterior_probabilities(candidates)
        expected_shape = inputs_comp.shape[:-1]
        if probabilities.numel() != math.prod(expected_shape):
            raise IncompatibilityError(
                f"The feasibility model for node '{node_name}' returned "
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
                f"The feasibility model for node '{node_name}' returned values "
                "outside the probability interval [0, 1]."
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
        node = self.dag.node_by_name[node_name]
        surrogate = self._node_surrogates.get(node_name)
        if isinstance(surrogate, Surrogate) and surrogate._searchspace is not None:
            return surrogate._searchspace.comp_rep_columns

        # Models supplied through ``from_node_models`` have no retained BayBE fit
        # search space. Preserve their historical input-order contract only for
        # that advanced entry point.
        if node.inputs:
            discrete = node.searchspace.discrete.comp_rep_columns
            continuous = node.searchspace.continuous.comp_rep_columns
        else:
            discrete = ()
            continuous = ()
        return (*discrete, *sorted((*continuous, *node.parent_output_names)))

    def make_node_fantasies(
        self,
        node: MultiStepNode,
        inputs: Tensor,
        num_fantasies: int,
        sampling_seed: int,
    ) -> MultiStepModel:
        """Create one network model with a batched fantasy node.

        BoTorch samples the selected node's predictive observation distribution
        and conditions a temporary local model for each fantasy sample. The
        campaign model and real training data remain untouched, and no fitting
        is performed.

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
        from botorch.sampling import get_sampler
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
                conditioned = node_model.condition_on_observations(
                    X=inputs, Y=observations
                )
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

        import torch
        from botorch.posteriors.torch import TorchPosterior
        from botorch.sampling import get_sampler

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
            if outputs.shape not in {(inputs.shape[0],), (inputs.shape[0], 1)}:
                raise IncompatibilityError(
                    f"Hypothetical outputs of node '{name}' must have shape "
                    f"({inputs.shape[0]},) or ({inputs.shape[0]}, 1), but received "
                    f"{tuple(outputs.shape)}."
                )
            try:
                conditioned = node_model.condition_on_observations(
                    inputs.squeeze(-2), outputs.reshape(-1, 1)
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
                columns=list(node.model_input_names),
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
        surrogate: Surrogate,
        model: Model,
        *,
        node_name: str,
    ) -> Tensor:
        """Transform experimental local inputs into fitted model inputs.

        Args:
            inputs: The node's own parameters and observed parent outputs in
                experimental representation.
            surrogate: The fitted surrogate owning the local fit search space.
            model: The fitted BoTorch model supplying device and dtype.
            node_name: The node name used in error messages.

        Returns:
            A computational input tensor in the fitted column order and with the
            model's device and dtype.

        Raises:
            IncompatibilityError: If the fitted search space is unavailable.
        """
        searchspace = surrogate._searchspace
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

        n_input = len(self.dag.full_searchspace.comp_rep_columns)
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
    snapshot._node_surrogates = model._node_surrogates.copy()
    snapshot._feasibility_models = model._feasibility_models.copy()
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
        model: The fitted single-output node model.
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
    from botorch.sampling import get_sampler

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
    paths: list[Tensor] = []
    weights: list[Tensor] = []
    for path_seeds in base_samples:
        samples: dict[str, Tensor] = {}
        success_weights = X.new_ones(X.shape[:-1])
        for node_index, node_name in enumerate(model._node_names):
            node = model.dag.node_by_name[node_name]
            features: dict[str, Tensor] = {}
            for column in node._own_comp_rep_columns:
                try:
                    features[column] = X[..., full_index[column]].unsqueeze(-1)
                except KeyError as ex:
                    raise IncompatibilityError(
                        f"The computational-representation column '{column}' of node "
                        f"'{node_name}' is missing from the DAG-wide search space."
                    ) from ex
            for parent in node.parents:
                features[parent.output] = samples[parent.name].to(X)

            columns = model.get_node_model_columns(node_name)
            missing = [column for column in columns if column not in features]
            if missing:
                raise IncompatibilityError(
                    f"The node-model features of node '{node_name}' are incomplete: "
                    f"{missing}."
                )
            node_x = torch.cat([features[column] for column in columns], dim=-1)

            if include_feasibility and node.feasibility_conditions:
                probabilities = model._predict_feasibility_tensor(
                    node_name, node_x.reshape(-1, 1, node_x.shape[-1])
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

        The model owns one independently fitted model per node and composes them
        by forward sampling. Descendants see the sampled outputs of their
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
